# PRD: LLM provider fallback chain

## Objective

Replace the single-provider `LLM_PROVIDER` setting with an ordered fallback chain, `LLM_PROVIDERS` (a comma-separated list, default `gemini,claude,openai`), so that if a provider fails for any reason -- overloaded, out of quota, network error, a missing API key, or any other exception -- the next provider in the chain is tried immediately within the same turn instead of ending the call or skipping the reply. OpenAI becomes a third first-class reply-generating provider alongside Claude and Gemini, since today it's only used for TTS/STT/moderation.

## Context

Today `config.LLM_PROVIDER` (`claude_phone/config.py:35`) picks exactly one LLM for the whole run, and `Conversation.__init__` binds `self._ask` to that single provider's function once, at call start (`claude_phone/assistant.py:85-92`, via the `_PROVIDERS` dict at `assistant.py:76-79`). If that one provider raises `QuotaExceededError` or `ModelOverloadedError`, `main.py`'s `handle_call` is the only resilience mechanism: it apologizes and either ends the call (`_apologize_and_end_call`, on quota exhaustion) or apologizes and skips just that turn (`_apologize_and_continue`, on a Gemini 503) -- see `main.py:86-94`. The `ModelOverloadedError` path only exists because of Gemini's existing 503 handling in `_ask_gemini` (`assistant.py:69-72`, added in commit `370b6f5`); `_ask_claude` has no equivalent overload mapping today. A transient overload or an exhausted quota currently interrupts the call rather than transparently trying another model.

OpenAI is already a mandatory dependency -- `OPENAI_API_KEY` is required unconditionally (`config.py:7`) because `tts.py` needs it for every reply's speech synthesis, and `safety.py` needs it for moderation. Adding OpenAI as a *reply*-generating provider therefore needs no new required key, just a new chat-completions call path.

## Requirements

1. `LLM_PROVIDERS` (comma-separated, default `"gemini,claude,openai"`) replaces `LLM_PROVIDER` and defines an ordered fallback chain of providers tried per turn.
2. OpenAI is a third reply-generating provider, via a new `_ask_openai` function using OpenAI's chat completions API and a new `OPENAI_CHAT_MODEL` config value (default `gpt-4o-mini`).
3. For each turn, providers are tried in the configured order. A provider raising any exception -- overloaded, out of quota, network failure, malformed response, anything -- is logged and the next provider in the chain is tried immediately, within the same turn, rather than surfacing an apology or ending the turn.
4. This includes a provider whose API key isn't set, which raises `RuntimeError` from its `_get_*_client()` helper -- that failure is treated like any other, so a configured chain doesn't require every listed provider's key to actually be set; unconfigured entries are simply skipped every turn.
5. If every provider in the chain fails for a turn, the call ends gracefully with a spoken apology, the same way an unrecoverable failure ends the call today -- never a crash or a silent hang.
6. The greeting played at pickup uses the first provider in the chain's pre-recorded asset. The per-turn console log names whichever provider actually produced that reply, not a fixed configured provider.
7. An invalid provider name anywhere in `LLM_PROVIDERS`, or an empty chain, raises a clear `ValueError` at call start, the same way an unknown `LLM_PROVIDER` does today.
8. The `except ModelOverloadedError` path in `main.py`'s `handle_call`, and its `_apologize_and_continue` helper, are removed -- they're only ever reachable via Gemini's 503 mapping, which the fallback chain now handles internally before it can bubble up.
9. Child-safety moderation (`safety.is_child_safe`) continues to run on whichever reply the chain ultimately returns, unchanged.

## Acceptance Criteria

- With the default `.env` (`LLM_PROVIDERS=gemini,claude,openai`), a normal call is answered by Gemini, and the console logs `gemini: <reply>`.
- Temporarily forcing Gemini to fail (e.g. an invalid `GEMINI_API_KEY`) results in that turn falling through to Claude within the same turn -- the caller just hears a normal (if slightly slower) reply, `[fallback] gemini failed (...); trying next provider` appears in the logs, and the reply is logged as `claude: <reply>`.
- Setting `LLM_PROVIDERS=openai` alone routes replies through OpenAI's chat completions API and speaks a normal reply.
- Forcing every provider in the chain to fail (e.g. invalid keys for all three) results in a spoken "none of my language models are available" apology and the call ending cleanly, not a crash or hang.
- Setting `LLM_PROVIDERS=bogus,claude` raises a clear `ValueError` naming the bad provider, at the start of a call.
- The pre-recorded greeting still plays with zero synthesis latency, using the asset for the first provider in the chain.

## Design

- `config.py`: replace `LLM_PROVIDER` with the chain setting, following the existing `os.environ.get` + inline-comment convention used for `LLM_PROVIDER`/`STT_PROVIDER`:
  ```python
  # Comma-separated fallback chain, tried in order for each reply; if a provider
  # raises any exception (overloaded, out of quota, network error, missing key,
  # etc.) the next one is tried within the same turn. Any combination/order of
  # "claude" (Anthropic), "gemini" (Google), "openai" (OpenAI).
  LLM_PROVIDERS = os.environ.get("LLM_PROVIDERS", "gemini,claude,openai")
  LLM_PROVIDER_CHAIN = [p.strip() for p in LLM_PROVIDERS.split(",") if p.strip()]
  CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
  GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")  # only used when "gemini" is in LLM_PROVIDERS
  OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # only used when "openai" is in LLM_PROVIDERS, for replies (separate from TTS_MODEL/STT_MODEL)
  ```
  Add an `"openai"` entry to `GREETINGS`:
  ```python
  GREETINGS = {
      "claude": "Hi, this is Claude. What's your question?",
      "gemini": "Hi, this is Gemini. What's your question?",
      "openai": "Hi, this is ChatGPT. What's your question?",
  }
  ```

- `errors.py`: add a new exception for the exhausted-chain case, alongside the existing `QuotaExceededError`/`ModelOverloadedError`:
  ```python
  class AllProvidersFailedError(Exception):
      """Raised when every provider in the LLM_PROVIDERS fallback chain fails for one turn."""

      def __init__(self, attempts: list[tuple[str, Exception]]):
          self.attempts = attempts
          detail = "; ".join(f"{name}: {err}" for name, err in attempts)
          super().__init__(f"all LLM providers failed: {detail}")
  ```

- `assistant.py`: add an OpenAI reply path, following the eager-client convention already used in `tts.py`/`stt.py` (no lazy "key not set" check needed, since `OPENAI_API_KEY` is already mandatory):
  ```python
  from openai import InternalServerError, OpenAI, RateLimitError as OpenAIRateLimitError
  ...
  _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


  def _ask_openai(messages: list[dict]) -> str:
      try:
          response = _openai_client.chat.completions.create(
              model=config.OPENAI_CHAT_MODEL,
              messages=[{"role": "system", "content": config.SYSTEM_PROMPT}] + messages,
          )
      except OpenAIRateLimitError as e:
          raise QuotaExceededError("OpenAI", e.message) from e
      except InternalServerError as e:
          raise ModelOverloadedError("OpenAI", e.message) from e
      return response.choices[0].message.content.strip()
  ```
  Extend `_PROVIDERS`:
  ```python
  _PROVIDERS = {
      "claude": _ask_claude,
      "gemini": _ask_gemini,
      "openai": _ask_openai,
  }
  ```
  Rewrite `Conversation` to walk the chain instead of binding to one function, tracking which provider actually answered:
  ```python
  class Conversation:
      """Message history for a single phone call, reset on hangup."""

      def __init__(self):
          self._messages = []
          self.last_provider = None
          try:
              self._chain = [(name, _PROVIDERS[name]) for name in config.LLM_PROVIDER_CHAIN]
          except KeyError as e:
              raise ValueError(
                  f"Unknown provider {e.args[0]!r} in LLM_PROVIDERS (expected some of {sorted(_PROVIDERS)})"
              ) from None
          if not self._chain:
              raise ValueError("LLM_PROVIDERS must name at least one provider")

      def ask(self, text: str) -> str:
          self._messages.append({"role": "user", "content": text})
          reply, attempts = None, []
          for name, ask_fn in self._chain:
              try:
                  reply = ask_fn(self._messages)
                  self.last_provider = name
                  break
              except Exception as e:
                  print(f"[fallback] {name} failed ({e}); trying next provider")
                  attempts.append((name, e))
          if reply is None:
              raise AllProvidersFailedError(attempts)
          if config.CHILD_SAFETY_ENABLED and not safety.is_child_safe(reply):
              reply = config.CHILD_SAFETY_FALLBACK_REPLY
          self._messages.append({"role": "assistant", "content": reply})
          return reply
  ```
  The broad `except Exception` is deliberate: quota/overload errors, network errors, a missing-key `RuntimeError`, or anything else from a provider all mean "try the next one," per Requirement 3/4.

- `main.py`: use the chain instead of the single provider for the greeting and per-turn logging, and replace the now-unreachable overloaded path with an all-failed path:
  ```python
  greeting_audio = _load_greeting_audio(config.LLM_PROVIDER_CHAIN[0])
  ```
  ```python
  print(f"{conversation.last_provider}: {reply}")
  ```
  Remove `_apologize_and_continue` and its `except ModelOverloadedError` branch; add:
  ```python
  def _apologize_and_end_call_all_failed(e: AllProvidersFailedError, trigger) -> None:
      print(f"[fallback] {e}")
      message = "Sorry, none of my language models are available right now. Please try again later. Goodbye."
      try:
          audio_io.play_with_comfort_noise(lambda: tts.synthesize_stream(message), tts.PCM_SAMPLE_RATE, trigger)
      except QuotaExceededError:
          print("[fallback] couldn't speak the apology either -- TTS quota is also exhausted")
  ```
  and, in `handle_call`'s try/except:
  ```python
  except QuotaExceededError as e:
      _apologize_and_end_call(e, trigger)
      break
  except AllProvidersFailedError as e:
      _apologize_and_end_call_all_failed(e, trigger)
      break
  ```
  `QuotaExceededError` handling is unchanged and still needed -- it's also raised by `tts.py`/`stt.py` for TTS/STT quota, which has nothing to do with the LLM chain.

- New asset: generate and commit `assets/greetings/openai.pcm` via `python scripts/generate_greeting.py openai`. No changes needed to `scripts/generate_greeting.py` itself -- it already iterates `config.GREETINGS`, so the new key is picked up automatically.

- `README.md`: update the intro line and "How it works" step 2 to mention three providers instead of two, and update the Configuration table: replace the `LLM_PROVIDER` row with an `LLM_PROVIDERS` row describing the chain/fallback behavior, and add an `OPENAI_CHAT_MODEL` row next to `CLAUDE_MODEL`/`GEMINI_MODEL`.

- `.env.example`: replace the `LLM_PROVIDER` block with the chain default and the new model override, keeping the existing per-key comments (they still correctly describe when each key is *used*, just no longer gate a single exclusive choice):
  ```
  # Comma-separated fallback chain, tried in order for each reply; falls through
  # to the next provider on any failure (overloaded, out of quota, network error,
  # missing key, etc.) within the same turn.
  LLM_PROVIDERS=gemini,claude,openai

  # Required only when "claude" is in LLM_PROVIDERS
  ANTHROPIC_API_KEY=
  # CLAUDE_MODEL=claude-sonnet-5

  # Required only when "gemini" is in LLM_PROVIDERS. Get one at https://aistudio.google.com/apikey
  GEMINI_API_KEY=
  # GEMINI_MODEL=gemini-flash-latest

  # Only used when "openai" is in LLM_PROVIDERS, for replies
  # OPENAI_CHAT_MODEL=gpt-4o-mini
  ```

## Tasks

1. Replace `LLM_PROVIDER` with `LLM_PROVIDERS`/`LLM_PROVIDER_CHAIN` in `config.py`, add `OPENAI_CHAT_MODEL`, and add the `"openai"` greeting to `GREETINGS`.
2. Add `AllProvidersFailedError` to `errors.py`.
3. Add `_ask_openai` (and its client) to `assistant.py`, extend `_PROVIDERS`, and rewrite `Conversation` to walk the chain and track `last_provider`.
4. Update `main.py`: chain-based greeting lookup, `last_provider`-based logging, remove the dead overloaded-path/`_apologize_and_continue`, add the all-failed path.
5. Generate and commit `assets/greetings/openai.pcm`.
6. Update `README.md`'s intro/"How it works"/Configuration table and `.env.example`.
7. Manually test: default chain answers via Gemini with correct logging; a forced Gemini failure falls through to Claude within the same turn with the expected `[fallback]` log line; `LLM_PROVIDERS=openai` alone works end-to-end; forcing all three to fail apologizes and ends the call cleanly; `LLM_PROVIDERS=bogus,claude` raises `ValueError` at call start; the greeting still plays with zero synthesis latency.

## Deployment steps

- Pure code change plus one new binary asset (`assets/greetings/openai.pcm`) -- no new required env vars.
- Existing single-key deployments (e.g. only `ANTHROPIC_API_KEY` set, no `GEMINI_API_KEY`) keep working unchanged after pulling: the default chain's first entry (`gemini`) fails instantly with a clear `RuntimeError` (missing key, no network call attempted) and falls through to `claude` within the same turn, so there's no perceptible latency added and no env changes required to deploy safely.
- Pull latest code and restart the `claude-phone` service; no new dependencies.
- To take advantage of the full three-way chain, optionally add `GEMINI_API_KEY` in `.env`; reorder, trim, or extend `LLM_PROVIDERS` to change provider preference or drop one you don't want tried.
