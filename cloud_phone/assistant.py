"""LLM replies via Anthropic Claude, Google Gemini, or OpenAI.

LLM_PROVIDERS is a comma-separated fallback chain (e.g. "gemini,openai,claude");
providers are tried in order per reply, falling through to the next one on any
exception. Conversation history is kept in a provider-neutral form (role
"user"/"assistant" + text) and translated to each SDK's native message format
at call time.
"""

from anthropic import Anthropic, RateLimitError as AnthropicRateLimitError
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from openai import InternalServerError as OpenAIInternalServerError, OpenAI, RateLimitError as OpenAIRateLimitError

from . import config, safety
from .errors import AllProvidersFailedError, ModelOverloadedError, QuotaExceededError

_anthropic_client = None
_gemini_client = None
_openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (required when 'claude' is in LLM_PROVIDERS)")
        _anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set (required when 'gemini' is in LLM_PROVIDERS)")
        # Pin v1beta: google-genai >= 3 defaults to v1, where aliases like
        # "gemini-flash-latest" don't exist and generate_content 404s.
        _gemini_client = genai.Client(
            api_key=config.GEMINI_API_KEY,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
    return _gemini_client


def _ask_claude(messages: list[dict]) -> str:
    try:
        response = _get_anthropic_client().messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=config.SYSTEM_PROMPT,
            messages=messages,
        )
    except AnthropicRateLimitError as e:
        raise QuotaExceededError("Claude (Anthropic)", e.message) from e
    return "".join(block.text for block in response.content if block.type == "text")


def _ask_gemini(messages: list[dict]) -> str:
    contents = [
        types.Content(
            role="model" if message["role"] == "assistant" else "user",
            parts=[types.Part(text=message["content"])],
        )
        for message in messages
    ]
    try:
        response = _get_gemini_client().models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=config.SYSTEM_PROMPT),
        )
    except genai_errors.ClientError as e:
        if e.code == 429:
            raise QuotaExceededError("Gemini (Google)", e.message) from e
        raise
    except genai_errors.ServerError as e:
        if e.code == 503:
            raise ModelOverloadedError("Gemini (Google)", e.message) from e
        raise
    return response.text.strip()


def _ask_openai(messages: list[dict]) -> str:
    try:
        response = _openai_client.chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{"role": "system", "content": config.SYSTEM_PROMPT}] + messages,
        )
    except OpenAIRateLimitError as e:
        raise QuotaExceededError("OpenAI", e.message) from e
    except OpenAIInternalServerError as e:
        raise ModelOverloadedError("OpenAI", e.message) from e
    return response.choices[0].message.content.strip()


_PROVIDERS = {
    "claude": _ask_claude,
    "gemini": _ask_gemini,
    "openai": _ask_openai,
}


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
