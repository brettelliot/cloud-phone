import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import audio_io, config, connectivity, safety, stt, tts
from .assistant import Conversation
from .errors import AllProvidersFailedError, QuotaExceededError
from .trigger import get_trigger

# When stdout isn't a terminal (e.g. piped to journald under systemd), Python
# fully block-buffers it, so prints only reach the journal in bursts on exit
# instead of live. Force line buffering so `journalctl -f` shows lines as they
# happen. This is independent of PYTHONUNBUFFERED so it works regardless of
# how the process is launched.
sys.stdout.reconfigure(line_buffering=True)

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "greetings"
OFFLINE_AUDIO_PATH = ASSETS_DIR.parent / "offline.pcm"


def _load_greeting_audio(provider: str) -> bytes:
    """Loads the pre-recorded greeting for `provider` so it can be played with zero
    TTS latency at pickup. Generate/refresh it with scripts/generate_greeting.py.
    """
    path = ASSETS_DIR / f"{provider}.pcm"
    if not path.exists():
        raise RuntimeError(
            f"Missing greeting asset for provider {provider!r}: {path}. "
            f"Run `python scripts/generate_greeting.py {provider}` to generate it."
        )
    return path.read_bytes()


def _load_offline_audio() -> bytes | None:
    """Loads the pre-recorded "I'm offline" clip. It has to be pre-recorded because
    TTS needs the internet too. Missing is tolerated (with a warning) so a stale
    checkout still runs; offline calls then just end silently like before.
    """
    if not OFFLINE_AUDIO_PATH.exists():
        print(
            f"[offline] warning: {OFFLINE_AUDIO_PATH} is missing, so the phone can't say it's offline. "
            f"Run `python scripts/generate_greeting.py offline` to generate it."
        )
        return None
    return OFFLINE_AUDIO_PATH.read_bytes()


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    print(f"[timing] {label}: {time.perf_counter() - start:.2f}s")


def _apologize_and_end_call(e: QuotaExceededError, trigger) -> None:
    print(f"[quota] {e}")
    message = f"Sorry, I've hit my usage quota for {e.provider}. Please try again later. Goodbye."
    try:
        audio_io.play_with_comfort_noise(lambda: tts.synthesize_stream(message), tts.PCM_SAMPLE_RATE, trigger)
    except QuotaExceededError:
        print("[quota] couldn't speak the apology either -- TTS quota is also exhausted")


def _apologize_and_end_call_all_failed(e: AllProvidersFailedError, trigger) -> None:
    print(f"[fallback] {e}")
    message = "Sorry, none of my language models are available right now. Please try again later. Goodbye."
    try:
        audio_io.play_with_comfort_noise(lambda: tts.synthesize_stream(message), tts.PCM_SAMPLE_RATE, trigger)
    except QuotaExceededError:
        print("[fallback] couldn't speak the apology either -- TTS quota is also exhausted")


def _say_offline(trigger, offline_audio: bytes | None) -> None:
    print("[offline] no internet connection")
    if offline_audio:
        audio_io.play_pcm_stream([offline_audio], tts.PCM_SAMPLE_RATE, trigger)


def _wait_for_hangup(trigger) -> None:
    """After the call has been ended on our side (offline, quota, ...), wait for the
    handset to go back down. Otherwise a still-lifted handset counts as a fresh
    pickup and the greeting immediately plays again.
    """
    while not trigger.is_hung_up():
        time.sleep(0.1)


def handle_call(trigger, greeting_audio: bytes, offline_audio: bytes | None) -> None:
    if not connectivity.is_online():
        _say_offline(trigger, offline_audio)
        _wait_for_hangup(trigger)
        print("Call ended.\n")
        return

    conversation = Conversation()
    with timed("greeting playback"):
        audio_io.play_pcm_stream([greeting_audio], tts.PCM_SAMPLE_RATE, trigger)

    while not trigger.is_hung_up():
        with timed("record (includes caller's speaking time)"):
            samples = audio_io.record_until_silence(trigger)

        if trigger.poll_hung_up():
            break

        wav_bytes = audio_io.audio_to_wav_bytes(samples)
        result = {"heard": None, "reply": None}

        def produce():
            with timed("stt"):
                result["heard"] = stt.transcribe(wav_bytes)
            if not result["heard"] or trigger.poll_hung_up():
                return
            with timed("llm"):
                result["reply"] = conversation.ask(result["heard"])
            if trigger.poll_hung_up():
                return
            with timed("tts:synthesize"):
                yield from tts.synthesize_stream(result["reply"])

        try:
            with timed("reply (wait+playback)"):
                audio_io.play_with_comfort_noise(produce, tts.PCM_SAMPLE_RATE, trigger)
        except Exception as e:
            # Check connectivity first: offline, the TTS apologies below would fail too.
            if not connectivity.is_online():
                print(f"[offline] turn failed: {e!r}")
                _say_offline(trigger, offline_audio)
            elif isinstance(e, QuotaExceededError):
                _apologize_and_end_call(e, trigger)
            elif isinstance(e, AllProvidersFailedError):
                _apologize_and_end_call_all_failed(e, trigger)
            else:
                raise
            _wait_for_hangup(trigger)
            break

        heard = result["heard"]
        if not heard:
            continue
        print(f"You said: {heard}")

        reply = result["reply"]
        if reply is None:
            break
        print(f"{conversation.last_provider}: {reply}")

    print("Call ended.\n")


def main() -> None:
    trigger = get_trigger(config.PHONE_TRIGGER, config.HOOK_GPIO_PIN)
    greeting_audio = _load_greeting_audio(config.LLM_PROVIDER_CHAIN[0])
    offline_audio = _load_offline_audio()
    with timed("warmup"):
        tts.warm_up()
        if config.STT_PROVIDER == "openai":
            stt.warm_up()
        if config.CHILD_SAFETY_ENABLED:
            safety.warm_up()
    print(f"Ready. Using '{config.PHONE_TRIGGER}' trigger.")
    while True:
        trigger.wait_for_pickup()
        handle_call(trigger, greeting_audio, offline_audio)


if __name__ == "__main__":
    main()
