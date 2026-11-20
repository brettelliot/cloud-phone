import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# Only required for their respective providers; validated at point of use,
# not here, so switching providers doesn't force you to set unused keys.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WISPR_FLOW_API_KEY = os.environ.get("WISPR_FLOW_API_KEY", "")

PHONE_TRIGGER = os.environ.get("PHONE_TRIGGER", "keyboard")
HOOK_GPIO_PIN = int(os.environ.get("HOOK_GPIO_PIN", "17"))

INPUT_DEVICE = os.environ.get("INPUT_DEVICE") or None
OUTPUT_DEVICE = os.environ.get("OUTPUT_DEVICE") or None
OUTPUT_VOLUME = float(os.environ.get("OUTPUT_VOLUME", "1.5"))

# Quiet R2D2-style beeps/chirps played while waiting for STT/LLM/TTS instead of dead air.
COMFORT_NOISE_ENABLED = os.environ.get("COMFORT_NOISE_ENABLED", "true").lower() != "false"
COMFORT_NOISE_VOLUME = float(os.environ.get("COMFORT_NOISE_VOLUME", "0.11"))  # before OUTPUT_VOLUME scaling

# Hard technical backstop: every LLM reply is checked against OpenAI's moderation
# API before being spoken; if flagged (or if the check itself errors), the reply
# is replaced with CHILD_SAFETY_FALLBACK_REPLY. Independent of LLM_PROVIDERS.
CHILD_SAFETY_ENABLED = os.environ.get("CHILD_SAFETY_ENABLED", "true").lower() != "false"
CHILD_SAFETY_FALLBACK_REPLY = os.environ.get(
    "CHILD_SAFETY_FALLBACK_REPLY",
    "Let's talk about something else -- what else are you curious about?",
)

# Comma-separated fallback chain, tried in order for each reply; if a provider
# raises any exception (overloaded, out of quota, network error, missing key,
# etc.) the next one is tried within the same turn. Any combination/order of
# "claude" (Anthropic), "gemini" (Google), "openai" (OpenAI).
LLM_PROVIDERS = os.environ.get("LLM_PROVIDERS", "gemini,openai,claude")
LLM_PROVIDER_CHAIN = [p.strip() for p in LLM_PROVIDERS.split(",") if p.strip()]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")  # only used when "gemini" is in LLM_PROVIDERS
GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "15000"))  # per-request; no SDK retries, the fallback chain handles failure
OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # only used when "openai" is in LLM_PROVIDERS, for replies (separate from TTS_MODEL/STT_MODEL)
TTS_MODEL = os.environ.get("TTS_MODEL", "tts-1")
TTS_VOICE = os.environ.get("TTS_VOICE", "alloy")

# "wispr" (Wispr Flow REST API) or "openai" (Whisper)
STT_PROVIDER = os.environ.get("STT_PROVIDER", "wispr")
STT_MODEL = os.environ.get("STT_MODEL", "whisper-1")  # only used when STT_PROVIDER=openai
WISPR_FLOW_LANGUAGE = os.environ.get("WISPR_FLOW_LANGUAGE", "en")

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = float(os.environ.get("SILENCE_THRESHOLD", "0.02"))
SILENCE_DURATION = float(os.environ.get("SILENCE_DURATION", "1.5"))
MAX_RECORDING_SECONDS = float(os.environ.get("MAX_RECORDING_SECONDS", "30"))

GREETINGS = {
    "claude": "Hi, this is Claude. What's your question?",
    "gemini": "Hi, this is Gemini. What's your question?",
    "openai": "Hi, this is ChatGPT. What's your question?",
}

# Pre-recorded (assets/offline.pcm) because live TTS needs the internet too.
# Regenerate with `python scripts/generate_greeting.py offline` after changing it.
OFFLINE_MESSAGE = "I'm offline. Please connect me to a wifi network so I can answer your questions."

# Checked at pickup and whenever a turn fails: a TCP connect to this host:443.
ONLINE_CHECK_HOST = os.environ.get("ONLINE_CHECK_HOST", "api.openai.com")
ONLINE_CHECK_TIMEOUT = float(os.environ.get("ONLINE_CHECK_TIMEOUT", "3"))

SYSTEM_PROMPT = (
    "You are the voice on the other end of a phone call. Keep replies short, "
    "conversational, and natural to hear spoken aloud -- a sentence or two "
    "unless the caller clearly wants more detail. No markdown, lists, or "
    "headers, since this is read aloud, not read on screen.\n\n"
    "This phone can be answered by a child of any age, so every reply must be "
    "G-rated: no profanity, sexual content, graphic violence, or other mature "
    "themes. If a caller asks about something inappropriate or sensitive for a "
    "child (e.g. violence, drugs, sexual topics, self-harm), don't lecture or "
    "refuse at length -- briefly and kindly steer away from it, for example by "
    "suggesting they ask a parent or grown-up about that one, then offer to "
    "help with something else. Stay warm and friendly, like a favorite aunt or "
    "uncle, not a disclaimer machine. This doesn't mean avoiding real topics "
    "that merely touch on serious subjects in a factual, age-appropriate way -- "
    "questions like why dinosaurs went extinct, how predators hunt, or how "
    "volcanoes erupt should still get normal, straightforward answers. The bar "
    "is G-rated content, not avoiding every topic that isn't strictly cheerful.\n\n"
    "Don't end replies with a follow-up question to keep the conversation going "
    "(e.g. \"What else would you like to know?\" or \"Anything else on your "
    "mind?\"). Once you've answered, stop and let the caller decide what to say "
    "next. The only exception is a genuine clarifying question you need "
    "answered in order to give a correct or useful reply in the first place."
)
