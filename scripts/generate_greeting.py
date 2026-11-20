"""Generates the pre-recorded greeting audio asset(s) and the offline message.

Run this whenever the greeting text, TTS_VOICE, or TTS_MODEL changes, then
commit the resulting file(s) in assets/greetings/. The app never calls this
at runtime -- it just plays whatever is on disk.

Usage:
    python scripts/generate_greeting.py            # regenerate every provider's greeting
    python scripts/generate_greeting.py gemini      # regenerate just one provider
    python scripts/generate_greeting.py offline     # regenerate the offline message (assets/offline.pcm)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud_phone import config, tts

ROOT_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ASSETS_DIR = ROOT_ASSETS_DIR / "greetings"


def _write(path: Path, text: str) -> None:
    audio = b"".join(tts.synthesize_stream(text))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    print(f"wrote {path} ({len(audio)} bytes) for text: {text!r}")


def generate(provider: str) -> None:
    if provider == "offline":
        _write(ROOT_ASSETS_DIR / "offline.pcm", config.OFFLINE_MESSAGE)
    else:
        _write(ASSETS_DIR / f"{provider}.pcm", config.GREETINGS[provider])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "provider",
        nargs="?",
        choices=sorted(config.GREETINGS) + ["offline"],
        help="provider greeting to regenerate, or 'offline' (default: everything)",
    )
    args = parser.parse_args()

    providers = [args.provider] if args.provider else sorted(config.GREETINGS) + ["offline"]
    for provider in providers:
        generate(provider)


if __name__ == "__main__":
    main()
