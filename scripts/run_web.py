"""Run the local browser-capture SpeakSure API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn

from speaksure.configuration import get_report_stt_model
from speaksure.speech_to_text import (
    LocalTranscriptionConfig,
    describe_stt_configuration,
)
from speaksure.text_to_speech import SpeechSynthesisConfig
from speaksure.web.app import create_app


def build_parser() -> argparse.ArgumentParser:
    """Create the web runner parser with a localhost-only default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print active local model details and run Uvicorn."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be from 1 to 65535.")
    if args.host != "127.0.0.1" and not args.allow_remote:
        print(
            "Refusing non-local host binding without --allow-remote.",
            file=sys.stderr,
        )
        return 2
    if args.allow_remote and args.host != "127.0.0.1":
        print(
            "WARNING: remote binding exposes the local interview API to other machines."
        )

    # Median of three ratings per question unless the .env says otherwise.
    os.environ.setdefault("CONTENT_RATING_SAMPLES", "3")
    # The rule-based grammar section was almost all false alarms on speech (E2).
    os.environ.setdefault("SHOW_RULE_GRAMMAR", "0")
    stt_config = LocalTranscriptionConfig.from_env()
    description = describe_stt_configuration(stt_config)
    tts_config = SpeechSynthesisConfig.from_env()
    print(f"SpeakSure API: http://{args.host}:{args.port}")
    print(
        f"STT: model={description['model']} "
        f"device={description['device_setting']}"
    )
    print(f"TTS voice: {tts_config.voice_name}")
    print(f"Report transcription pass: {get_report_stt_model()}")
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
