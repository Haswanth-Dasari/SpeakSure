"""Load SpeakSure environment configuration from the project root."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def load_environment(env_file: Path | None = None) -> None:
    """Load a specific dotenv file without overriding process environment values."""
    load_dotenv(dotenv_path=ENV_FILE if env_file is None else env_file, override=False)


load_environment()


def get_api_key() -> str:
    """Return the configured key for internal provider use."""
    return os.environ.get("OPENAI_API_KEY", "").strip()


def get_chat_model() -> str:
    """Return the configured chat model or its default."""
    return os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini"


def get_transcription_model() -> str:
    """Return the configured transcription model or its default."""
    return os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "").strip() or "whisper-1"


def get_speech_to_text_provider() -> str:
    """Return the selected speech-to-text backend, defaulting to local inference."""
    return os.environ.get("SPEECH_TO_TEXT_PROVIDER", "").strip().lower() or "local"


def get_local_model_size() -> str:
    """Return the configured local Whisper model size."""
    return os.environ.get("SPEECH_TO_TEXT_MODEL_SIZE", "").strip().lower() or "base"


def get_local_device() -> str:
    """Return the preferred local inference device."""
    return os.environ.get("SPEECH_TO_TEXT_DEVICE", "").strip().lower() or "auto"
