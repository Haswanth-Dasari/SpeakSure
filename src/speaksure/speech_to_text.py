"""Replaceable speech-to-text providers for short practice recordings."""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from src.speaksure.configuration import (
    ENV_FILE,
    get_api_key,
    get_local_device,
    get_local_model_size,
    get_speech_to_text_provider,
    get_transcription_model,
)


MAX_AUDIO_BYTES = 25_000_000
SUPPORTED_AUDIO_TYPES = frozenset(
    {
        "audio/m4a",
        "audio/mp4",
        "audio/mpeg",
        "audio/mpga",
        "audio/wav",
        "audio/webm",
        "audio/x-m4a",
        "audio/x-wav",
    }
)


def _configure_windows_cuda_dlls() -> None:
    """Add installed NVIDIA package DLL folders to this process's PATH on Windows."""
    if sys.platform != "win32":
        return

    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec is None or nvidia_spec.submodule_search_locations is None:
        return

    dll_directories = []
    for package_root in nvidia_spec.submodule_search_locations:
        root = Path(package_root)
        if not root.is_dir():
            continue
        for package_directory in sorted(root.iterdir()):
            dll_directory = package_directory / "bin"
            if dll_directory.is_dir() and next(dll_directory.glob("*.dll"), None):
                dll_directories.append(dll_directory)

    if not dll_directories:
        return

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    existing_paths = {
        os.path.normcase(os.path.abspath(entry))
        for entry in path_entries
        if entry
    }
    directories_to_add = [
        str(directory)
        for directory in dll_directories
        if os.path.normcase(os.path.abspath(directory)) not in existing_paths
    ]
    if directories_to_add:
        os.environ["PATH"] = os.pathsep.join([*directories_to_add, *path_entries])


class SpeechToTextError(RuntimeError):
    """A safe, user-displayable transcription error."""


class MissingTranscriptionConfiguration(SpeechToTextError):
    """Raised when the transcription provider cannot be configured."""


class EmptyAudioError(SpeechToTextError):
    """Raised when no audio data or recognizable speech is available."""


class UnsupportedAudioFormat(SpeechToTextError):
    """Raised when audio format or size is not supported."""


class LocalModelError(SpeechToTextError):
    """Raised when a local transcription model cannot be loaded or run."""


class SpeechToTextProvider(Protocol):
    def transcribe(self, audio: bytes, *, filename: str, content_type: str) -> str: ...


class TranscriptionSegment(Protocol):
    text: str


class WhisperModelLike(Protocol):
    def transcribe(
        self, audio: object, *, beam_size: int
    ) -> tuple[Iterable[TranscriptionSegment], object]: ...


def validate_audio(audio: bytes, content_type: str) -> None:
    """Reject empty, oversized, or unsupported audio before calling a provider."""
    if not isinstance(audio, bytes) or not audio:
        raise EmptyAudioError("No audio was recorded. Allow microphone access and try again.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise UnsupportedAudioFormat("The recording is too large. Please record a shorter response.")
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_type not in SUPPORTED_AUDIO_TYPES:
        raise UnsupportedAudioFormat(
            "This audio format is not supported. Record audio in WAV, WebM, MP3, MP4, MPEG, or M4A format."
        )


@dataclass(frozen=True)
class TranscriptionConfig:
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "TranscriptionConfig":
        key = get_api_key()
        if not key:
            raise MissingTranscriptionConfiguration(
                "Speech transcription is not configured. Set OPENAI_API_KEY in the process environment "
                "or the project-root .env file (not .env.example), then restart SpeakSure."
            )
        return cls(key, get_transcription_model())


def diagnose_transcription_configuration() -> dict[str, bool | str]:
    """Report safe configuration diagnostics without exposing the API key."""
    key = get_api_key()
    provider = get_speech_to_text_provider()
    if provider == "local":
        model = get_local_model_size()
        try:
            local_config = LocalTranscriptionConfig.from_env()
            selected_device = resolve_local_device(get_local_device())
            device = selected_device
            provider_matches = local_config.model_size == model
        except LocalModelError:
            device = "unavailable"
            provider_matches = False
        actual_device = get_loaded_local_device(model, get_local_device())
        sdk_installed = importlib.util.find_spec("faster_whisper") is not None
    elif provider == "openai":
        model = get_transcription_model()
        device = "remote"
        actual_device = "remote"
        sdk_installed = importlib.util.find_spec("openai") is not None
        try:
            provider_config = TranscriptionConfig.from_env()
        except MissingTranscriptionConfiguration:
            provider_matches = not key
        else:
            provider_matches = provider_config.api_key == key and provider_config.model == model
    else:
        model = get_local_model_size()
        device = "unsupported-provider"
        actual_device = "not-initialized"
        sdk_installed = False
        provider_matches = False
    return {
        "root_env_exists": ENV_FILE.is_file(),
        "provider": provider,
        "api_key_configured": bool(key),
        "model": model,
        "device": device,
        "actual_device": actual_device,
        "provider_sdk_installed": sdk_installed,
        "provider_configuration_matches": provider_matches,
    }


@dataclass(frozen=True)
class LocalTranscriptionConfig:
    model_size: str
    device: str

    @classmethod
    def from_env(cls) -> "LocalTranscriptionConfig":
        model_size = get_local_model_size()
        if model_size not in {"tiny", "base", "small", "medium", "large-v3"}:
            raise LocalModelError(
                "Unsupported local speech model. Choose tiny, base, small, medium, or large-v3."
            )
        device = get_local_device()
        if device not in {"auto", "cpu", "cuda"}:
            raise LocalModelError("SPEECH_TO_TEXT_DEVICE must be auto, cpu, or cuda.")
        return cls(model_size=model_size, device=device)


def resolve_local_device(preference: str) -> str:
    """Select CUDA only when explicitly available; otherwise use the CPU fallback."""
    if preference == "cpu":
        return "cpu"
    try:
        _configure_windows_cuda_dlls()
        import ctranslate2

        cuda_available = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda_available = False
    if preference == "cuda":
        if not cuda_available:
            raise LocalModelError(
                "CUDA was requested but is unavailable to CTranslate2. Install compatible CUDA "
                "libraries or set SPEECH_TO_TEXT_DEVICE=cpu."
            )
        return "cuda"
    if preference != "auto":
        raise LocalModelError("SPEECH_TO_TEXT_DEVICE must be auto, cpu, or cuda.")
    return "cuda" if cuda_available else "cpu"


@lru_cache(maxsize=3)
def _cached_local_model(
    model_size: str, device_preference: str
) -> tuple[WhisperModelLike, str, str]:
    """Load and cache a model; auto mode falls back to CPU int8 on CUDA load errors."""
    _configure_windows_cuda_dlls()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise LocalModelError(
            "Local speech recognition is not installed. Install requirements with "
            "'.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
        ) from None

    device = resolve_local_device(device_preference)
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type), device, compute_type
    except Exception:
        if device_preference == "auto" and device == "cuda":
            try:
                return WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu", "int8"
            except Exception:
                raise LocalModelError(
                    "The local Whisper model could not be loaded on CUDA or CPU. Check model "
                    "download access, available memory, and your faster-whisper installation."
                ) from None
        raise LocalModelError(
            f"The local Whisper model could not be loaded on {device.upper()}. Check model "
            "download access, available memory, and the configured device dependencies."
        ) from None


_LOADED_LOCAL_DEVICES: dict[tuple[str, str], str] = {}


def _load_local_model(
    model_size: str, device_preference: str
) -> tuple[WhisperModelLike, str, str]:
    result = _cached_local_model(model_size, device_preference)
    _LOADED_LOCAL_DEVICES[(model_size, device_preference)] = result[1]
    return result


def get_loaded_local_device(model_size: str, preference: str) -> str:
    """Return the device used by an already cached model, if any."""
    return _LOADED_LOCAL_DEVICES.get(
        (model_size, preference), "not-initialized"
    )


def _decode_local_audio(audio: bytes):
    """Decode and resample with PyAV without relying on version-specific kwargs."""
    try:
        import av
        import numpy as np

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        with av.open(io.BytesIO(audio), mode="r") as container:
            for frame in container.decode(audio=0):
                chunks.extend(
                    output.to_ndarray().reshape(-1)
                    for output in resampler.resample(frame)
                )
            chunks.extend(
                output.to_ndarray().reshape(-1) for output in resampler.resample(None)
            )
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32) / 32768.0
    except Exception:
        raise LocalModelError(
            "The recording could not be decoded. Try recording again in WAV format."
        ) from None


class LocalSpeechToTextProvider:
    """Local faster-whisper provider with cached model weights and in-memory audio."""

    def __init__(
        self,
        config: LocalTranscriptionConfig | None = None,
        model_loader: Callable[
            [str, str], tuple[WhisperModelLike, str, str]
        ] | None = None,
    ) -> None:
        self.config = config or LocalTranscriptionConfig.from_env()
        self._model_loader = model_loader or _load_local_model
        self.actual_device = "not-initialized"

    def transcribe(self, audio: bytes, *, filename: str, content_type: str) -> str:
        del filename
        validate_audio(audio, content_type)
        try:
            model, device, _ = self._model_loader(
                self.config.model_size, self.config.device
            )
            decoded_audio = _decode_local_audio(audio)
            if decoded_audio.size == 0:
                raise EmptyAudioError(
                    "No audio was found in the recording. Record a response or type your answer."
                )
            try:
                transcript = self._transcribe_decoded(model, decoded_audio)
                self.actual_device = device
            except SpeechToTextError:
                raise
            except Exception:
                if self.config.device != "auto" or device != "cuda":
                    raise
                model, _, _ = self._model_loader(self.config.model_size, "cpu")
                transcript = self._transcribe_decoded(model, decoded_audio)
                self.actual_device = "cpu"
                _LOADED_LOCAL_DEVICES[(self.config.model_size, "auto")] = "cpu"
        except SpeechToTextError:
            raise
        except Exception:
            raise LocalModelError(
                "Local transcription failed on the selected device and its CPU fallback. Check the "
                "recording, available memory, and local model installation, then try again."
            ) from None
        if not transcript:
            raise EmptyAudioError(
                "No speech was recognized. Try a clearer or longer recording, or type your answer."
            )
        return transcript

    @staticmethod
    def _transcribe_decoded(model: WhisperModelLike, audio: object) -> str:
        segments, _ = model.transcribe(audio, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()


def create_speech_to_text_provider(
    provider: str | None = None,
) -> SpeechToTextProvider:
    """Construct the explicitly selected transcription provider without backend fallback."""
    selected = (provider or get_speech_to_text_provider()).strip().lower()
    if selected == "local":
        return LocalSpeechToTextProvider()
    if selected == "openai":
        return OpenAISpeechToTextProvider()
    raise SpeechToTextError(
        "Unsupported speech provider. Set SPEECH_TO_TEXT_PROVIDER to 'local' or 'openai'."
    )


class OpenAISpeechToTextProvider:
    """OpenAI transcription implementation; the SDK is imported only for a request."""

    def __init__(
        self,
        config: TranscriptionConfig | None = None,
        client_factory: Callable[..., object] | None = None,
    ) -> None:
        self.config = config or TranscriptionConfig.from_env()
        self._client_factory = client_factory

    def transcribe(self, audio: bytes, *, filename: str, content_type: str) -> str:
        validate_audio(audio, content_type)
        normalized_content_type = content_type.split(";", 1)[0].strip().lower()
        try:
            if self._client_factory is None:
                try:
                    from openai import OpenAI
                except ImportError:
                    raise SpeechToTextError(
                        "The OpenAI SDK is not installed. Install project requirements and try again."
                    ) from None

                client_factory = OpenAI
            else:
                client_factory = self._client_factory
            client = client_factory(api_key=self.config.api_key)
            response = client.audio.transcriptions.create(
                model=self.config.model,
                file=(filename, audio, normalized_content_type),
            )
            transcript = response.text
            if not isinstance(transcript, str) or not transcript.strip():
                raise EmptyAudioError(
                    "No speech was recognized. Try a clearer or longer recording, or type your answer."
                )
            return transcript.strip()
        except SpeechToTextError:
            raise
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            error_code = getattr(error, "code", None)
            error_type = type(error).__name__
            if status_code == 401 or error_type == "AuthenticationError":
                message = "OpenAI authentication failed. Check OPENAI_API_KEY and try again."
            elif status_code == 429 or error_code == "insufficient_quota":
                message = (
                    "OpenAI transcription quota or rate limit reached. Check your OpenAI account and try again later."
                )
            elif error_type in {"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError"}:
                message = "Could not connect to OpenAI for transcription. Check your network and try again."
            else:
                message = "Speech transcription failed. Try again, or type your answer instead."
            raise SpeechToTextError(message) from None
