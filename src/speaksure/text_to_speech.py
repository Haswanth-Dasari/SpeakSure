"""Offline speech synthesis and playback using Piper."""

from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from numpy.typing import NDArray

from .configuration import (
    get_text_to_speech_length_scale,
    get_text_to_speech_leading_silence_seconds,
    get_text_to_speech_models_dir,
    get_text_to_speech_trailing_silence_seconds,
    get_text_to_speech_voice,
)


class SpeechSynthesisError(RuntimeError):
    """An actionable error raised when Piper cannot synthesize speech."""


class SpeechSynthesizer(Protocol):
    """Interface implemented by text-to-speech engines."""

    def synthesize(self, text: str) -> SynthesizedAudio:
        """Synthesize text into mono floating-point audio."""


@dataclass(frozen=True)
class SynthesizedAudio:
    """Mono audio samples and timing information returned by synthesis."""

    samples: NDArray[np.float32]
    sample_rate: int
    synthesis_seconds: float

    def __post_init__(self) -> None:
        """Normalize and validate the immutable audio result."""
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional mono array.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if not math.isfinite(self.synthesis_seconds) or self.synthesis_seconds < 0:
            raise ValueError("synthesis_seconds must be a finite non-negative value.")
        object.__setattr__(self, "samples", np.clip(samples, -1.0, 1.0))

    @property
    def duration_seconds(self) -> float:
        """Return the duration of the sample array in seconds."""
        return len(self.samples) / self.sample_rate

    @property
    def real_time_factor(self) -> float:
        """Return synthesis time divided by audio duration, or zero for silence."""
        duration = self.duration_seconds
        return self.synthesis_seconds / duration if duration else 0.0


@dataclass(frozen=True)
class SpeechSynthesisConfig:
    """Validated Piper voice, model directory, and speech-rate settings."""

    voice_name: str
    models_dir: Path
    length_scale: float
    leading_silence_seconds: float = 0.15
    trailing_silence_seconds: float = 0.5

    def __post_init__(self) -> None:
        """Validate voice identifiers, paths, speaking speed, and playback padding."""
        if not self.voice_name or Path(self.voice_name).name != self.voice_name:
            raise ValueError("TEXT_TO_SPEECH_VOICE must be a voice name, not a path.")
        if "\x00" in str(self.models_dir):
            raise ValueError("TEXT_TO_SPEECH_MODELS_DIR must be a valid directory path.")
        if not math.isfinite(self.length_scale) or not 0.5 <= self.length_scale <= 2.0:
            raise ValueError("TEXT_TO_SPEECH_LENGTH_SCALE must be from 0.5 to 2.0.")
        if (
            not math.isfinite(self.leading_silence_seconds)
            or not 0 <= self.leading_silence_seconds <= 1
        ):
            raise ValueError(
                "TEXT_TO_SPEECH_LEADING_SILENCE_SECONDS must be from 0 to 1."
            )
        if (
            not math.isfinite(self.trailing_silence_seconds)
            or not 0 <= self.trailing_silence_seconds <= 2
        ):
            raise ValueError(
                "TEXT_TO_SPEECH_TRAILING_SILENCE_SECONDS must be from 0 to 2."
            )

    @classmethod
    def from_env(cls) -> SpeechSynthesisConfig:
        """Read and validate Piper settings from the process environment."""
        voice_name = get_text_to_speech_voice()
        return cls(
            voice_name=voice_name,
            models_dir=Path(get_text_to_speech_models_dir()),
            length_scale=get_text_to_speech_length_scale(),
            leading_silence_seconds=get_text_to_speech_leading_silence_seconds(),
            trailing_silence_seconds=get_text_to_speech_trailing_silence_seconds(),
        )


class _PiperChunk(Protocol):
    """Piper audio chunk fields used by the synthesizer."""

    sample_rate: int
    audio_float_array: NDArray[np.float32]
    audio_int16_array: NDArray[np.int16]


class _PiperConfig(Protocol):
    """Piper voice configuration fields used by the synthesizer."""

    sample_rate: int


class _PiperVoice(Protocol):
    """Piper voice methods and fields used by the synthesizer."""

    config: _PiperConfig

    def synthesize(self, text: str, syn_config: object) -> Iterable[_PiperChunk]:
        """Yield audio chunks for text."""


def _load_piper_voice(model_path: Path) -> _PiperVoice:
    """Load a Piper ONNX voice from its model path."""
    from piper import PiperVoice

    return PiperVoice.load(model_path)


def _chunk_samples(chunk: _PiperChunk) -> NDArray[np.float32]:
    """Convert a Piper float or signed-integer chunk to normalized float samples."""
    float_samples = getattr(chunk, "audio_float_array", None)
    if float_samples is not None:
        values = np.asarray(float_samples)
        if np.issubdtype(values.dtype, np.integer):
            info = np.iinfo(values.dtype)
            samples = values.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            samples = values.astype(np.float32)
    else:
        integer_samples = np.asarray(chunk.audio_int16_array)
        if not np.issubdtype(integer_samples.dtype, np.signedinteger):
            raise TypeError("Piper audio chunks must contain signed integer or float samples.")
        info = np.iinfo(integer_samples.dtype)
        samples = integer_samples.astype(np.float32) / float(max(abs(info.min), info.max))
    if samples.ndim != 1:
        raise ValueError("Piper audio chunks must contain one-dimensional mono samples.")
    return np.clip(samples, -1.0, 1.0)


def _download_command(voice_name: str, models_dir: Path) -> list[str]:
    """Return the exact Piper downloader command for a voice and model directory."""
    return [
        sys.executable,
        "-m",
        "piper.download_voices",
        "--download-dir",
        str(models_dir),
        voice_name,
    ]


def _command_text(command: list[str]) -> str:
    """Format a downloader command as a copyable command-line string."""
    return f"python {subprocess.list2cmdline(command[1:])}"


class PiperSpeechSynthesizer:
    """Synthesize text offline with one lazily loaded, cached Piper voice."""

    def __init__(
        self,
        voice_name: str,
        models_dir: str | Path,
        length_scale: float = 1.0,
        voice_loader: Callable[[Path], _PiperVoice] | None = None,
    ) -> None:
        """Create a Piper synthesizer without loading a voice model yet."""
        if not voice_name or Path(voice_name).name != voice_name:
            raise ValueError("voice_name must be a non-empty Piper voice name.")
        if not math.isfinite(length_scale) or not 0.5 <= length_scale <= 2.0:
            raise ValueError("length_scale must be from 0.5 to 2.0.")
        self.voice_name = voice_name
        self.models_dir = Path(models_dir)
        self.length_scale = length_scale
        self._voice_loader = voice_loader or _load_piper_voice
        self._voice: _PiperVoice | None = None
        self._voice_lock = threading.Lock()

    def _get_voice(self) -> _PiperVoice:
        """Load the voice once, even if warm-up and a request race to do it."""
        if self._voice is not None:
            return self._voice
        with self._voice_lock:
            if self._voice is not None:
                return self._voice
            return self._load_voice()

    def _load_voice(self) -> _PiperVoice:
        """Load the configured voice, raising an actionable error if absent."""

        model_path = self.models_dir / f"{self.voice_name}.onnx"
        config_path = Path(f"{model_path}.json")
        if not model_path.is_file() or not config_path.is_file():
            command = _command_text(_download_command(self.voice_name, self.models_dir))
            missing = []
            if not model_path.is_file():
                missing.append(str(model_path))
            if not config_path.is_file():
                missing.append(str(config_path))
            raise SpeechSynthesisError(
                f"Piper voice files are missing ({', '.join(missing)}). "
                f"Download them with: {command}"
            )

        try:
            self._voice = self._voice_loader(model_path)
        except Exception as error:
            raise SpeechSynthesisError(
                f"Could not load Piper voice '{self.voice_name}' from {model_path}. "
                "Check that the ONNX model and JSON config are valid."
            ) from error
        return self._voice

    def warmup(self) -> None:
        """Load the configured Piper voice before the first interview request."""
        self._get_voice()

    def synthesize(self, text: str) -> SynthesizedAudio:
        """Synthesize non-empty text and return normalized mono audio."""
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")
        if len(text) > 2000:
            raise ValueError("text must not exceed 2000 characters.")

        voice = self._get_voice()
        try:
            from piper.config import SynthesisConfig

            synthesis_config = SynthesisConfig(length_scale=self.length_scale)
            started_at = time.perf_counter()
            chunks = list(voice.synthesize(text, syn_config=synthesis_config))
            synthesis_seconds = time.perf_counter() - started_at
            samples_by_chunk = [_chunk_samples(chunk) for chunk in chunks]
            samples = (
                np.concatenate(samples_by_chunk).astype(np.float32, copy=False)
                if samples_by_chunk
                else np.empty(0, dtype=np.float32)
            )
            return SynthesizedAudio(
                samples=samples,
                sample_rate=voice.config.sample_rate,
                synthesis_seconds=synthesis_seconds,
            )
        except SpeechSynthesisError:
            raise
        except Exception as error:
            raise SpeechSynthesisError(
                f"Piper could not synthesize text with voice '{self.voice_name}'. "
                "Check the voice model and try again."
            ) from error


def ensure_voice(voice_name: str, models_dir: str | Path) -> Path:
    """Download a Piper voice only when its model or JSON config is missing."""
    directory = Path(models_dir)
    model_path = directory / f"{voice_name}.onnx"
    config_path = Path(f"{model_path}.json")
    if model_path.is_file() and config_path.is_file():
        return model_path
    if not voice_name or Path(voice_name).name != voice_name:
        raise ValueError("voice_name must be a non-empty Piper voice name.")

    command = _download_command(voice_name, directory)
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SpeechSynthesisError(
            f"Could not download Piper voice '{voice_name}'. "
            f"Run this command and retry: {_command_text(command)}"
        ) from error
    if not model_path.is_file() or not config_path.is_file():
        raise SpeechSynthesisError(
            f"The Piper downloader did not create both voice files in {directory}. "
            f"Run this command and retry: {_command_text(command)}"
        )
    return model_path


def build_speech_synthesizer(config: SpeechSynthesisConfig) -> SpeechSynthesizer:
    """Build a Piper synthesizer from validated speech-synthesis settings."""
    return PiperSpeechSynthesizer(
        voice_name=config.voice_name,
        models_dir=config.models_dir,
        length_scale=config.length_scale,
    )


def play_audio(
    audio: SynthesizedAudio,
    device: int | None = None,
    *,
    leading_silence_seconds: float = 0.15,
    trailing_silence_seconds: float = 0.5,
) -> float:
    """Play synthesized audio with silence padding and wait for completion."""
    if not math.isfinite(leading_silence_seconds) or leading_silence_seconds < 0:
        raise ValueError("leading_silence_seconds must be finite and non-negative.")
    if not math.isfinite(trailing_silence_seconds) or trailing_silence_seconds < 0:
        raise ValueError("trailing_silence_seconds must be finite and non-negative.")

    import sounddevice as sd

    if leading_silence_seconds == 0 and trailing_silence_seconds == 0:
        played_samples = audio.samples
    else:
        leading_samples = int(round(leading_silence_seconds * audio.sample_rate))
        trailing_samples = int(round(trailing_silence_seconds * audio.sample_rate))
        played_samples = np.concatenate(
            (
                np.zeros(leading_samples, dtype=audio.samples.dtype),
                audio.samples,
                np.zeros(trailing_samples, dtype=audio.samples.dtype),
            )
        )

    started_at = time.perf_counter()
    sd.play(played_samples, samplerate=audio.sample_rate, device=device)
    sd.wait()
    return time.perf_counter() - started_at


def speak(
    text: str,
    synthesizer: SpeechSynthesizer,
    device: int | None = None,
) -> tuple[float, float]:
    """Synthesize and play text, returning synthesis and playback wall-clock times."""
    audio = synthesizer.synthesize(text)
    playback_seconds = play_audio(
        audio,
        device=device,
        leading_silence_seconds=get_text_to_speech_leading_silence_seconds(),
        trailing_silence_seconds=get_text_to_speech_trailing_silence_seconds(),
    )
    return audio.synthesis_seconds, playback_seconds


def list_output_devices() -> list[tuple[int, str]]:
    """Return indices and names for sound devices with output channels."""
    import sounddevice as sd

    return [
        (index, str(device["name"]))
        for index, device in enumerate(sd.query_devices())
        if int(device["max_output_channels"]) > 0
    ]
