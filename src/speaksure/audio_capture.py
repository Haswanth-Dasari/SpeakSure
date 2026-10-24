"""Microphone audio capture with frame-based voice-activity detection."""

from __future__ import annotations

import io
import math
import queue
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


SAMPLE_RATE = 16000
FRAME_SECONDS = 0.03
FRAME_SAMPLES = 480
FloatFrame = NDArray[np.float32]
CaptureEvent = Callable[[str, dict[str, float]], None]


@dataclass(frozen=True)
class CaptureSettings:
    """Immutable settings for calibration and end-of-speech detection."""

    calibration_seconds: float = 1.0
    threshold_multiplier: float = 3.0
    min_threshold: float = 0.008
    release_ratio: float = 0.7
    onset_frames: int = 4
    preroll_seconds: float = 0.3
    tail_seconds: float = 0.3
    end_silence_seconds: float = 1.5
    no_speech_timeout_seconds: float = 20.0
    max_answer_seconds: float = 120.0

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive settings and invalid release ratios."""
        values = {
            "calibration_seconds": self.calibration_seconds,
            "threshold_multiplier": self.threshold_multiplier,
            "min_threshold": self.min_threshold,
            "release_ratio": self.release_ratio,
            "onset_frames": self.onset_frames,
            "preroll_seconds": self.preroll_seconds,
            "tail_seconds": self.tail_seconds,
            "end_silence_seconds": self.end_silence_seconds,
            "no_speech_timeout_seconds": self.no_speech_timeout_seconds,
            "max_answer_seconds": self.max_answer_seconds,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value.")
        if self.release_ratio > 1:
            raise ValueError("release_ratio must not exceed 1.")
        if isinstance(self.onset_frames, bool) or not isinstance(self.onset_frames, int):
            raise ValueError("onset_frames must be a positive integer.")


@dataclass(frozen=True)
class CapturedAnswer:
    """A captured answer and its frame-derived timing and calibration data."""

    samples: FloatFrame
    sample_rate: int
    thinking_time_seconds: float
    answer_duration_seconds: float
    stop_reason: str
    noise_rms: float
    threshold: float


@dataclass(frozen=True)
class NoiseProfile:
    """Median calibration noise level and the derived speech threshold."""

    noise_rms: float
    threshold: float


def _rms(frame: FloatFrame) -> float:
    """Return the root-mean-square amplitude for a single audio frame."""
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def _build_noise_profile(
    calibration_frames: Sequence[FloatFrame], settings: CaptureSettings
) -> NoiseProfile:
    """Compute median calibration RMS and its configured detection threshold."""
    if not calibration_frames:
        raise ValueError("At least one calibration frame is required.")
    noise_rms = float(np.median([_rms(frame) for frame in calibration_frames]))
    threshold = max(
        noise_rms * settings.threshold_multiplier,
        settings.min_threshold,
    )
    return NoiseProfile(noise_rms=noise_rms, threshold=threshold)


def _empty_answer(
    stop_reason: str,
    thinking_time_seconds: float,
    noise_rms: float,
    threshold: float,
) -> CapturedAnswer:
    """Build a no-audio result with the known capture metadata."""
    return CapturedAnswer(
        samples=np.empty(0, dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        thinking_time_seconds=thinking_time_seconds,
        answer_duration_seconds=0.0,
        stop_reason=stop_reason,
        noise_rms=noise_rms,
        threshold=threshold,
    )


def capture_answer(
    frames: Iterator[FloatFrame],
    settings: CaptureSettings | None = None,
    on_event: CaptureEvent | None = None,
    noise_profile: NoiseProfile | None = None,
) -> CapturedAnswer:
    """Consume audio frames until speech ends or a configured limit is reached."""
    configuration = settings or CaptureSettings()
    iterator = iter(frames)
    try:
        calibration_count = round(configuration.calibration_seconds / FRAME_SECONDS)
        preroll_count = round(configuration.preroll_seconds / FRAME_SECONDS)
        tail_count = round(configuration.tail_seconds / FRAME_SECONDS)
        silence_count = round(configuration.end_silence_seconds / FRAME_SECONDS)
        wait_count = round(configuration.no_speech_timeout_seconds / FRAME_SECONDS)
        max_answer_count = round(configuration.max_answer_seconds / FRAME_SECONDS)

        if noise_profile is None:
            calibration_frames: list[FloatFrame] = []
            for _ in range(calibration_count):
                try:
                    frame = np.asarray(next(iterator), dtype=np.float32)
                except StopIteration:
                    return _empty_answer(
                        "source_ended", 0.0, 0.0, configuration.min_threshold
                    )
                if frame.ndim != 1 or frame.size != FRAME_SAMPLES:
                    raise ValueError(
                        f"Each frame must contain {FRAME_SAMPLES} mono samples."
                    )
                calibration_frames.append(frame)
            active_profile = _build_noise_profile(
                calibration_frames, configuration
            )
            frame_index = calibration_count
        else:
            active_profile = noise_profile
            frame_index = 0
        noise_rms = active_profile.noise_rms
        threshold = active_profile.threshold
        if on_event is not None:
            on_event("calibrated", {"noise_rms": noise_rms, "threshold": threshold})

        preroll: deque[tuple[int, FloatFrame]] = deque(maxlen=preroll_count)
        onset_run: list[tuple[int, FloatFrame]] = []
        waiting_frames = 0
        waiting_start_index = frame_index
        onset_index: int | None = None

        while waiting_frames < wait_count:
            try:
                frame = np.asarray(next(iterator), dtype=np.float32)
            except StopIteration:
                elapsed = waiting_frames * FRAME_SECONDS
                return _empty_answer("source_ended", elapsed, noise_rms, threshold)
            if frame.ndim != 1 or frame.size != FRAME_SAMPLES:
                raise ValueError(f"Each frame must contain {FRAME_SAMPLES} mono samples.")
            frame = np.array(frame, dtype=np.float32, copy=True)
            current_index = frame_index
            frame_index += 1
            waiting_frames += 1
            if _rms(frame) >= threshold:
                onset_run.append((current_index, frame))
                if len(onset_run) >= configuration.onset_frames:
                    onset_run = onset_run[-configuration.onset_frames :]
                    onset_index = onset_run[0][0]
                    thinking_time = (
                        onset_index - waiting_start_index
                    ) * FRAME_SECONDS
                    pre_speech_frames = [
                        item for item in preroll if item[0] < onset_index
                    ]
                    if on_event is not None:
                        on_event("speech_started", {})
                    break
            else:
                onset_run.clear()
            preroll.append((current_index, frame))

        if onset_index is None:
            return _empty_answer(
                "no_speech",
                configuration.no_speech_timeout_seconds,
                noise_rms,
                threshold,
            )

        answer_frames: list[tuple[int, FloatFrame]] = list(onset_run)
        last_voiced_index = onset_run[-1][0]
        trailing_silence = 0
        stop_reason = "source_ended"
        while True:
            answer_length = frame_index - onset_index
            if answer_length >= max_answer_count:
                stop_reason = "max_duration"
                break
            try:
                frame = np.asarray(next(iterator), dtype=np.float32)
            except StopIteration:
                stop_reason = "source_ended"
                break
            if frame.ndim != 1 or frame.size != FRAME_SAMPLES:
                raise ValueError(f"Each frame must contain {FRAME_SAMPLES} mono samples.")
            frame = np.array(frame, dtype=np.float32, copy=True)
            current_index = frame_index
            frame_index += 1
            answer_frames.append((current_index, frame))
            if _rms(frame) >= threshold * configuration.release_ratio:
                last_voiced_index = current_index
                trailing_silence = 0
            else:
                trailing_silence += 1
                if trailing_silence >= silence_count:
                    stop_reason = "silence"
                    break

        answer_duration = (last_voiced_index - onset_index + 1) * FRAME_SECONDS
        if stop_reason == "silence":
            captured_after_voice = [
                frame
                for index, frame in answer_frames
                if index > last_voiced_index
            ][:tail_count]
            voiced_and_preroll = [
                frame for _, frame in pre_speech_frames
            ] + [
                frame
                for index, frame in answer_frames
                if onset_index <= index <= last_voiced_index
            ]
            selected_frames = voiced_and_preroll + captured_after_voice
        else:
            selected_frames = [frame for _, frame in pre_speech_frames] + [
                frame for _, frame in answer_frames
            ]
        samples = (
            np.concatenate(selected_frames).astype(np.float32, copy=False)
            if selected_frames
            else np.empty(0, dtype=np.float32)
        )
        return CapturedAnswer(
            samples=samples,
            sample_rate=SAMPLE_RATE,
            thinking_time_seconds=thinking_time,
            answer_duration_seconds=answer_duration,
            stop_reason=stop_reason,
            noise_rms=noise_rms,
            threshold=threshold,
        )
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def calibrate_microphone(
    settings: CaptureSettings | None = None,
    device: int | str | None = None,
) -> NoiseProfile:
    """Capture calibration frames once and return their shared noise profile."""
    configuration = settings or CaptureSettings()
    calibration_count = round(configuration.calibration_seconds / FRAME_SECONDS)
    frames = microphone_frames(device=device)
    calibration_frames: list[FloatFrame] = []
    try:
        for _ in range(calibration_count):
            try:
                frame = np.asarray(next(frames), dtype=np.float32)
            except StopIteration:
                raise RuntimeError(
                    "The microphone stopped before calibration completed."
                ) from None
            if frame.ndim != 1 or frame.size != FRAME_SAMPLES:
                raise ValueError(f"Each frame must contain {FRAME_SAMPLES} mono samples.")
            calibration_frames.append(frame)
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    return _build_noise_profile(calibration_frames, configuration)


def microphone_frames(device: int | str | None = None) -> Iterator[FloatFrame]:
    """Yield copied microphone frames from a lazily opened sounddevice stream."""
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Microphone capture requires sounddevice; install project requirements."
        ) from error

    frames: queue.Queue[FloatFrame] = queue.Queue()

    def callback(
        input_data: NDArray[np.float32],
        frame_count: int,
        time_info: object,
        status: object,
    ) -> None:
        del frame_count, time_info, status
        frames.put(np.asarray(input_data, dtype=np.float32).reshape(-1).copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        device=device,
        callback=callback,
    )
    try:
        stream.start()
        while True:
            try:
                yield frames.get(timeout=2.0)
            except queue.Empty:
                raise RuntimeError(
                    "No microphone audio arrived for 2 seconds; check the selected input device."
                ) from None
    finally:
        stream.close()


def record_answer(
    settings: CaptureSettings | None = None,
    device: int | str | None = None,
    on_event: CaptureEvent | None = None,
    noise_profile: NoiseProfile | None = None,
) -> CapturedAnswer:
    """Capture microphone frames and detect the end of one spoken answer."""
    return capture_answer(
        microphone_frames(device=device),
        settings=settings,
        on_event=on_event,
        noise_profile=noise_profile,
    )


def list_input_devices() -> list[tuple[int, str]]:
    """Return device indices and names for audio inputs."""
    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError(
            "Listing microphone devices requires sounddevice; install project requirements."
        ) from error
    return [
        (index, str(device["name"]))
        for index, device in enumerate(sd.query_devices())
        if int(device.get("max_input_channels", 0)) > 0
    ]


def describe_input_device(device: int | None) -> dict[str, object]:
    """Describe an input device, returning an unknown marker on lookup failures."""
    try:
        import sounddevice as sd

        index = int(sd.default.device[0]) if device is None else device
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            return {"name": "unknown"}
        details = sd.query_devices(index)
        host_api_index = details.get("hostapi")
        host_api: str | int | None = host_api_index
        if isinstance(host_api_index, int):
            host_apis = sd.query_hostapis()
            if 0 <= host_api_index < len(host_apis):
                host_api = str(host_apis[host_api_index].get("name", host_api_index))
        return {
            "name": str(details["name"]),
            "index": index,
            "default_samplerate": float(details["default_samplerate"]),
            "host_api": host_api,
        }
    except Exception:
        return {"name": "unknown"}


def wav_bytes(
    samples: NDArray[np.float32], sample_rate: int = SAMPLE_RATE
) -> bytes:
    """Encode mono float samples as clipped 16-bit PCM WAV bytes."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("samples must be a one-dimensional mono array.")
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


def save_wav(
    path: str | Path,
    samples: NDArray[np.float32],
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Save mono float samples as a 16-bit PCM WAV file."""
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(wav_bytes(samples, sample_rate))
    return wav_path
