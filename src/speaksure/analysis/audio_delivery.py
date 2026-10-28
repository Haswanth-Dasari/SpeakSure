"""Heuristic pause and loudness measures from saved WAV files.

These measurements are heuristic and depend on energy thresholds rather than
phonetic boundaries; pitch analysis is intentionally out of scope.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

# Duration of each non-overlapping analysis frame, in seconds.
FRAME_SECONDS = 0.03
# Minimum RMS level required for a frame to be considered voiced.
MIN_VOICED_RMS = 0.008
# Fraction of the 90th-percentile RMS used as an adaptive voiced threshold.
VOICED_FRACTION_OF_P90 = 0.10
# Silent runs shorter than this are bridged when enclosed by voiced frames.
SHORT_GAP_SECONDS = 0.2
# Minimum interior silent-run duration counted as a pause.
PAUSE_MIN_SECONDS = 0.5
# Minimum interior silent-run duration counted as a long pause.
LONG_PAUSE_SECONDS = 1.5
# Mean speech level below this value is flagged as too quiet.
TOO_QUIET_DB = -40.0
# Minimum word count for reporting articulation rate.
MIN_WORD_COUNT_FOR_ARTICULATION = 5
# Minimum articulation-time denominator, in seconds, for reporting a rate.
MIN_ARTICULATION_SECONDS = 2.0
# Absolute sample level at or above which a sample is considered clipped.
CLIPPING_SAMPLE_LEVEL = 0.99
# Clipping share above which audio is flagged.
CLIPPING_RATIO_THRESHOLD = 0.001
# Minimum speech span, in seconds, before audio is not considered very short.
VERY_SHORT_SECONDS = 1.0
# PCM integer scale used to normalize signed 16-bit samples.
PCM16_SCALE = 32768.0


def analyze_audio_delivery(
    audio_path: Path | str | None,
    word_count: int,
    audio_root: Path = Path("."),
) -> dict[str, object]:
    """Return JSON-compatible heuristic delivery metrics for a mono PCM WAV."""
    result: dict[str, object] = {
        "analyzer_version": 1,
        "has_audio": False,
    }
    try:
        if audio_path is None:
            return _audio_error(result, "No audio path was provided.")
        path = Path(audio_path)
        if not path.is_absolute():
            path = Path(audio_root) / path
        with wave.open(str(path), "rb") as wav_file:
            if (
                wav_file.getsampwidth() != 2
                or wav_file.getnchannels() != 1
                or wav_file.getcomptype() != "NONE"
            ):
                return _audio_error(result, "Audio must be uncompressed 16-bit PCM mono.")
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if sample_rate <= 0 or frame_count <= 0:
                return _audio_error(result, "Audio file is empty or has an invalid sample rate.")
            raw_samples = wav_file.readframes(frame_count)
        if not raw_samples:
            return _audio_error(result, "Audio file is empty.")
        samples = np.frombuffer(raw_samples, dtype="<i2").astype(np.float64)
        if samples.size != frame_count:
            return _audio_error(result, "Audio file contains incomplete PCM samples.")
        samples /= PCM16_SCALE
        frame_size = max(1, round(sample_rate * FRAME_SECONDS))
        frame_samples = [
            samples[start : start + frame_size]
            for start in range(0, samples.size, frame_size)
        ]
        frame_durations = [len(frame) / sample_rate for frame in frame_samples]
        frame_rms = np.asarray(
            [float(np.sqrt(np.mean(np.square(frame)))) for frame in frame_samples],
            dtype=np.float64,
        )
        voiced_threshold = max(
            MIN_VOICED_RMS,
            VOICED_FRACTION_OF_P90 * float(np.percentile(frame_rms, 90)),
        )
        voiced = frame_rms >= voiced_threshold
        _bridge_short_silent_runs(voiced, frame_durations)

        voiced_indices = np.flatnonzero(voiced)
        peak = float(np.max(np.abs(samples)))
        clipping_ratio = float(np.count_nonzero(np.abs(samples) >= CLIPPING_SAMPLE_LEVEL) / samples.size)
        quality_flags: list[str] = []
        if voiced_indices.size:
            first = int(voiced_indices[0])
            last = int(voiced_indices[-1])
            speech_start = first * frame_size / sample_rate
            speech_end = min((last + 1) * frame_size, samples.size) / sample_rate
            speech_span = max(0.0, speech_end - speech_start)
            voiced_seconds = float(sum(frame_durations[index] for index in voiced_indices))
            voiced_levels = 20.0 * np.log10(
                np.maximum(frame_rms[voiced_indices], 1e-6)
            )
            mean_level = float(np.mean(voiced_levels))
            level_std = float(np.std(voiced_levels))
            dynamic_range = float(
                np.percentile(voiced_levels, 95) - np.percentile(voiced_levels, 10)
            )
            if mean_level < TOO_QUIET_DB:
                quality_flags.append("too_quiet")
            silent_runs = _interior_silent_runs(
                voiced, frame_durations, first, last
            )
            pauses = [
                duration
                for duration in silent_runs
                if duration >= PAUSE_MIN_SECONDS
            ]
        else:
            speech_span = 0.0
            voiced_seconds = 0.0
            mean_level = None
            level_std = None
            dynamic_range = None
            pauses = []
            max_level_db = 20.0 * math.log10(max(float(np.max(frame_rms)), 1e-6))
            if max_level_db < TOO_QUIET_DB:
                quality_flags.append("too_quiet")

        pause_count = len(pauses)
        total_pause = float(sum(pauses))
        longest_pause = max(pauses, default=0.0)
        articulation_seconds = speech_span - total_pause
        articulation_rate = (
            word_count / articulation_seconds * 60.0
            if word_count >= MIN_WORD_COUNT_FOR_ARTICULATION
            and articulation_seconds >= MIN_ARTICULATION_SECONDS
            else None
        )
        if clipping_ratio > CLIPPING_RATIO_THRESHOLD:
            quality_flags.append("clipped")
        if speech_span < VERY_SHORT_SECONDS:
            quality_flags.append("very_short")

        result.update(
            {
                "has_audio": True,
                "duration_seconds": samples.size / sample_rate,
                "speech_span_seconds": speech_span,
                "voiced_seconds": voiced_seconds,
                "pause_count": pause_count,
                "long_pause_count": sum(
                    duration >= LONG_PAUSE_SECONDS for duration in pauses
                ),
                "total_pause_seconds": total_pause,
                "longest_pause_seconds": float(longest_pause),
                "pause_ratio": total_pause / speech_span if speech_span > 0 else None,
                "articulation_rate_wpm": articulation_rate,
                "mean_level_db": mean_level,
                "level_std_db": level_std,
                "dynamic_range_db": dynamic_range,
                "peak": peak,
                "clipping_ratio": clipping_ratio,
                "quality_flags": quality_flags,
            }
        )
        return result
    except Exception as error:
        return _audio_error(result, _short_error(error))


def _audio_error(result: dict[str, object], message: str) -> dict[str, object]:
    """Return the standard unavailable-audio result."""
    result["audio_error"] = message
    return result


def _short_error(error: Exception) -> str:
    """Create a short, non-empty audio read error."""
    message = str(error).strip()
    return message[:200] if message else "Audio file could not be read."


def _bridge_short_silent_runs(
    voiced: np.ndarray, frame_durations: list[float]
) -> None:
    """Mark short interior silent frame runs as voiced in-place."""
    index = 0
    while index < len(voiced):
        if voiced[index]:
            index += 1
            continue
        start = index
        while index < len(voiced) and not voiced[index]:
            index += 1
        if (
            start > 0
            and index < len(voiced)
            and sum(frame_durations[start:index]) < SHORT_GAP_SECONDS
        ):
            voiced[start:index] = True


def _interior_silent_runs(
    voiced: np.ndarray,
    frame_durations: list[float],
    first: int,
    last: int,
) -> list[float]:
    """Return durations of silent runs strictly inside the speech span."""
    runs: list[float] = []
    index = first + 1
    while index < last:
        if voiced[index]:
            index += 1
            continue
        start = index
        while index <= last and not voiced[index]:
            index += 1
        runs.append(sum(frame_durations[start:index]))
    return runs
