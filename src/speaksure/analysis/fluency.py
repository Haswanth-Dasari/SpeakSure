"""Heuristic fluency metrics computed from transcript text and word timings."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..speech_to_text import TranscriptionResult
from .lexicons import DISCOURSE_MARKERS, HEDGES, HESITATION_FILLERS
from .text_utils import count_phrases, normalize_token, split_sentences

# Minimum word gap counted as a pause, in seconds.
PAUSE_MIN_SECONDS = 0.5
# Minimum word gap counted as a long pause, in seconds.
LONG_PAUSE_SECONDS = 1.5
# Minimum number of valid timed words required to calculate words per minute.
MIN_TIMED_WORDS_FOR_WPM = 5
# Minimum speech span, in seconds, required to calculate words per minute.
MIN_SPEECH_SECONDS_FOR_WPM = 2.0


def analyze_fluency(
    answer: TranscriptionResult | None,
    thinking_time_seconds: float | None,
) -> dict[str, object]:
    """Return JSON-compatible word timing, pause, filler, and repetition metrics."""
    text = _answer_text(answer)
    words = _answer_words(answer)
    tokens = [normalize_token(token) for token in text.split()]
    tokens = [token for token in tokens if token]
    word_count = len(words) if words else len(tokens)
    sentences = split_sentences(text)
    timed_words = _finite_timed_words(words)
    speech_seconds = None
    pause_durations: list[float] = []
    if len(timed_words) >= 2:
        span = timed_words[-1][1] - timed_words[0][0]
        if math.isfinite(span) and span >= 0:
            speech_seconds = span
        pause_durations = [
            next_start - previous_end
            for (_, previous_end), (next_start, _) in zip(
                timed_words, timed_words[1:]
            )
            if next_start - previous_end >= PAUSE_MIN_SECONDS
        ]

    pause_count = len(pause_durations)
    total_pause = sum(pause_durations)
    longest_pause = max(pause_durations, default=0.0)
    word_rate = (
        word_count / speech_seconds * 60
        if len(timed_words) >= MIN_TIMED_WORDS_FOR_WPM
        and speech_seconds is not None
        and speech_seconds >= MIN_SPEECH_SECONDS_FOR_WPM
        else None
    )
    filler_count = sum(token in HESITATION_FILLERS for token in tokens)
    discourse_count = count_phrases(tokens, DISCOURSE_MARKERS)
    hedge_count = count_phrases(tokens, HEDGES)
    immediate_repetitions = sum(
        left == right and left not in {"had", "that"}
        for left, right in zip(tokens, tokens[1:])
    )
    has_answer = bool(text.strip() or words)
    return {
        "analyzer_version": 1,
        "has_answer": has_answer,
        "word_count": word_count,
        "sentence_count": len(sentences),
        "avg_words_per_sentence": (
            word_count / len(sentences) if sentences else 0.0
        ),
        "speech_seconds": speech_seconds,
        "words_per_minute": word_rate,
        "pause_count": pause_count,
        "long_pause_count": sum(
            duration >= LONG_PAUSE_SECONDS for duration in pause_durations
        ),
        "total_pause_seconds": total_pause,
        "longest_pause_seconds": longest_pause,
        "pause_ratio": (
            total_pause / speech_seconds
            if speech_seconds is not None and speech_seconds > 0
            else None
        ),
        "thinking_time_seconds": _finite_optional(thinking_time_seconds),
        "filler_count": filler_count,
        "fillers_per_100_words": (
            filler_count * 100 / word_count if word_count else None
        ),
        "discourse_marker_count": discourse_count,
        "hedge_count": hedge_count,
        "hedges_per_100_words": (
            hedge_count * 100 / word_count if word_count else None
        ),
        "immediate_repetitions": immediate_repetitions,
    }


def _answer_text(answer: TranscriptionResult | None) -> str:
    """Read text defensively from a possibly malformed transcription object."""
    try:
        value = answer.text if answer is not None else ""
        return value if isinstance(value, str) else ""
    except Exception:
        return ""


def _answer_words(answer: TranscriptionResult | None) -> tuple[object, ...]:
    """Read word metadata defensively from a possibly malformed answer."""
    try:
        words = answer.words if answer is not None else ()
        return tuple(words) if words is not None else ()
    except Exception:
        return ()


def _finite_optional(value: object) -> float | None:
    """Convert a finite numeric timing value to float, returning None if unavailable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _finite_timed_words(words: Iterable[object]) -> list[tuple[float, float]]:
    """Collect valid finite start/end pairs in their original word order."""
    result = []
    for word in words:
        try:
            start = _finite_optional(word.start)
            end = _finite_optional(word.end)
        except Exception:
            continue
        if start is not None and end is not None and end >= start:
            result.append((start, end))
    return result
