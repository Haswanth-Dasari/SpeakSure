"""Word-confidence measures for speech-recognition transcripts."""

from __future__ import annotations

import math

from ..speech_to_text import TranscriptionResult

# Probability below which a recognized word is flagged as low confidence.
LOW_CONFIDENCE = 0.5


def analyze_transcript_quality(
    answer: TranscriptionResult | None,
) -> dict[str, object]:
    """Return confidence summary and ordered low-confidence words."""
    words = _answer_words(answer)
    probabilities = [
        probability
        for word in words
        if (probability := _word_probability(word)) is not None
    ]
    low_confidence_words = [
        {
            "word": _word_text(word),
            "probability": probability,
        }
        for word in words
        if (probability := _word_probability(word)) is not None
        and probability < LOW_CONFIDENCE
    ]
    count = len(words)
    return {
        "analyzer_version": 1,
        "has_timed_words": bool(words),
        "mean_word_probability": (
            sum(probabilities) / len(probabilities) if probabilities else None
        ),
        "low_confidence_ratio": (
            len(low_confidence_words) / count if count else 0.0
        ),
        "low_confidence_words": low_confidence_words,
        "min_word_probability": min(probabilities) if probabilities else None,
    }


def _answer_words(answer: TranscriptionResult | None) -> tuple[object, ...]:
    """Read word metadata defensively from a possibly malformed answer."""
    try:
        words = answer.words if answer is not None else ()
        return tuple(words) if words is not None else ()
    except Exception:
        return ()


def _word_probability(word: object) -> float | None:
    """Return a finite confidence value from one word, if available."""
    try:
        probability = word.probability
    except Exception:
        return None
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        return None
    converted = float(probability)
    return converted if math.isfinite(converted) else None


def _word_text(word: object) -> str:
    """Return a safe textual representation for one word."""
    try:
        text = word.text
    except Exception:
        return ""
    return text if isinstance(text, str) else str(text)
