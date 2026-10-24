"""Pooled delivery metrics for one interview question group."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any


def aggregate_group_delivery(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool one question group's delivery metrics instead of averaging turn ratios.

    Rate metrics use summed counts over summed denominators, so longer answers
    contribute proportionally. Turns without speech timing are excluded from
    both the pooled WPM numerator and denominator; other word-based rates use
    all available group counts.
    """
    total_words = 0
    timed_words = 0
    speech_seconds = 0.0
    speech_turns = 0
    pause_count = 0
    longest_pauses: list[float] = []
    total_pause_seconds = 0.0
    speech_span_seconds = 0.0
    level_values: list[float] = []
    voiced_weights: list[float | None] = []
    filler_count = 0
    hedge_count = 0
    grammar_word_count = 0
    grammar_error_count = 0
    grammar_category_counts: dict[str, int] = {}
    first_thinking_time: float | None = None

    for turn_index, turn in enumerate(turns):
        analysis = turn.get("analysis", {})
        fluency = analysis.get("fluency", {})
        grammar = analysis.get("grammar", {})
        audio = analysis.get("audio_delivery", {})

        if grammar.get("status") == "ok":
            checked_words = _number(grammar.get("word_count_checked"))
            errors = _number(grammar.get("error_count"))
            if checked_words is not None:
                grammar_word_count += max(0, int(checked_words))
            if errors is not None:
                grammar_error_count += max(0, int(errors))
            issues = grammar.get("issues", [])
            for issue in issues if isinstance(issues, list) else []:
                if not isinstance(issue, dict):
                    continue
                category = issue.get("category")
                if isinstance(category, str) and category:
                    grammar_category_counts[category] = (
                        grammar_category_counts.get(category, 0) + 1
                    )

        words = _number(fluency.get("word_count"))
        if words is not None:
            word_count = max(0, int(words))
            total_words += word_count
        else:
            word_count = 0

        timed_speech = _number(fluency.get("speech_seconds"))
        if timed_speech is not None and timed_speech >= 0:
            speech_seconds += timed_speech
            timed_words += word_count
            speech_turns += 1

        current_pause_count = _number(audio.get("pause_count"))
        if current_pause_count is not None:
            pause_count += max(0, int(current_pause_count))
        longest_pause = _number(audio.get("longest_pause_seconds"))
        if longest_pause is not None and longest_pause >= 0:
            longest_pauses.append(longest_pause)

        pause_seconds = _number(audio.get("total_pause_seconds"))
        span_seconds = _number(audio.get("speech_span_seconds"))
        if pause_seconds is not None and pause_seconds >= 0:
            total_pause_seconds += pause_seconds
        if span_seconds is not None and span_seconds >= 0:
            speech_span_seconds += span_seconds

        level_std = _number(audio.get("level_std_db"))
        if level_std is not None:
            level_values.append(level_std)
            voiced = _number(audio.get("voiced_seconds"))
            voiced_weights.append(voiced if voiced is not None and voiced > 0 else None)

        fillers = _number(fluency.get("filler_count"))
        hedges = _number(fluency.get("hedge_count"))
        if fillers is not None:
            filler_count += max(0, int(fillers))
        if hedges is not None:
            hedge_count += max(0, int(hedges))

        if turn_index == 0:
            first_thinking_time = _number(fluency.get("thinking_time_seconds"))

    has_all_voiced_weights = bool(voiced_weights) and all(
        weight is not None for weight in voiced_weights
    )
    if has_all_voiced_weights:
        weights = [weight for weight in voiced_weights if weight is not None]
        weight_total = sum(weights)
        level_std_db = (
            sum(value * weight for value, weight in zip(level_values, weights))
            / weight_total
            if weight_total > 0
            else mean(level_values)
        )
    else:
        level_std_db = mean(level_values) if level_values else None

    return {
        "total_words": total_words,
        "speech_seconds": speech_seconds,
        "words_per_minute": (
            timed_words * 60 / speech_seconds
            if speech_turns and speech_seconds >= 2.0
            else None
        ),
        "pause_count": pause_count,
        "longest_pause_seconds": max(longest_pauses) if longest_pauses else None,
        "pause_ratio": (
            total_pause_seconds / speech_span_seconds
            if speech_span_seconds > 0
            else None
        ),
        "level_std_db": level_std_db,
        "thinking_time_seconds": first_thinking_time,
        "fillers_per_100_words": filler_count * 100 / total_words if total_words else None,
        "hedges_per_100_words": hedge_count * 100 / total_words if total_words else None,
        "grammar_word_count_checked": grammar_word_count,
        "grammar_error_count": grammar_error_count,
        "grammar_errors_per_100_words": (
            round(grammar_error_count * 100 / grammar_word_count, 2)
            if grammar_word_count
            else None
        ),
        "grammar_category_counts": grammar_category_counts,
    }


def _number(value: object) -> float | None:
    """Return a finite, non-boolean number, or None when unavailable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
