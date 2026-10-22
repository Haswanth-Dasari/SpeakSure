"""Combine transcript-only analyzers into a copied interview session."""

from __future__ import annotations

import copy
from collections.abc import Iterable

from ..session_log import InterviewSession
from .fluency import analyze_fluency
from .transcript_quality import analyze_transcript_quality


def analyze_session(session: InterviewSession) -> InterviewSession:
    """Return a deep-copied session with per-turn analysis and a session summary."""
    analyzed = copy.deepcopy(session)
    fluencies = []
    transcript_qualities = []
    for turn in analyzed.turns:
        fluency = analyze_fluency(turn.answer, turn.thinking_time_seconds)
        transcript_quality = analyze_transcript_quality(turn.answer)
        turn.analysis["fluency"] = fluency
        turn.analysis["transcript_quality"] = transcript_quality
        fluencies.append(fluency)
        transcript_qualities.append(transcript_quality)

    wpm_by_turn = [metric["words_per_minute"] for metric in fluencies]
    pause_ratio_by_turn = [metric["pause_ratio"] for metric in fluencies]
    thinking_time_by_turn = [metric["thinking_time_seconds"] for metric in fluencies]
    summary = {
        "answered_turns": sum(bool(metric["has_answer"]) for metric in fluencies),
        "unanswered_turns": sum(not bool(metric["has_answer"]) for metric in fluencies),
        "mean_words_per_minute": _mean(wpm_by_turn),
        "mean_pause_ratio": _mean(pause_ratio_by_turn),
        "total_fillers": sum(int(metric["filler_count"]) for metric in fluencies),
        "total_hedges": sum(int(metric["hedge_count"]) for metric in fluencies),
        "total_immediate_repetitions": sum(
            int(metric["immediate_repetitions"]) for metric in fluencies
        ),
        "mean_thinking_time_seconds": _mean(thinking_time_by_turn),
        "mean_transcript_probability": _mean(
            [
                metric["mean_word_probability"]
                for metric in transcript_qualities
            ]
        ),
        "wpm_by_turn": wpm_by_turn,
        "pause_ratio_by_turn": pause_ratio_by_turn,
        "thinking_time_by_turn": thinking_time_by_turn,
        "analysis_versions": {"fluency": 1, "transcript_quality": 1},
    }
    analyzed.analysis["fluency_summary"] = summary
    return analyzed


def _mean(values: Iterable[object]) -> float | None:
    """Return the mean of available finite numeric values, or None if empty."""
    available = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    return sum(available) / len(available) if available else None
