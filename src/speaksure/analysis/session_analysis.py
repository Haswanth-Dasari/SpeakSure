"""Combine transcript-only analyzers into a copied interview session."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path

from ..session_log import InterviewSession
from .audio_delivery import analyze_audio_delivery
from .fluency import analyze_fluency
from .transcript_quality import analyze_transcript_quality


def analyze_session(
    session: InterviewSession, audio_root: Path = Path(".")
) -> InterviewSession:
    """Return a deep-copied session with per-turn analysis and a session summary."""
    analyzed = copy.deepcopy(session)
    fluencies = []
    transcript_qualities = []
    audio_deliveries = []
    for turn in analyzed.turns:
        fluency = analyze_fluency(turn.answer, turn.thinking_time_seconds)
        transcript_quality = analyze_transcript_quality(turn.answer)
        audio_delivery = analyze_audio_delivery(
            turn.audio_path, int(fluency["word_count"]), audio_root
        )
        turn.analysis["fluency"] = fluency
        turn.analysis["transcript_quality"] = transcript_quality
        turn.analysis["audio_delivery"] = audio_delivery
        fluencies.append(fluency)
        transcript_qualities.append(transcript_quality)
        audio_deliveries.append(audio_delivery)

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
        "analysis_versions": {
            "fluency": 1,
            "transcript_quality": 1,
            "audio_delivery": 1,
        },
    }
    analyzed.analysis["fluency_summary"] = summary
    pause_ratio_audio_by_turn = [
        delivery["pause_ratio"] if delivery["has_audio"] else None
        for delivery in audio_deliveries
    ]
    level_std_db_by_turn = [
        delivery["level_std_db"] if delivery["has_audio"] else None
        for delivery in audio_deliveries
    ]
    analyzed.analysis["delivery_summary"] = {
        "audio_turns": sum(bool(delivery["has_audio"]) for delivery in audio_deliveries),
        "mean_pause_ratio_audio": _mean(pause_ratio_audio_by_turn),
        "total_pause_seconds_audio": sum(
            float(delivery.get("total_pause_seconds", 0.0))
            for delivery in audio_deliveries
            if delivery["has_audio"]
        ),
        "total_pause_count_audio": sum(
            int(delivery.get("pause_count", 0))
            for delivery in audio_deliveries
            if delivery["has_audio"]
        ),
        "mean_articulation_rate_wpm": _mean(
            [
                delivery.get("articulation_rate_wpm")
                for delivery in audio_deliveries
                if delivery["has_audio"]
            ]
        ),
        "mean_level_std_db": _mean(
            [
                delivery.get("level_std_db")
                for delivery in audio_deliveries
                if delivery["has_audio"]
            ]
        ),
        "pause_ratio_audio_by_turn": pause_ratio_audio_by_turn,
        "level_std_db_by_turn": level_std_db_by_turn,
        "asr_vs_audio_pauses": [
            {
                "turn": index,
                "asr_pause_count": fluency["pause_count"],
                "audio_pause_count": delivery.get("pause_count")
                if delivery["has_audio"]
                else None,
            }
            for index, (fluency, delivery) in enumerate(
                zip(fluencies, audio_deliveries), start=1
            )
        ],
    }
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
