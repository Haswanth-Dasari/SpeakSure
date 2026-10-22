"""Combine transcript-only analyzers into a copied interview session."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path

from ..session_log import InterviewSession
from .audio_delivery import analyze_audio_delivery
from .content import analyze_content
from .fluency import analyze_fluency
from .transcript_quality import analyze_transcript_quality
from ..llm_client import LLMClient
from ..question_bank import HR_QUESTIONS


def analyze_session(
    session: InterviewSession,
    audio_root: Path = Path("."),
    llm: LLMClient | None = None,
    role: str = "software engineering fresher",
) -> InterviewSession:
    """Return a deep-copied session with per-turn analysis and a session summary."""
    analyzed = copy.deepcopy(session)
    fluencies = []
    transcript_qualities = []
    audio_deliveries = []
    content_results = []
    bank_by_id = {question.id: question for question in HR_QUESTIONS}
    for turn in analyzed.turns:
        fluency = analyze_fluency(turn.answer, turn.thinking_time_seconds)
        transcript_quality = analyze_transcript_quality(turn.answer)
        audio_delivery = analyze_audio_delivery(
            turn.audio_path, int(fluency["word_count"]), audio_root
        )
        turn.analysis["fluency"] = fluency
        turn.analysis["transcript_quality"] = transcript_quality
        turn.analysis["audio_delivery"] = audio_delivery
        if llm is not None:
            content = analyze_content(turn, bank_by_id, role, llm)
            turn.analysis["content"] = content
            content_results.append(content)
        fluencies.append(fluency)
        transcript_qualities.append(transcript_quality)
        audio_deliveries.append(audio_delivery)

    wpm_by_turn = [metric["words_per_minute"] for metric in fluencies]
    pause_ratio_by_turn = [metric["pause_ratio"] for metric in fluencies]
    thinking_time_by_turn = [metric["thinking_time_seconds"] for metric in fluencies]
    analysis_versions = {
        "fluency": 1,
        "transcript_quality": 1,
        "audio_delivery": 1,
    }
    if llm is not None:
        analysis_versions["content"] = 1
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
        "analysis_versions": analysis_versions,
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
    if llm is not None:
        rated_content = [
            result for result in content_results if result.get("rated") is True
        ]
        content_score_by_turn = [
            result.get("content_score") if result.get("rated") is True else None
            for result in content_results
        ]
        feedback_by_question_id: dict[str, dict[str, list[str]]] = {}
        for result in rated_content:
            question_id = str(result["question_id"])
            feedback = feedback_by_question_id.setdefault(
                question_id, {"strengths": [], "improvements": []}
            )
            feedback["strengths"].extend(result["strengths"])
            feedback["improvements"].extend(result["improvements"])
        analyzed.analysis["content_summary"] = {
            "rated_turns": len(rated_content),
            "unrated_turns": len(content_results) - len(rated_content),
            "mean_content_score": _mean(
                [result.get("content_score") for result in rated_content]
            ),
            "mean_relevance": _mean(
                [result.get("relevance") for result in rated_content]
            ),
            "mean_specificity": _mean(
                [result.get("specificity") for result in rated_content]
            ),
            "mean_clarity": _mean(
                [result.get("clarity") for result in rated_content]
            ),
            "mean_coverage_ratio": _mean(
                [result.get("coverage_ratio") for result in rated_content]
            ),
            "content_score_by_turn": content_score_by_turn,
            "feedback_by_question_id": feedback_by_question_id,
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
