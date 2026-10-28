"""Combine transcript, audio, and optional question-level content analyzers."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

from ..llm_client import LLMClient
from ..question_bank import HR_QUESTIONS
from ..session_log import InterviewSession, Turn
from .audio_delivery import analyze_audio_delivery
from .content import ASR_HIGH, ASR_LOW, analyze_content
from .fluency import analyze_fluency
from .grammar import analyze_grammar
from .question_labels import resolve_asked_questions
from .transcript_quality import analyze_transcript_quality


def analyze_session(
    session: InterviewSession,
    audio_root: Path = Path("."),
    llm: LLMClient | None = None,
    role: str = "software engineering fresher",
    progress_callback: Callable[[str, float], None] | None = None,
) -> InterviewSession:
    """Return a deep-copied session with transcript, audio, and optional content analysis."""
    analyzed = copy.deepcopy(session)
    fluencies = []
    transcript_qualities = []
    audio_deliveries = []
    turn_count = len(analyzed.turns)
    if progress_callback is not None:
        progress_callback("grammar", 0.0)
    for turn_index, turn in enumerate(analyzed.turns, start=1):
        fluency = analyze_fluency(turn.answer, turn.thinking_time_seconds)
        grammar = analyze_grammar(turn.answer)
        transcript_quality = analyze_transcript_quality(turn.answer)
        if progress_callback is not None:
            progress_callback("delivery", (turn_index - 1) / max(1, turn_count))
        audio_delivery = analyze_audio_delivery(
            turn.audio_path, int(fluency["word_count"]), audio_root
        )
        turn.analysis["fluency"] = fluency
        turn.analysis["grammar"] = grammar
        turn.analysis["transcript_quality"] = transcript_quality
        turn.analysis["audio_delivery"] = audio_delivery
        fluencies.append(fluency)
        transcript_qualities.append(transcript_quality)
        audio_deliveries.append(audio_delivery)
        if progress_callback is not None:
            progress_callback("grammar", turn_index / max(1, turn_count))
            progress_callback("delivery", turn_index / max(1, turn_count))

    wpm_by_turn = [metric["words_per_minute"] for metric in fluencies]
    pause_ratio_by_turn = [metric["pause_ratio"] for metric in fluencies]
    thinking_time_by_turn = [metric["thinking_time_seconds"] for metric in fluencies]
    analysis_versions = {
        "fluency": 1,
        "grammar": 1,
        "transcript_quality": 1,
        "audio_delivery": 1,
    }
    if llm is not None:
        analysis_versions["content"] = 2
    analyzed.analysis["fluency_summary"] = {
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
            [metric["mean_word_probability"] for metric in transcript_qualities]
        ),
        "wpm_by_turn": wpm_by_turn,
        "pause_ratio_by_turn": pause_ratio_by_turn,
        "thinking_time_by_turn": thinking_time_by_turn,
        "analysis_versions": analysis_versions,
    }

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
        _analyze_content_groups(
            analyzed,
            transcript_qualities,
            llm,
            role,
            progress_callback=progress_callback,
        )
    return analyzed


def _analyze_content_groups(
    session: InterviewSession,
    transcript_qualities: list[dict[str, object]],
    llm: LLMClient,
    role: str,
    progress_callback: Callable[[str, float], None] | None = None,
) -> None:
    """Analyze consecutive question groups, attach references, and summarize scores."""
    bank_by_id = {question.id: question for question in HR_QUESTIONS}
    labels = resolve_asked_questions(session, bank_by_id)
    groups: list[tuple[list[int], str | None, str]] = []
    for turn_index, label in enumerate(labels):
        main_question_id = label["main_question_id"]
        question_id = main_question_id if isinstance(main_question_id, str) else None
        source = str(label["source"])
        if groups and groups[-1][1] == question_id:
            groups[-1][0].append(turn_index)
        else:
            groups.append(([turn_index], question_id, source))

    content_by_question: list[dict[str, Any]] = []
    if progress_callback is not None:
        progress_callback("content", 0.0)
    for group_index, (turn_indexes, question_id, source) in enumerate(groups, start=1):
        turns = [session.turns[index] for index in turn_indexes]
        asr_mean = _group_asr_mean(turn_indexes, session.turns, transcript_qualities)
        asr_reliability = asr_reliability_for(asr_mean)
        content = analyze_content(
            turns,
            question_id,
            bank_by_id,
            role,
            llm,
            asr_mean_probability=asr_mean,
            asr_reliability=asr_reliability,
            include_attempts=True,
        )
        if question_id is None or content.get("error") == "unknown_question":
            content["reason"] = "unknown_question"
            content.pop("error", None)
        group_result: dict[str, Any] = {
            "main_question_id": question_id,
            "turn_indexes": [session.turns[index].index for index in turn_indexes],
            "label_source": source,
            "rated": content.get("rated") is True,
            "asr_mean_probability": asr_mean,
            "asr_reliability": asr_reliability,
            **content,
        }
        content_by_question.append(group_result)
        for turn_index in turn_indexes:
            turn = session.turns[turn_index]
            turn.analysis.pop("content", None)
            turn.analysis["content_ref"] = {
                "main_question_id": question_id,
                "group_index": group_index,
            }
        if progress_callback is not None:
            progress_callback("content", group_index / max(1, len(groups)))

    rated = [result for result in content_by_question if result["rated"]]
    content_score_by_question = [
        result.get("content_score") if result["rated"] else None
        for result in content_by_question
    ]
    feedback_by_question_id: dict[str, dict[str, list[str]]] = {}
    for result in rated:
        question_id = str(result["main_question_id"])
        feedback = feedback_by_question_id.setdefault(
            question_id, {"strengths": [], "improvements": []}
        )
        feedback["strengths"].extend(result["strengths"])
        feedback["improvements"].extend(result["improvements"])

    source_counts = {"recorded": 0, "inferred": 0, "unknown": 0}
    for label in labels:
        source = str(label["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    session.analysis["content_by_question"] = content_by_question
    session.analysis["content_summary"] = {
        "rated_questions": len(rated),
        "unrated_questions": len(content_by_question) - len(rated),
        "mean_content_score": _mean(
            [result.get("content_score") for result in rated]
        ),
        "mean_relevance": _mean([result.get("relevance") for result in rated]),
        "mean_specificity": _mean([result.get("specificity") for result in rated]),
        "mean_clarity": _mean([result.get("clarity") for result in rated]),
        "mean_coverage_ratio": _mean(
            [result.get("coverage_ratio") for result in rated]
        ),
        "content_score_by_question": content_score_by_question,
        "feedback_by_question_id": feedback_by_question_id,
        "label_sources": source_counts,
    }


def _group_asr_mean(
    turn_indexes: list[int],
    turns: list[Turn],
    transcript_qualities: list[dict[str, object]],
) -> float | None:
    """Compute word-count-weighted mean ASR confidence for available timed words."""
    weighted_total = 0.0
    total_words = 0
    for index in turn_indexes:
        quality = transcript_qualities[index]
        answer = turns[index].answer
        if not quality["has_timed_words"] or answer is None:
            continue
        probability = quality["mean_word_probability"]
        word_count = len(answer.words)
        if isinstance(probability, (int, float)) and word_count > 0:
            weighted_total += float(probability) * word_count
            total_words += word_count
    return weighted_total / total_words if total_words else None


def asr_reliability_for(probability: float | None) -> str:
    """Assign confidence to the named low, medium, high, or unknown band."""
    if probability is None:
        return "unknown"
    if probability < ASR_LOW:
        return "low"
    if probability >= ASR_HIGH:
        return "high"
    return "medium"


def _mean(values: Iterable[object]) -> float | None:
    """Return the mean of available finite numeric values, or None if empty."""
    available = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(available) / len(available) if available else None
