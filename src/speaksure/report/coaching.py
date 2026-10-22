"""Heuristic coaching findings derived from transcript, audio, and content metrics.

All rules in this module are heuristics and should be treated as approximate
coaching guidance rather than definitive evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..analysis.content import BEHAVIOURAL_QUESTION_IDS
from ..question_bank import BankQuestion

# Content scores below this value create the strongest content priority.
CONTENT_LOW = 40
# Content scores below this value remain in the developing range.
CONTENT_HIGH = 70
# Pace above this value is considered fast.
WPM_FAST = 170
# Pace below this value is considered slow.
WPM_SLOW = 110
# Pace in this inclusive range supports a delivery strength.
WPM_STRENGTH_LOW = 120
WPM_STRENGTH_HIGH = 160
# Audio pause ratio above this value merits a pause-management priority.
PAUSE_RATIO_HIGH = 0.25
# Longest audio pause at or above this duration merits a priority, in seconds.
LONG_SILENCE = 2.0
# Hedge rate at or above this value merits a priority per 100 words.
HEDGES_HIGH = 5
# Hedge rate at or below this value can support a strength per 100 words.
HEDGES_LOW = 1
# Filler rate at or above this value merits a priority per 100 words.
FILLERS_HIGH = 4
# Answer length below this word count is considered very short.
SHORT_ANSWER_WORDS = 25
# Thinking time above this value merits a priority, in seconds.
THINKING_LONG = 5.0
# Minimum number of answered question groups needed for a low-hedge strength.
MIN_QUESTIONS_FOR_HEDGE_STRENGTH = 2
# Specificity scores at or below this value merit a concrete-example priority.
CONTENT_SPECIFICITY_LOW = 2
FindingKind = Literal["strength", "priority"]
FindingArea = Literal["content", "delivery", "transcript"]


@dataclass(frozen=True)
class Finding:
    """One concise evidence-linked coaching strength or priority."""

    kind: FindingKind
    severity: int
    area: FindingArea
    text: str
    evidence: str


def build_findings(
    analyzed: dict[str, Any], bank_by_id: Mapping[str, BankQuestion]
) -> list[Finding]:
    """Build interview-specific findings sorted by severity and interview position."""
    turns = analyzed.get("turns", [])
    session_analysis = analyzed.get("analysis", {})
    groups = session_analysis.get("content_by_question", [])
    findings: list[tuple[int, int, Finding]] = []
    group_index_by_turn = {
        turn_index: index
        for index, group in enumerate(groups)
        for turn_index in group.get("turn_indexes", [])
    }

    for group_position, group in enumerate(groups):
        question_id = group.get("main_question_id")
        question = bank_by_id.get(question_id) if isinstance(question_id, str) else None
        question_name = question.text if question is not None else str(question_id or "Unknown question")
        turn_indexes = group.get("turn_indexes", [])
        position = min(turn_indexes, default=group_position + 1)
        if group.get("asr_reliability") == "low":
            _append_finding(
                findings,
                position,
                Finding(
                    "priority",
                    3,
                    "transcript",
                    f"Recognition for {question_name} was unreliable, so its scores may be understated.",
                    f"Mean word probability: {_display(group.get('asr_mean_probability'))}.",
                ),
            )
        if group.get("rated") is not True:
            continue

        coverage = group.get("coverage", [])
        uncovered_model_hints = {
            entry.get("hint")
            for entry in coverage
            if not entry.get("covered")
        }
        uncovered = (
            [
                hint
                for hint in question.rubric_hints
                if hint in uncovered_model_hints
            ]
            if question is not None
            else [
                entry.get("hint", "")
                for entry in coverage
                if not entry.get("covered")
            ]
        )
        score = _number(group.get("content_score"))
        uncovered_text = "; ".join(
            hint
            for hint in uncovered
            if isinstance(hint, str) and hint
        )
        if score is not None and score < CONTENT_LOW and uncovered_text:
            _append_finding(
                findings,
                position,
                Finding(
                    "priority",
                    3,
                    "content",
                    f"For {question_name}, include these points next time: {uncovered_text}.",
                    f"Content score: {score:.1f}/100.",
                ),
            )
        elif (
            score is not None
            and CONTENT_LOW <= score < CONTENT_HIGH
            and uncovered_text
        ):
            _append_finding(
                findings,
                position,
                Finding(
                    "priority",
                    2,
                    "content",
                    f"For {question_name}, add the uncovered points: {uncovered_text}.",
                    f"Content score: {score:.1f}/100.",
                ),
            )
        elif score is not None and score >= CONTENT_HIGH:
            verified_quote = next(
                (
                    entry.get("evidence")
                    for entry in coverage
                    if entry.get("covered")
                    and entry.get("evidence_verified")
                    and isinstance(entry.get("evidence"), str)
                    and entry["evidence"]
                ),
                "",
            )
            if verified_quote:
                _append_finding(
                    findings,
                    position,
                    Finding(
                        "strength",
                        2,
                        "content",
                        f"For {question_name}, you clearly addressed a key point.",
                        verified_quote,
                    ),
                )

        specificity = group.get("specificity")
        if (
            isinstance(specificity, int)
            and not isinstance(specificity, bool)
            and specificity <= CONTENT_SPECIFICITY_LOW
        ):
            _append_finding(
                findings,
                position,
                Finding(
                    "priority",
                    2,
                    "content",
                    f"For {question_name}, add concrete examples, numbers or outcomes.",
                    f"Specificity rating: {specificity}/5.",
                ),
            )
        star = group.get("star")
        if question_id in BEHAVIOURAL_QUESTION_IDS and isinstance(star, dict):
            missing_parts = [
                part
                for part in ("situation", "task", "action", "result")
                if star.get(part) is False
            ]
            if missing_parts:
                _append_finding(
                    findings,
                    position,
                    Finding(
                        "priority",
                        2,
                        "content",
                        f"For {question_name}, add the missing STAR parts: {', '.join(missing_parts)}.",
                        "STAR checklist.",
                    ),
                )

    low_hedge_questions: list[tuple[int, str]] = []
    for turn_position, turn in enumerate(turns, start=1):
        turn_analysis = turn.get("analysis", {})
        fluency = turn_analysis.get("fluency", {})
        audio = turn_analysis.get("audio_delivery", {})
        if not _turn_has_answer(turn):
            continue
        word_count = fluency.get("word_count", 0)
        question_index = group_index_by_turn.get(turn_position)
        question_id = (
            groups[question_index].get("main_question_id")
            if question_index is not None
            else None
        )
        question = bank_by_id.get(question_id) if isinstance(question_id, str) else None
        question_name = question.text if question is not None else f"answer {turn_position}"

        wpm = _number(fluency.get("words_per_minute"))
        if wpm is not None and wpm > WPM_FAST:
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    1,
                    "delivery",
                    f"Your pace on {question_name} was fast at {wpm:.0f} words per minute.",
                    f"{wpm:.1f} words per minute.",
                ),
            )
        elif wpm is not None and wpm < WPM_SLOW:
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    1,
                    "delivery",
                    f"Your pace on {question_name} was slow at {wpm:.0f} words per minute.",
                    f"{wpm:.1f} words per minute.",
                ),
            )
        elif (
            wpm is not None
            and WPM_STRENGTH_LOW <= wpm <= WPM_STRENGTH_HIGH
        ):
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "strength",
                    1,
                    "delivery",
                    f"Your pace on {question_name} was within the 120-160 words-per-minute range.",
                    f"{wpm:.1f} words per minute.",
                ),
            )

        pause_ratio = _number(audio.get("pause_ratio"))
        longest_pause = _number(audio.get("longest_pause_seconds"))
        if (
            (pause_ratio is not None and pause_ratio > PAUSE_RATIO_HIGH)
            or (longest_pause is not None and longest_pause >= LONG_SILENCE)
        ):
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    2,
                    "delivery",
                    f"Manage pauses in your answer to {question_name} more deliberately.",
                    f"Audio pause ratio: {_display(pause_ratio)}; longest pause: {_display(longest_pause)} seconds.",
                ),
            )

        hedge_rate = _number(fluency.get("hedges_per_100_words"))
        if hedge_rate is not None and hedge_rate >= HEDGES_HIGH:
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    1,
                    "delivery",
                    f"Reduce hedging in your answer to {question_name}.",
                    f"{hedge_rate:.1f} hedges per 100 words.",
                ),
            )
        filler_rate = _number(fluency.get("fillers_per_100_words"))
        if filler_rate is not None and filler_rate >= FILLERS_HIGH:
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    1,
                    "delivery",
                    f"Reduce hesitation fillers in your answer to {question_name}.",
                    f"{filler_rate:.1f} fillers per 100 words; fillers may be under-detected by speech recognition.",
                ),
            )
        if isinstance(word_count, int) and word_count < SHORT_ANSWER_WORDS:
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    2,
                    "content",
                    f"Your answer to {question_name} was very short.",
                    f"{word_count} words.",
                ),
            )
        thinking_time = _number(fluency.get("thinking_time_seconds"))
        if thinking_time is not None and thinking_time > THINKING_LONG:
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    1,
                    "delivery",
                    f"Reduce the thinking time before answering {question_name}.",
                    f"{thinking_time:.1f} seconds.",
                ),
            )

    for group in groups:
        group_words = 0
        group_hedges = 0
        for turn_index in group.get("turn_indexes", []):
            if not isinstance(turn_index, int) or not 1 <= turn_index <= len(turns):
                continue
            fluency = turns[turn_index - 1].get("analysis", {}).get("fluency", {})
            words = fluency.get("word_count")
            hedges = fluency.get("hedge_count")
            if (
                fluency.get("has_answer")
                and isinstance(words, int)
                and words > 0
                and isinstance(hedges, int)
            ):
                group_words += words
                group_hedges += hedges
        if group_words:
            group_rate = group_hedges * 100 / group_words
            if group_rate <= HEDGES_LOW:
                position = min(group.get("turn_indexes", [1]))
                low_hedge_questions.append((position, str(group.get("main_question_id") or "question")))

    answered_questions = {
        group.get("main_question_id")
        for group in groups
        if group.get("rated") is True
        or any(
            _turn_has_answer(turns[index - 1])
            for index in group.get("turn_indexes", [])
            if isinstance(index, int) and 1 <= index <= len(turns)
        )
    }
    if (
        len(low_hedge_questions) >= MIN_QUESTIONS_FOR_HEDGE_STRENGTH
        and len(answered_questions) >= MIN_QUESTIONS_FOR_HEDGE_STRENGTH
    ):
        _append_finding(
            findings,
            low_hedge_questions[0][0],
            Finding(
                "strength",
                1,
                "delivery",
                f"You kept hedging to {HEDGES_LOW:g} or fewer per 100 words in at least two answers.",
                f"{len(low_hedge_questions)} questions met the low-hedging threshold.",
            ),
        )

    findings.sort(key=lambda item: (-item[2].severity, item[0], item[1]))
    return [finding for _, _, finding in findings]


def _append_finding(
    findings: list[tuple[int, int, Finding]], position: int, finding: Finding
) -> None:
    """Append a finding while preserving interview order for severity ties."""
    findings.append((position, len(findings), finding))


def _number(value: object) -> float | None:
    """Return a finite numeric value as float when available."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _display(value: float | None) -> str:
    """Format an optional numeric finding value."""
    return "-" if value is None else f"{value:.2f}"


def _turn_has_answer(turn: dict[str, Any]) -> bool:
    """Return whether a serialized turn contains non-empty transcript text."""
    answer = turn.get("answer")
    return isinstance(answer, dict) and bool(
        isinstance(answer.get("text"), str) and answer["text"].strip()
    )
