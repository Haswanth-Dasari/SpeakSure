"""Heuristic coaching findings derived from transcript, audio, and content metrics.

All rules in this module are heuristics and should be treated as approximate
coaching guidance rather than definitive evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..analysis.content import BEHAVIOURAL_QUESTION_IDS
from ..analysis.question_labels import resolve_asked_questions
from ..question_bank import BankQuestion
from ..session_log import InterviewSession
from .aggregate import aggregate_group_delivery

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
# Each answer group must have enough words to estimate hedging meaningfully.
MIN_WORDS_PER_GROUP_FOR_HEDGE_STRENGTH = 20
# Eligible answer groups together must have enough words for a stable estimate.
MIN_WORDS_FOR_HEDGE_STRENGTH = 40
# Summary findings per question group before other groups are preferred.
MAX_FINDINGS_PER_QUESTION = 2
# Specificity scores at or below this value merit a concrete-example priority.
CONTENT_SPECIFICITY_LOW = 2
# Fewer than one detected issue per 100 checked words is a cautious strength threshold;
# require 20 checked words so a very short answer cannot earn a rate-based strength.
GRAMMAR_LOW_RATE_PER_100_WORDS = 1.0
GRAMMAR_MIN_WORDS_FOR_STRENGTH = 20
# Question identifiers are presented as concise, reader-facing topic labels.
TOPIC_LABELS = {
    "intro": "introduction",
    "motivation": "motivation",
    "strength": "strengths",
    "weakness": "weaknesses",
    "teamwork": "teamwork",
    "conflict": "disagreement",
    "challenge": "problem solving",
    "mistake": "mistakes",
    "pressure": "pressure",
    "goals": "career goals",
}
GRAMMAR_CATEGORY_LABELS = {
    "subject_verb_agreement": "subject-verb agreement",
    "article": "article use",
    "verb_tense": "simple verb tense",
}
FindingKind = Literal["strength", "priority"]
FindingArea = Literal["content", "delivery", "transcript", "grammar"]


@dataclass(frozen=True)
class Finding:
    """One concise evidence-linked coaching strength or priority."""

    kind: FindingKind
    severity: int
    area: FindingArea
    text: str
    evidence: str
    group_key: int | str | None = None


def build_findings(
    analyzed: dict[str, Any], bank_by_id: Mapping[str, BankQuestion]
) -> list[Finding]:
    """Build interview-specific findings sorted by severity and interview position."""
    turns = analyzed.get("turns", [])
    session_analysis = analyzed.get("analysis", {})
    groups = session_analysis.get("content_by_question", [])
    if not isinstance(groups, list):
        groups = []
    grammar_groups = groups or _inferred_groups(analyzed, bank_by_id)
    findings: list[tuple[int, int, Finding]] = []
    group_index_by_turn = {
        turn_index: index
        for index, group in enumerate(groups)
        for turn_index in group.get("turn_indexes", [])
    }

    for group_position, group in enumerate(groups):
        question_id = group.get("main_question_id")
        question = bank_by_id.get(question_id) if isinstance(question_id, str) else None
        topic = _topic_label(question_id)
        question_name = f"On the {topic} question,"
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
                    f"{question_name} speech recognition was unreliable, so its scores may be understated.",
                    f"Mean word probability: {_display(group.get('asr_mean_probability'))}.",
                    group_position,
                ),
            )
        if group.get("rated") is not True and _reason_category(group) == "too_short_or_empty":
            _append_finding(
                findings,
                position,
                Finding(
                    "priority",
                    2,
                    "content",
                    f"{question_name} your answer was too short to rate; give a fuller answer with a concrete example.",
                    "The answer did not meet the minimum length for content rating.",
                    group_position,
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
                    f"{question_name} include these points next time: {uncovered_text}.",
                    f"Content score: {score:.1f}/100.",
                    group_position,
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
                    f"{question_name} add the uncovered points: {uncovered_text}.",
                    f"Content score: {score:.1f}/100.",
                    group_position,
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
                        f"{question_name} you clearly addressed a key point.",
                        verified_quote,
                        group_position,
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
                    f"{question_name} add concrete examples, numbers or outcomes.",
                    f"Specificity rating: {specificity}/5.",
                    group_position,
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
                        f"{question_name} add the missing STAR parts: {', '.join(missing_parts)}.",
                        "STAR checklist.",
                        group_position,
                    ),
                )

    for group_position, group in enumerate(grammar_groups):
        group_turns = _group_turns(group, turns)
        metrics = aggregate_group_delivery(group_turns)
        checked_words = int(metrics["grammar_word_count_checked"])
        error_count = int(metrics["grammar_error_count"])
        rate = _number(metrics["grammar_errors_per_100_words"])
        turn_indexes = group.get("turn_indexes", [])
        position = min(
            (index for index in turn_indexes if isinstance(index, int)),
            default=group_position + 1,
        )
        group_key = group_position
        phrase = f"On the {_topic_label(group.get('main_question_id'))} question,"

        if (
            checked_words >= GRAMMAR_MIN_WORDS_FOR_STRENGTH
            and rate is not None
            and rate <= GRAMMAR_LOW_RATE_PER_100_WORDS
        ):
            _append_finding(
                findings,
                position,
                Finding(
                    "strength",
                    1,
                    "grammar",
                    f"{phrase} your checked grammar error rate was low at {rate:.1f} per 100 words.",
                    f"{error_count} detected issues across {checked_words} checked words.",
                    group_key,
                ),
            )

        category_counts = metrics["grammar_category_counts"]
        if error_count and category_counts:
            category, category_count = min(
                category_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            example = next(
                (
                    issue
                    for turn in group_turns
                    for issue in _grammar_issues(turn)
                    if issue.get("category") == category
                ),
                None,
            )
            if example is not None:
                label = GRAMMAR_CATEGORY_LABELS.get(
                    category, category.replace("_", " ")
                )
                span = _text(example.get("span_text"))
                suggestion = _text(example.get("suggestion"))
                explanation = _text(example.get("explanation"))
                _append_finding(
                    findings,
                    position,
                    Finding(
                        "priority",
                        2,
                        "grammar",
                        f"{phrase} review {label}; it was the most frequent detected category ({category_count} issue(s)).",
                        f'Example from your answer: "{span}" -> "{suggestion}". {explanation}',
                        group_key,
                    ),
                )

    low_hedge_questions: list[tuple[int, int, str]] = []
    eligible_hedge_words = 0
    for group_position, group in enumerate(groups):
        group_turns = _group_turns(group, turns)
        delivery = aggregate_group_delivery(group_turns)
        group_words = delivery["total_words"]
        group_hedges = sum(
            max(0, int(_number(turn.get("analysis", {}).get("fluency", {}).get("hedge_count")) or 0))
            for turn in group_turns
        )
        if group_words >= MIN_WORDS_PER_GROUP_FOR_HEDGE_STRENGTH:
            eligible_hedge_words += group_words
            group_rate = group_hedges * 100 / group_words
            if group_rate <= HEDGES_LOW:
                position = min(group.get("turn_indexes", [group_position + 1]))
                low_hedge_questions.append(
                    (position, group_position, str(group.get("main_question_id") or "question"))
                )

        topic = _topic_label(group.get("main_question_id"))
        phrase = f"On the {topic} question,"
        wpm = _number(delivery.get("words_per_minute"))
        if wpm is not None and wpm > WPM_FAST:
            _append_finding(
                findings,
                min(group.get("turn_indexes", [group_position + 1])),
                Finding(
                    "priority", 1, "delivery",
                    f"{phrase} your pace was fast at {wpm:.0f} words per minute.",
                    f"{wpm:.1f} words per minute.",
                    group_position,
                ),
            )
        elif wpm is not None and wpm < WPM_SLOW:
            _append_finding(
                findings,
                min(group.get("turn_indexes", [group_position + 1])),
                Finding(
                    "priority", 1, "delivery",
                    f"{phrase} your pace was slow at {wpm:.0f} words per minute.",
                    f"{wpm:.1f} words per minute.",
                    group_position,
                ),
            )
        elif wpm is not None and WPM_STRENGTH_LOW <= wpm <= WPM_STRENGTH_HIGH:
            _append_finding(
                findings,
                min(group.get("turn_indexes", [group_position + 1])),
                Finding(
                    "strength", 1, "delivery",
                    f"{phrase} your pace was within the 120-160 words-per-minute range.",
                    f"{wpm:.1f} words per minute.",
                    group_position,
                ),
            )

        pause_ratio = _number(delivery.get("pause_ratio"))
        longest_pause = _number(delivery.get("longest_pause_seconds"))
        if (pause_ratio is not None and pause_ratio > PAUSE_RATIO_HIGH) or (
            longest_pause is not None and longest_pause >= LONG_SILENCE
        ):
            _append_finding(
                findings,
                min(group.get("turn_indexes", [group_position + 1])),
                Finding(
                    "priority", 2, "delivery",
                    f"{phrase} manage pauses more deliberately.",
                    f"Audio pause ratio: {_display(pause_ratio)}; longest pause: {_display(longest_pause)} seconds.",
                    group_position,
                ),
            )

        if group.get("rated") is True and group_words < SHORT_ANSWER_WORDS:
            _append_finding(
                findings,
                min(group.get("turn_indexes", [group_position + 1])),
                Finding(
                    "priority", 2, "content",
                    f"{phrase} your answer was very short.",
                    f"{group_words} words.",
                    group_position,
                ),
            )

    for turn_position, turn in enumerate(turns, start=1):
        turn_analysis = turn.get("analysis", {})
        fluency = turn_analysis.get("fluency", {})
        if not _turn_has_answer(turn):
            continue
        question_index = group_index_by_turn.get(turn_position)
        question_id = (
            groups[question_index].get("main_question_id")
            if question_index is not None
            else None
        )
        group_key = question_index
        topic = _topic_label(question_id)
        phrase = f"On the {topic} question,"

        hedge_rate = _number(fluency.get("hedges_per_100_words"))
        if hedge_rate is not None and hedge_rate >= HEDGES_HIGH:
            _append_finding(
                findings,
                turn_position,
                Finding(
                    "priority",
                    1,
                    "delivery",
                    f"{phrase} reduce hedging in your answer.",
                    f"{hedge_rate:.1f} hedges per 100 words.",
                    group_key if group_key is not None else f"turn:{turn_position}",
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
                    f"{phrase} reduce hesitation fillers in your answer.",
                    f"{filler_rate:.1f} fillers per 100 words; fillers may be under-detected by speech recognition.",
                    group_key if group_key is not None else f"turn:{turn_position}",
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
                    f"{phrase} reduce the thinking time before answering.",
                    f"{thinking_time:.1f} seconds.",
                    group_key if group_key is not None else f"turn:{turn_position}",
                ),
            )

    answered_questions = {
        group_position
        for group_position, group in enumerate(groups)
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
        and eligible_hedge_words >= MIN_WORDS_FOR_HEDGE_STRENGTH
    ):
        position, group_key, question_id = low_hedge_questions[0]
        topic = _topic_label(question_id)
        _append_finding(
            findings,
            position,
            Finding(
                "strength",
                1,
                "delivery",
                f"On the {topic} question, you kept hedging to {HEDGES_LOW:g} or fewer per 100 words across at least two answers.",
                f"{len(low_hedge_questions)} questions met the low-hedging threshold.",
                group_key,
            ),
        )

    findings.sort(key=lambda item: (-item[2].severity, item[0], item[1]))
    return [finding for _, _, finding in findings]


def _append_finding(
    findings: list[tuple[int, int, Finding]], position: int, finding: Finding
) -> None:
    """Append a finding while preserving interview order for severity ties."""
    findings.append((position, len(findings), finding))


def _group_turns(
    group: dict[str, Any], turns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return valid turns referenced by one question group."""
    return [
        turns[index - 1]
        for index in group.get("turn_indexes", [])
        if isinstance(index, int) and 1 <= index <= len(turns)
    ]


def _inferred_groups(
    analyzed: dict[str, Any], bank_by_id: Mapping[str, BankQuestion]
) -> list[dict[str, Any]]:
    """Group turns by their recorded or inferred main question without content scores."""
    session = InterviewSession.from_dict(analyzed)
    labels = resolve_asked_questions(session, bank_by_id)
    groups: list[dict[str, Any]] = []
    for index, label in enumerate(labels, start=1):
        question_id = label.get("main_question_id")
        question_id = question_id if isinstance(question_id, str) else None
        if groups and groups[-1]["main_question_id"] == question_id:
            groups[-1]["turn_indexes"].append(index)
        else:
            groups.append(
                {"main_question_id": question_id, "turn_indexes": [index]}
            )
    return groups


def _grammar_issues(turn: dict[str, Any]) -> list[dict[str, Any]]:
    grammar = turn.get("analysis", {}).get("grammar", {})
    issues = grammar.get("issues", [])
    return [issue for issue in issues if isinstance(issue, dict)] if isinstance(issues, list) else []


def _topic_label(question_id: object) -> str:
    """Return a readable topic label, falling back to the raw question id."""
    if isinstance(question_id, str) and question_id:
        return TOPIC_LABELS.get(question_id, question_id)
    return "unknown"


def _reason_category(group: dict[str, Any]) -> str:
    """Read the unrated category from either the reason or error field."""
    for key in ("reason", "error"):
        value = group.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _number(value: object) -> float | None:
    """Return a finite numeric value as float when available."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _display(value: float | None) -> str:
    """Format an optional numeric finding value."""
    return "-" if value is None else f"{value:.2f}"


def _text(value: object) -> str:
    """Return text safely for coaching evidence."""
    return value if isinstance(value, str) else ""


def _turn_has_answer(turn: dict[str, Any]) -> bool:
    """Return whether a serialized turn contains non-empty transcript text."""
    answer = turn.get("answer")
    return isinstance(answer, dict) and bool(
        isinstance(answer.get("text"), str) and answer["text"].strip()
    )
