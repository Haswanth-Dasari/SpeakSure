"""Resolve the planned main question associated with each recorded answer."""

from __future__ import annotations

import re
from collections.abc import Mapping

from ..question_bank import BankQuestion
from ..session_log import InterviewSession


def resolve_asked_questions(
    session: InterviewSession, bank_by_id: Mapping[str, BankQuestion]
) -> list[dict[str, object]]:
    """Return one recorded, inferred, or unknown main-question label per turn."""
    labels: list[dict[str, object]] = []
    for index, turn in enumerate(session.turns):
        if "asked" in turn.analysis:
            asked = turn.analysis.get("asked")
            asked = asked if isinstance(asked, dict) else {}
            question_id = asked.get("main_question_id")
            is_follow_up = asked.get("is_follow_up")
            labels.append(
                {
                    "main_question_id": question_id if isinstance(question_id, str) else None,
                    "is_follow_up": is_follow_up if isinstance(is_follow_up, bool) else False,
                    "source": "recorded",
                }
            )
            continue

        if index == 0:
            question_id = _match_first_question(turn.question, bank_by_id)
            labels.append(
                {
                    "main_question_id": question_id,
                    "is_follow_up": False,
                    "source": "inferred" if question_id is not None else "unknown",
                }
            )
            continue

        previous_interviewer = session.turns[index - 1].analysis.get(
            "interviewer"
        )
        if not isinstance(previous_interviewer, dict):
            labels.append(_unknown_label())
            continue
        question_id = previous_interviewer.get("question_id")
        kind = previous_interviewer.get("kind")
        if not isinstance(question_id, str) or not question_id or not isinstance(kind, str):
            labels.append(_unknown_label())
            continue
        labels.append(
            {
                "main_question_id": question_id,
                "is_follow_up": kind == "follow_up",
                "source": "inferred",
            }
        )
    return labels


def _match_first_question(
    asked_text: str, bank_by_id: Mapping[str, BankQuestion]
) -> str | None:
    """Find the longest bank question matching the first turn's asked text."""
    normalized_asked = _collapse_whitespace(asked_text)
    matches = [
        question
        for question in bank_by_id.values()
        if (normalized_question := _collapse_whitespace(question.text))
        and (
            normalized_asked == normalized_question
            or normalized_question in normalized_asked
        )
    ]
    if not matches:
        return None
    return max(matches, key=lambda question: len(_collapse_whitespace(question.text))).id


def _collapse_whitespace(text: str) -> str:
    """Normalize question text for case-insensitive bank matching."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _unknown_label() -> dict[str, object]:
    """Build the canonical unknown question label."""
    return {
        "main_question_id": None,
        "is_follow_up": False,
        "source": "unknown",
    }
