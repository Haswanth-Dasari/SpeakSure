"""Tests for the natural-style interviewer guard and deeper project follow-ups."""

from __future__ import annotations

import json
from collections import deque

import pytest

from src.speaksure.interviewer import (
    InterviewPlan,
    Interviewer,
    answer_denies_experience,
    follow_up_presupposes,
)
from src.speaksure.interviewer_prompts import DENIAL_NOTE, STYLE_NOTES
from src.speaksure.llm_client import LLMResult
from src.speaksure.question_bank import BankQuestion

QUESTIONS = (
    BankQuestion("intro", "background", "Tell me about yourself.", ("background",)),
    BankQuestion("conflict", "teamwork", "Describe a disagreement with a teammate.", ("situation",)),
    BankQuestion("resume_project_1", "project: X", "Tell me about your project X.", ("role",)),
    BankQuestion("goals", "goals", "Where do you see yourself?", ("direction",)),
)


class FakeLLM:
    def __init__(self, decisions: list[dict[str, str]]) -> None:
        self.queue = deque(decisions)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages, **kwargs):
        self.messages.append(messages)
        return LLMResult(json.dumps(self.queue.popleft()), "fake", 0.2, None, None, "stop")


def decision(action: str, utterance: str) -> dict[str, str]:
    return {"action": action, "utterance": utterance, "reason": "test"}


def started(llm: FakeLLM, **options) -> Interviewer:
    interviewer = Interviewer(llm, InterviewPlan(questions=QUESTIONS, **options))
    interviewer.opening()
    return interviewer


@pytest.mark.parametrize(
    "text",
    [
        "I have no disagreements",
        "No, never had any conflict with my teammates",
        "Honestly I haven't had a real disagreement",
        "nope",
        "I didn't have any major problems in that project",
    ],
)
def test_denials_are_detected(text: str) -> None:
    assert answer_denies_experience(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "We disagreed about the database and I proposed a small test",
        "I solved the problem by adding a cache to the service",
    ],
)
def test_real_experiences_are_not_denials(text: str) -> None:
    assert answer_denies_experience(text) is False


def test_presupposing_follow_ups_are_detected() -> None:
    assert follow_up_presupposes("How did you solve those disagreements?")
    assert follow_up_presupposes("How did you handle that?")
    assert not follow_up_presupposes("Can you think of a time you had a different approach?")
    assert not follow_up_presupposes("What would you do if a teammate disagreed?")


def test_presupposing_follow_up_after_a_denial_is_replaced_by_the_next_question() -> None:
    llm = FakeLLM(
        [
            decision("next_question", "Okay, describe a disagreement with a teammate."),
            decision("follow_up", "How did you solve those disagreements?"),
        ]
    )
    interviewer = started(llm, natural_style=True)
    interviewer.next_turn("I study computer science.")
    turn = interviewer.next_turn("I have no disagreements.")
    assert turn.kind == "next_question" and turn.question_id == "resume_project_1"
    assert "disagreements" not in turn.utterance.lower() or "Tell me" in turn.utterance


def test_a_sensible_follow_up_after_a_denial_is_kept() -> None:
    llm = FakeLLM(
        [
            decision("next_question", "Okay, describe a disagreement with a teammate."),
            decision("follow_up", "Fair enough. Was there a time you and a teammate wanted different approaches?"),
        ]
    )
    interviewer = started(llm, natural_style=True)
    interviewer.next_turn("I study computer science.")
    turn = interviewer.next_turn("I have no disagreements.")
    assert turn.kind == "follow_up"


def test_guard_is_off_by_default() -> None:
    llm = FakeLLM(
        [
            decision("next_question", "Okay, describe a disagreement."),
            decision("follow_up", "How did you solve those disagreements?"),
        ]
    )
    interviewer = started(llm)
    interviewer.next_turn("I study computer science.")
    assert interviewer.next_turn("I have no disagreements.").kind == "follow_up"
    assert "Interviewer style notes" not in llm.messages[1][1]["content"]


def test_style_and_denial_notes_are_only_added_when_enabled() -> None:
    llm = FakeLLM([decision("next_question", "Next."), decision("next_question", "Next.")])
    interviewer = started(llm, natural_style=True)
    interviewer.next_turn("I study computer science and build tools.")
    interviewer.next_turn("I have no disagreements.")
    first, second = (m[1]["content"] for m in llm.messages)
    assert STYLE_NOTES.strip() in first and DENIAL_NOTE.strip() not in first
    assert STYLE_NOTES.strip() in second and DENIAL_NOTE.strip() in second


def test_project_questions_allow_two_follow_ups_when_configured() -> None:
    llm = FakeLLM(
        [
            decision("next_question", "Next: your project."),
            decision("next_question", "Tell me about project X."),
            decision("follow_up", "Which part was hardest?"),
            decision("follow_up", "What would you change?"),
            decision("next_question", "Where do you see yourself?"),
        ]
    )
    interviewer = started(llm, deep_dive_followups=2)
    interviewer.next_turn("I study computer science.")
    interviewer.next_turn("We never really disagreed, we just talked it through.")
    first = interviewer.next_turn("I built the API with Flask and Docker.")
    second = interviewer.next_turn("The queue was hardest, I used a lock.")
    third = interviewer.next_turn("I would add tests.")
    assert (first.kind, second.kind, third.kind) == ("follow_up", "follow_up", "next_question")


def test_other_questions_keep_the_normal_limit() -> None:
    llm = FakeLLM(
        [
            decision("next_question", "Next."),
            decision("follow_up", "Tell me more."),
            decision("follow_up", "And more?"),
        ]
    )
    interviewer = started(llm, deep_dive_followups=2)
    interviewer.next_turn("I study computer science.")
    first = interviewer.next_turn("We worked on a shared report.")
    second = interviewer.next_turn("It went fine.")
    assert first.kind == "follow_up" and second.kind == "next_question"


def test_invalid_deep_dive_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        InterviewPlan(questions=QUESTIONS, deep_dive_followups=3)
