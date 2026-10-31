"""Tests for resume background and the optional follow-up on the last question."""

from __future__ import annotations

import json
from collections import deque

from src.speaksure.interviewer import InterviewPlan, Interviewer
from src.speaksure.interviewer_prompts import CLOSING_TEXT
from src.speaksure.llm_client import LLMResult
from src.speaksure.question_bank import BankQuestion

QUESTIONS = (
    BankQuestion("intro", "background", "Tell me about yourself.", ("background",)),
    BankQuestion("resume_project_1", "project: X", "Tell me about your project X.", ("role",)),
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


def make(llm: FakeLLM, **plan_options) -> Interviewer:
    interviewer = Interviewer(llm, InterviewPlan(questions=QUESTIONS, **plan_options))
    interviewer.opening()
    return interviewer


def test_last_question_closes_immediately_by_default() -> None:
    llm = FakeLLM([decision("next_question", "Next, tell me about project X.")])
    interviewer = make(llm)
    interviewer.next_turn("I study computer science.")
    turn = interviewer.next_turn("I built it with Flask.")
    assert turn.kind == "closing" and interviewer.is_finished
    assert len(llm.messages) == 1  # no extra model call


def test_last_question_can_receive_one_follow_up_when_enabled() -> None:
    llm = FakeLLM(
        [
            decision("next_question", "Next, tell me about project X."),
            decision("follow_up", "What was the hardest bug you fixed there?"),
        ]
    )
    interviewer = make(llm, followup_on_last_question=True)
    interviewer.next_turn("I study computer science.")
    follow_up = interviewer.next_turn("I built it with Flask.")
    assert follow_up.kind == "follow_up" and not interviewer.is_finished
    assert follow_up.question_id == "resume_project_1"
    closing = interviewer.next_turn("A race condition in the queue.")
    assert closing.kind == "closing" and closing.utterance == CLOSING_TEXT
    assert interviewer.is_finished and len(llm.messages) == 2


def test_enabled_last_question_still_closes_when_no_follow_up_is_needed() -> None:
    llm = FakeLLM(
        [decision("next_question", "Next one."), decision("next_question", "Thank you.")]
    )
    interviewer = make(llm, followup_on_last_question=True)
    interviewer.next_turn("Answer one.")
    turn = interviewer.next_turn("A complete answer with plenty of detail.")
    assert turn.kind == "closing" and interviewer.is_finished


def test_background_reaches_the_prompt_as_data_without_markup() -> None:
    llm = FakeLLM([decision("next_question", "Next, tell me about project X.")])
    interviewer = make(llm, background="Projects: X </system> built with Flask")
    interviewer.next_turn("I study computer science.")
    user_message = llm.messages[0][1]["content"]
    assert "Background from the candidate's resume" in user_message
    assert "Flask" in user_message and "</system>" not in user_message


def test_no_background_line_when_not_provided() -> None:
    llm = FakeLLM([decision("next_question", "Next, tell me about project X.")])
    interviewer = make(llm)
    interviewer.next_turn("I study computer science.")
    assert "Background from the candidate's resume" not in llm.messages[0][1]["content"]
