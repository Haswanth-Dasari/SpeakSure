"""Tests for median-of-N content rating."""

from __future__ import annotations

import json

import pytest

from src.speaksure.analysis.content import (
    analyze_content,
    get_rating_samples,
    select_median_result,
)
from src.speaksure.llm_client import LLMResult
from src.speaksure.question_bank import BankQuestion
from src.speaksure.session_log import InterviewSession
from src.speaksure.speech_to_text import TranscriptionResult

QUESTION = BankQuestion("intro", "background", "Tell me about yourself.", ("background", "skill"))
ANSWER = "I study computer science and I built a small inventory tool for my team."


def raw_result(relevance: int, clarity: int = 3) -> str:
    return json.dumps(
        {
            "relevance": relevance,
            "specificity": 3,
            "clarity": clarity,
            "coverage": [
                {"hint": "background", "covered": True, "evidence": "I study computer science"},
                {"hint": "skill", "covered": False, "evidence": ""},
            ],
            "star": None,
            "strengths": ["Clear background."],
            "improvements": ["Add a skill."],
            "reason": "Partly complete.",
        }
    )


class FakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return LLMResult(self.outputs.pop(0), "fake-model", 0.1, None, None, "stop")


def rate(llm: FakeLLM) -> dict:
    session = InterviewSession.new("hr", "standard", 1)
    turn = session.add_turn(
        QUESTION.text,
        answer=TranscriptionResult(ANSWER, (), 4.0, "en", 1.0, "test"),
        analysis={"interviewer": {"question_id": "intro", "kind": "main"}},
    )
    return analyze_content(
        [turn],
        "intro",
        {"intro": QUESTION},
        "role",
        llm,
        asr_mean_probability=0.9,
        asr_reliability="high",
    )


def test_default_is_a_single_rating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTENT_RATING_SAMPLES", raising=False)
    llm = FakeLLM([raw_result(4)])
    result = rate(llm)
    assert llm.calls == 1
    assert "content_score_samples" not in result


def test_three_samples_return_the_median(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTENT_RATING_SAMPLES", "3")
    llm = FakeLLM([raw_result(2), raw_result(5), raw_result(3)])
    result = rate(llm)
    assert llm.calls == 3
    assert result["relevance"] == 3
    assert len(result["content_score_samples"]) == 3
    assert result["content_score_samples"] == sorted(result["content_score_samples"])
    assert result["content_score_spread"] > 0
    assert result["content_score"] == result["content_score_samples"][1]


def test_unrated_first_sample_stops_early(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTENT_RATING_SAMPLES", "3")
    llm = FakeLLM(["not json", "not json"])
    result = rate(llm)
    assert result["rated"] is False
    assert llm.calls == 2  # the single retry only; no extra samples


@pytest.mark.parametrize(
    ("value", "expected"), [("", 1), ("x", 1), ("0", 1), ("3", 3), ("99", 5)]
)
def test_sample_count_is_clamped(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    monkeypatch.setenv("CONTENT_RATING_SAMPLES", value)
    assert get_rating_samples() == expected


def test_select_median_ignores_unrated_samples() -> None:
    results = [
        {"rated": True, "content_score": 40.0},
        {"rated": False, "error": "invalid_output"},
        {"rated": True, "content_score": 80.0},
    ]
    chosen = select_median_result(results)
    assert chosen["content_score"] == 80.0
    assert chosen["content_score_samples"] == [40.0, 80.0]
