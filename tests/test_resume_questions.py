"""Tests for resume-based question planning and per-session question banks."""

from __future__ import annotations

import json

from src.speaksure.llm_client import LLMError, LLMResult
from src.speaksure.question_bank import HR_QUESTIONS, bank_for_session, question_to_dict
from src.speaksure.resume import validate_profile
from src.speaksure.resume_questions import (
    parse_generated_questions,
    plan_questions,
    profile_background,
    resume_question_count,
    template_questions,
)

PROFILE = validate_profile(
    {
        "headline": "Final-year B.Tech student",
        "skills": ["Python", "FastAPI", "spaCy"],
        "projects": [
            {"name": "SpeakSure", "summary": "A spoken interview coach.", "technologies": ["Python"]},
            {"name": "Inventory Tool", "summary": "Stock tracking in Flask."},
        ],
        "experience": ["Intern at a local startup"],
        "achievements": ["Finalist in a college hackathon"],
    }
)


class FakeLLM:
    def __init__(self, output: object) -> None:
        self.output = output

    def complete(self, messages, **kwargs):
        if isinstance(self.output, Exception):
            raise self.output
        return LLMResult(self.output, "fake", 0.1, None, None, "stop")


def generated(*items: dict) -> str:
    return json.dumps({"questions": list(items)})


GOOD = {
    "topic": "SpeakSure design",
    "text": "In SpeakSure, why did you choose FastAPI for the backend?",
    "hints": ["the reason for the choice", "alternatives considered", "the outcome"],
}


def test_background_is_compact_and_mentions_profile_items() -> None:
    text = profile_background(PROFILE)
    assert len(text) <= 700 and "SpeakSure" in text and "Python" in text


def test_template_questions_cover_projects_skill_experience_achievement() -> None:
    ids = [q.id for q in template_questions(PROFILE)]
    assert ids == [
        "resume_project_1", "resume_project_2", "resume_skill_1",
        "resume_experience_1", "resume_achievement_1",
    ]
    assert all(len(q.rubric_hints) >= 3 for q in template_questions(PROFILE))


def test_generated_questions_must_be_grounded_and_well_formed() -> None:
    raw = generated(
        GOOD,
        {"topic": "x", "text": "What is your favourite colour in general life?", "hints": ["a", "b"]},
        {"topic": "x", "text": "Too short", "hints": ["a", "b"]},
        {"topic": "x", "text": "Why did you pick Python for SpeakSure here?", "hints": ["only one"]},
        {"topic": "x", "text": "word " * 40, "hints": ["a", "b"]},
    )
    questions = parse_generated_questions(raw, PROFILE, 3)
    assert [q.text for q in questions] == [GOOD["text"]]
    assert questions[0].id == "resume_gen_1"
    assert parse_generated_questions("not json", PROFILE, 3) == []


def test_resume_question_count_rules() -> None:
    assert [resume_question_count(n, PROFILE) for n in (3, 5, 8, 10)] == [1, 2, 3, 3]
    assert resume_question_count(2, PROFILE) == 0
    assert resume_question_count(5, validate_profile({})) == 0


def test_plan_mixes_resume_and_bank_with_intro_first_and_bank_last() -> None:
    for total, expected_resume in ((3, 1), (5, 2), (8, 3)):
        plan = plan_questions(total, PROFILE, FakeLLM(generated(GOOD)), seed=1)
        assert len(plan) == total and plan[0].id == "intro"
        assert sum(q.id.startswith("resume_") for q in plan) == expected_resume
        assert not plan[-1].id.startswith("resume_")
        assert len({q.id for q in plan}) == total


def test_plan_falls_back_to_templates_when_the_model_fails() -> None:
    for llm in (FakeLLM(LLMError("down")), FakeLLM("garbage"), object(), None):
        plan = plan_questions(5, PROFILE, llm, seed=1)  # type: ignore[arg-type]
        resume = [q for q in plan if q.id.startswith("resume_")]
        assert len(resume) == 2 and resume[0].id == "resume_project_1"


def test_no_profile_uses_the_plain_bank() -> None:
    assert plan_questions(5, None, None, seed=3) == plan_questions(5, {}, None, seed=3)
    assert plan_questions(5, None, None, seed=3)[0] is HR_QUESTIONS[0]


def test_bank_for_session_adds_valid_custom_questions_only() -> None:
    custom = [q for q in template_questions(PROFILE)][:2]
    metadata = {
        "custom_questions": [question_to_dict(q) for q in custom]
        + [{"id": "intro", "text": "overrides?", "rubric_hints": ["x"]}, {"id": "bad"}, "junk"]
    }
    bank = bank_for_session(metadata)
    assert bank["resume_project_1"].rubric_hints == custom[0].rubric_hints
    assert bank["intro"].text == "Tell me about yourself."  # built-ins cannot be replaced
    assert "bad" not in bank
    assert set(bank_for_session({})) == {q.id for q in HR_QUESTIONS}
    assert set(bank_for_session(None)) == {q.id for q in HR_QUESTIONS}
