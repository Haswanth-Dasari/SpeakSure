"""Tests for the varied, realistic interview planner."""

from __future__ import annotations

import pytest

from src.speaksure.interview_planner import (
    INTRO_VARIANTS,
    all_bank,
    built_in_ids,
    plan_realistic,
)
from src.speaksure.question_bank import HR_QUESTIONS
from src.speaksure.resume import validate_profile

PROFILE = validate_profile(
    {
        "headline": "Student",
        "skills": ["Python", "FastAPI"],
        "projects": [
            {"name": "SpeakSure", "summary": "A spoken interview coach."},
            {"name": "Inventory Tool", "summary": "Stock tracking in Flask."},
        ],
        "experience": ["Intern at a startup"],
    }
)


@pytest.mark.parametrize("total", [1, 2, 3, 4, 5, 8, 10, 12])
@pytest.mark.parametrize("profile", [PROFILE, None])
def test_plan_has_exact_size_distinct_ids_and_intro_first(total: int, profile) -> None:
    for seed in range(40):
        plan = plan_realistic(total, profile, None, seed)
        assert len(plan) == total
        assert len({q.id for q in plan}) == total
        assert plan[0].id == "intro" and plan[0].text in INTRO_VARIANTS


def test_same_seed_gives_same_plan_and_different_seeds_vary() -> None:
    assert plan_realistic(5, PROFILE, None, 7) == plan_realistic(5, PROFILE, None, 7)
    orders = {tuple(q.id for q in plan_realistic(5, None, None, seed)) for seed in range(60)}
    assert len(orders) > 20  # real variety, not a fixed order
    openers = {plan_realistic(5, None, None, seed)[0].text for seed in range(60)}
    assert len(openers) == len(INTRO_VARIANTS)


def test_longer_plans_end_with_a_closing_question() -> None:
    for seed in range(40):
        plan = plan_realistic(5, PROFILE, None, seed)
        assert plan[-1].id in {"goals", "candidate_questions"}
        assert not plan[-1].id.startswith("resume_")


def test_plans_never_open_with_a_negative_question() -> None:
    for total in (3, 5, 8):
        for seed in range(80):
            plan = plan_realistic(total, PROFILE if seed % 2 else None, None, seed)
            assert plan[1].id not in {"weakness", "conflict", "mistake", "feedback"}


def test_resume_questions_are_mixed_in_at_varying_positions() -> None:
    positions = set()
    for seed in range(60):
        plan = plan_realistic(8, PROFILE, None, seed)
        positions.add(tuple(i for i, q in enumerate(plan) if q.id.startswith("resume_")))
    assert len(positions) > 5
    assert all(len(p) >= 1 for p in positions)


def test_intro_variants_keep_the_builtin_id_and_hints() -> None:
    for seed in range(10):
        intro = plan_realistic(3, None, None, seed)[0]
        assert intro.id == HR_QUESTIONS[0].id and intro.rubric_hints == HR_QUESTIONS[0].rubric_hints


def test_extra_questions_are_not_builtin_so_they_get_stored_with_the_session() -> None:
    extras = set(all_bank()) - built_in_ids()
    assert {"hire_you", "situational_disagree", "learning", "feedback", "proud", "candidate_questions"} <= extras
    assert built_in_ids() == {q.id for q in HR_QUESTIONS}


def test_invalid_sizes_are_rejected() -> None:
    for total in (0, 13):
        with pytest.raises(ValueError):
            plan_realistic(total, None, None, 1)
