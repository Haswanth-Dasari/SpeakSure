"""Tests for curated HR questions and deterministic selection."""

import pytest

from src.speaksure.question_bank import HR_QUESTIONS, select_questions


def test_bank_has_exactly_ten_unique_questions_with_hints() -> None:
    """The question bank should contain all ten complete entries."""
    assert len(HR_QUESTIONS) == 10
    assert len({question.id for question in HR_QUESTIONS}) == 10
    assert all(question.rubric_hints for question in HR_QUESTIONS)


@pytest.mark.parametrize("count", range(1, 11))
def test_selection_starts_with_intro_and_respects_count_without_duplicates(
    count: int,
) -> None:
    """Selection should contain the requested number of distinct questions."""
    selected = select_questions(count, seed=23)

    assert selected[0].id == "intro"
    assert len(selected) == count
    assert len({question.id for question in selected}) == count


def test_selection_is_deterministic_for_a_fixed_seed() -> None:
    """A fixed seed should reproduce the same question subset and order."""
    assert select_questions(5, seed=14) == select_questions(5, seed=14)


def test_different_seeds_change_selection() -> None:
    """Different seeds should change the selected subset or ordering."""
    assert select_questions(4, seed=1) != select_questions(4, seed=2)


@pytest.mark.parametrize("count", [0, 11])
def test_selection_rejects_counts_outside_the_bank(count: int) -> None:
    """Counts outside the available bank should raise ValueError."""
    with pytest.raises(ValueError):
        select_questions(count, seed=1)
