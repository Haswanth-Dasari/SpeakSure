"""Tests for choosing resume terms to add to the speech recognizer's vocabulary."""

from __future__ import annotations

from src.speaksure.resume import (
    MAX_RESUME_HOTWORDS,
    MAX_SESSION_HINTS,
    merge_hints,
    resume_hotwords,
    validate_profile,
)

PROFILE = validate_profile(
    {
        "education": ["B.Tech in Computer Science, IIITDM Kurnool, 2027"],
        "skills": ["Python", "FastAPI", "spaCy", "communication", "Docker", "SQL"],
        "projects": [
            {"name": "SpeakSure", "summary": "A coach.", "technologies": ["Piper TTS", "Python"]},
            {"name": "Inventory Tool", "summary": "Stock tracking.", "technologies": ["Flask"]},
        ],
        "experience": ["Intern at Zeta Labs working on NLP"],
    }
)


def test_distinctive_terms_are_chosen_in_priority_order() -> None:
    words = resume_hotwords(PROFILE)
    assert words[:3] == ["SpeakSure", "Inventory Tool", "IIITDM Kurnool"]
    assert {"Piper TTS", "Flask", "FastAPI", "spaCy", "Docker", "Zeta Labs"} <= set(words)


def test_common_and_generic_words_are_left_out() -> None:
    words = {w.casefold() for w in resume_hotwords(PROFILE)}
    assert not words & {"python", "sql", "communication", "computer science", "b.tech", "intern"}
    assert resume_hotwords(validate_profile({"skills": ["communication", "teamwork", "Python", "SQL"]})) == []


def test_the_list_is_short_unique_and_each_term_is_valid() -> None:
    profile = validate_profile(
        {
            "skills": [f"Tool{i}X" for i in range(30)],
            "projects": [{"name": "Alpha Project"}, {"name": "alpha project"}],
        }
    )
    words = resume_hotwords(profile)
    assert len(words) == MAX_RESUME_HOTWORDS
    assert len({w.casefold() for w in words}) == len(words)
    assert all(1 <= len(w) <= 40 and len(w.split()) <= 4 for w in words)


def test_merge_keeps_user_hints_first_without_duplicates_and_respects_the_cap() -> None:
    merged = merge_hints(["Haswanth", "speaksure"], ["SpeakSure", "FastAPI"])
    assert merged == ["Haswanth", "speaksure", "FastAPI"]
    many = merge_hints([f"user{i}" for i in range(15)], [f"extra{i}" for i in range(15)])
    assert len(many) == MAX_SESSION_HINTS and many[0] == "user0"
