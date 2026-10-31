"""Tests for hiding the rule-based grammar section from reports only."""

from __future__ import annotations

import copy

import pytest

from src.speaksure.question_bank import bank_for_session
from src.speaksure.report.options import rule_grammar_visible, without_rule_grammar
from src.speaksure.report.render import render_report


def session_data() -> dict:
    grammar = {
        "analyzer_version": "x", "status": "ok", "word_count_checked": 12, "error_count": 1,
        "errors_per_100_words": 8.3,
        "issues": [{"category": "article", "span_text": "end", "suggestion": "an end",
                    "explanation": "A singular noun needs an article."}],
    }
    return {
        "session_id": "a" * 32, "created_at": "2026-10-30T10:00:00", "planned_questions": 1,
        "metadata": {"role": "x"},
        "turns": [{"index": 1, "question": "Tell me about yourself.",
                   "answer": {"text": "I built things from beginning to end.", "words": [],
                              "duration_seconds": 4.0, "language": "en",
                              "language_probability": 1.0, "model": "m"},
                   "analysis": {"grammar": grammar}}],
        "analysis": {"content_by_question": [{"main_question_id": "intro", "turn_indexes": [1],
                                              "rated": False, "error": "too_short_or_empty"}]},
    }


def test_default_shows_and_env_can_hide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHOW_RULE_GRAMMAR", raising=False)
    assert rule_grammar_visible() is True
    monkeypatch.setenv("SHOW_RULE_GRAMMAR", "0")
    assert rule_grammar_visible() is False
    monkeypatch.setenv("SHOW_RULE_GRAMMAR", "1")
    assert rule_grammar_visible() is True


def test_removal_works_on_a_copy_and_keeps_the_original() -> None:
    original = session_data()
    hidden = without_rule_grammar(original)
    assert "grammar" in original["turns"][0]["analysis"]
    assert "rule_grammar_hidden" not in original["analysis"]
    assert "grammar" not in hidden["turns"][0]["analysis"]
    assert hidden["analysis"]["rule_grammar_hidden"] is True


def test_report_shows_grammar_normally_and_omits_it_when_hidden() -> None:
    bank = bank_for_session({})
    shown = render_report(copy.deepcopy(session_data()), bank, "now")
    assert "<h4>Grammar</h4>" in shown
    hidden = render_report(without_rule_grammar(session_data()), bank, "now")
    assert "<h4>Grammar</h4>" not in hidden and "an end" not in hidden
