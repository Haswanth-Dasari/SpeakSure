"""Tests for offline grammar checks on spoken transcript turns."""

from __future__ import annotations

import re

import pytest

from src.speaksure.analysis.fluency import analyze_fluency
from src.speaksure.analysis.grammar import analyze_grammar
from src.speaksure.speech_to_text import TranscriptWord, TranscriptionResult


_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def make_answer(
    text: str, low_confidence_words: set[str] | None = None
) -> TranscriptionResult:
    """Build a detailed transcript with controllable recognized-word confidence."""
    low_confidence_words = {
        word.casefold() for word in (low_confidence_words or set())
    }
    words = tuple(
        TranscriptWord(
            match.group(),
            float(index),
            float(index + 1),
            0.2 if match.group().casefold() in low_confidence_words else 0.95,
        )
        for index, match in enumerate(_WORD_PATTERN.finditer(text))
    )
    return TranscriptionResult(text, words, 10.0, "en", 0.99, "test")


def test_clean_sentence_has_no_grammar_issues():
    result = analyze_grammar(
        make_answer("I have worked on several software projects during college.")
    )

    assert result["status"] == "ok"
    assert result["error_count"] == 0
    assert result["errors_per_100_words"] == 0.0
    assert result["issues"] == []


@pytest.mark.parametrize(
    ("category", "bad", "good"),
    [
        (
            "subject_verb_agreement",
            "She go to work every day and she likes her job.",
            "She goes to work every day and she likes her job.",
        ),
        (
            "article",
            "I am engineer at a company today.",
            "I am an engineer at a company today.",
        ),
        (
            "verb_tense",
            "Yesterday I go to work after lunch.",
            "Yesterday I went to work after lunch.",
        ),
    ],
)
def test_each_grammar_rule_flags_bad_and_accepts_good_sentence(
    category: str, bad: str, good: str
):
    bad_result = analyze_grammar(make_answer(bad))
    good_result = analyze_grammar(make_answer(good))

    assert category in {issue["category"] for issue in bad_result["issues"]}
    assert category not in {issue["category"] for issue in good_result["issues"]}


def test_wrong_a_an_article_is_reported_with_a_correction():
    result = analyze_grammar(make_answer("I ate a apple after lunch today."))

    issue = next(issue for issue in result["issues"] if issue["category"] == "article")
    assert issue["span_text"] == "apple"
    assert issue["suggestion"] == "an apple"


def test_fillers_and_repeats_are_removed_and_repeats_remain_a_fluency_signal():
    answer = make_answer("Um I, I think I work with the team every day.")

    grammar = analyze_grammar(answer)
    fluency = analyze_fluency(answer, None)

    assert grammar["status"] == "ok"
    assert grammar["word_count_checked"] == 9
    assert grammar["error_count"] == 0
    assert fluency["immediate_repetitions"] == 1


def test_short_turn_skips_model_loading(monkeypatch):
    def fail_if_loaded():
        raise AssertionError("short answer should not load the grammar model")

    monkeypatch.setattr("src.speaksure.analysis.grammar._load_pipeline", fail_if_loaded)

    result = analyze_grammar(make_answer("I work with teams."))

    assert result["status"] == "too_short"
    assert result["word_count_checked"] == 4
    assert result["errors_per_100_words"] is None
    assert result["issues"] == []


def test_valid_spoken_fragment_is_not_reported_as_a_grammar_error():
    result = analyze_grammar(make_answer("Computer Science and Engineering."))

    assert result["status"] == "too_short"
    assert result["error_count"] == 0
    assert result["issues"] == []


def test_low_confidence_word_is_excluded_from_grammar_rules():
    answer = make_answer(
        "Yesterday I go to work after lunch.",
        low_confidence_words={"go"},
    )

    result = analyze_grammar(answer)

    assert result["status"] == "ok"
    assert result["word_count_checked"] == 6
    assert "verb_tense" not in {issue["category"] for issue in result["issues"]}


def test_low_confidence_article_does_not_create_a_missing_article_error():
    answer = make_answer(
        "I am studying in the Rubele QD environment every day.",
        low_confidence_words={"the", "Rubele", "QD"},
    )

    result = analyze_grammar(answer)

    assert result["status"] == "ok"
    assert result["word_count_checked"] == 7
    assert result["error_count"] == 0


def test_missing_model_returns_grammar_unavailable(monkeypatch):
    def missing_model():
        raise OSError("model not installed")

    monkeypatch.setattr("src.speaksure.analysis.grammar._load_pipeline", missing_model)

    result = analyze_grammar(make_answer("I work with a team every day."))

    assert result["status"] == "grammar_unavailable"
    assert result["word_count_checked"] == 7
    assert "en_core_web_sm" in result["message"]
    assert result["errors_per_100_words"] is None
