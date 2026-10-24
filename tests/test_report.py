"""Tests for heuristic coaching, SVG charts, and standalone HTML reports."""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET

import pytest

from scripts.make_report import main
from src.speaksure.question_bank import BankQuestion, HR_QUESTIONS
from src.speaksure.report.charts import (
    audio_pause_ratio_chart,
    level_variation_chart,
    words_per_minute_chart,
)
from src.speaksure.report.aggregate import aggregate_group_delivery
from src.speaksure.report.coaching import Finding, build_findings
from src.speaksure.report.render import _append_summary, render_report


BANK = {question.id: question for question in HR_QUESTIONS}


def make_analyzed(
    *,
    question_id: str | None = "intro",
    score: float = 60,
    turn_count: int = 1,
    content: bool = True,
    answered: bool = True,
) -> dict:
    """Build a minimal report-shaped session with tunable metrics."""
    turns = []
    for index in range(1, turn_count + 1):
        turns.append(
            {
                "index": index,
                "question": "Tell me about yourself." if index == 1 else "Could you add detail?",
                "asked_at": "2026-01-02T03:04:05+00:00",
                "answer": (
                    {
                        "text": "A useful answer with a concrete example.",
                        "words": [
                            {"text": "A ", "start": 0.0, "end": 0.1, "probability": 0.9},
                            {"text": "useful ", "start": 0.1, "end": 0.2, "probability": 0.4},
                            {"text": "answer.", "start": 0.2, "end": 0.4, "probability": 0.95},
                        ],
                        "duration_seconds": 1.0,
                        "language": "en",
                        "language_probability": 1.0,
                        "device": "test",
                    }
                    if answered
                    else None
                ),
                "audio_path": None,
                "thinking_time_seconds": 1.0,
                "answer_duration_seconds": 1.0,
                "analysis": {
                    "asked": {
                        "main_question_id": question_id,
                        "is_follow_up": index > 1,
                    },
                    "fluency": {
                        "has_answer": answered,
                        "word_count": 40,
                        "speech_seconds": 17.14,
                        "words_per_minute": 140.0,
                        "pause_count": 1,
                        "longest_pause_seconds": 0.5,
                        "pause_ratio": 0.1,
                        "filler_count": 0,
                        "fillers_per_100_words": 0.0,
                        "hedge_count": 0,
                        "hedges_per_100_words": 0.0,
                        "thinking_time_seconds": 1.0,
                    },
                    "audio_delivery": {
                        "pause_count": 1,
                        "longest_pause_seconds": 0.5,
                        "pause_ratio": 0.1,
                        "total_pause_seconds": 0.1,
                        "speech_span_seconds": 1.0,
                        "articulation_rate_wpm": 150.0,
                        "voiced_seconds": 1.0,
                        "level_std_db": 3.0,
                    },
                },
            }
        )
    analysis = {}
    if content:
        question = BANK.get(question_id) if question_id is not None else None
        hints = question.rubric_hints if question else ("a useful point",)
        analysis["content_by_question"] = [
            {
                "main_question_id": question_id,
                "turn_indexes": list(range(1, turn_count + 1)),
                "label_source": "recorded",
                "rated": True,
                "asr_mean_probability": 0.9,
                "asr_reliability": "high",
                "relevance": 4,
                "specificity": 4,
                "clarity": 4,
                "content_score": score,
                "coverage": [
                    {
                        "hint": hint,
                        "covered": False,
                        "evidence": "",
                        "evidence_verified": False,
                    }
                    for hint in hints
                ],
                "star": None,
                "strengths": ["Clear answer"],
                "improvements": ["Add one more detail"],
                "reason": "The answer is clear.",
            }
        ]
    return {
        "schema_version": 1,
        "session_id": "0123456789abcdef",
        "created_at": "2026-01-02T03:04:05+00:00",
        "mode": "live",
        "difficulty": "medium",
        "planned_questions": 3,
        "metadata": {
            "role": "engineer",
            "stt_model": "whisper-base",
            "llm_model": "test-model",
            "tts_voice": "test-voice",
        },
        "turns": turns,
        "analysis": analysis,
    }


def set_metric(analyzed: dict, metric: str, value: object, source: str = "fluency") -> None:
    """Set one metric on the first answer of a fixture."""
    analyzed["turns"][0]["analysis"][source][metric] = value


@pytest.mark.parametrize(
    ("metric", "value", "source", "text"),
    [
        ("words_per_minute", 180, "fluency", "pace"),
        ("words_per_minute", 100, "fluency", "pace"),
        ("pause_ratio", 0.3, "audio_delivery", "pauses"),
        ("longest_pause_seconds", 2.0, "audio_delivery", "pauses"),
        ("hedges_per_100_words", 5, "fluency", "hedging"),
        ("fillers_per_100_words", 4, "fluency", "under-detected"),
        ("word_count", 24, "fluency", "very short"),
        ("thinking_time_seconds", 5.1, "fluency", "thinking time"),
    ],
)
def test_delivery_rules_emit_expected_findings(metric, value, source, text):
    analyzed = make_analyzed()
    set_metric(analyzed, metric, value, source)
    if metric == "words_per_minute":
        analyzed["turns"][0]["analysis"]["fluency"]["speech_seconds"] = (
            40 * 60 / value
        )
    elif metric == "pause_ratio":
        analyzed["turns"][0]["analysis"]["audio_delivery"].update(
            total_pause_seconds=value, speech_span_seconds=1.0
        )

    findings = build_findings(analyzed, BANK)

    assert any(text in finding.text or text in finding.evidence for finding in findings)
    if metric == "fillers_per_100_words":
        assert any("under-detected" in finding.evidence for finding in findings)


def test_content_low_lists_bank_hints_as_points_for_next_time():
    analyzed = make_analyzed(score=39)

    finding = next(
        item for item in build_findings(analyzed, BANK) if item.area == "content"
    )

    assert finding.severity == 3
    assert "points next time" in finding.text
    assert BANK["intro"].rubric_hints[0] in finding.text


def test_strong_content_uses_a_verified_evidence_quote():
    analyzed = make_analyzed(score=70)
    coverage = analyzed["analysis"]["content_by_question"][0]["coverage"]
    coverage[0].update(
        covered=True,
        evidence="I built a useful tool",
        evidence_verified=True,
    )

    finding = next(
        item for item in build_findings(analyzed, BANK) if item.kind == "strength"
    )

    assert finding.area == "content"
    assert finding.evidence == "I built a useful tool"


def test_behavioral_question_missing_star_parts_creates_priority():
    analyzed = make_analyzed(question_id="teamwork")
    group = analyzed["analysis"]["content_by_question"][0]
    group["star"] = {
        "situation": True,
        "task": False,
        "action": True,
        "result": False,
    }

    finding = next(
        item
        for item in build_findings(analyzed, BANK)
        if "STAR parts" in item.text
    )

    assert "task" in finding.text
    assert "result" in finding.text


def test_low_recognition_reliability_creates_transcript_priority():
    analyzed = make_analyzed()
    analyzed["analysis"]["content_by_question"][0]["asr_reliability"] = "low"

    finding = next(
        item
        for item in build_findings(analyzed, BANK)
        if item.area == "transcript"
    )

    assert finding.severity == 3
    assert "scores may be understated" in finding.text


def test_findings_sort_by_severity_then_interview_position():
    analyzed = make_analyzed(turn_count=2)
    first, second = analyzed["analysis"]["content_by_question"][0], None
    first.update(main_question_id="intro", turn_indexes=[1], rated=True)
    second = {
        **first,
        "main_question_id": "strength",
        "turn_indexes": [2],
        "asr_reliability": "low",
    }
    analyzed["analysis"]["content_by_question"] = [first, second]
    set_metric(analyzed, "word_count", 20)

    findings = build_findings(analyzed, BANK)

    assert [finding.severity for finding in findings] == sorted(
        (finding.severity for finding in findings), reverse=True
    )
    low_priority = next(finding for finding in findings if finding.area == "transcript")
    assert findings.index(low_priority) < next(
        index for index, finding in enumerate(findings) if "very short" in finding.text
    )


def test_summary_limits_each_finding_kind_to_three():
    analyzed = make_analyzed()
    analyzed["analysis"]["content_by_question"][0]["content_score"] = 75
    analyzed["analysis"]["content_by_question"][0]["coverage"][0].update(
        covered=True,
        evidence="Evidence quote",
        evidence_verified=True,
    )
    for index in range(4):
        turn = copy.deepcopy(analyzed["turns"][0])
        turn["analysis"]["fluency"]["words_per_minute"] = 140
        turn["analysis"]["asked"]["main_question_id"] = "intro"
        analyzed["turns"].append(turn)
        analyzed["analysis"]["content_by_question"].append(
            {
                **analyzed["analysis"]["content_by_question"][0],
                "main_question_id": "intro",
                "turn_indexes": [index + 2],
            }
        )

    html = render_report(analyzed, BANK, "fixed")

    assert html.count('class="finding strength"') <= 3
    assert html.count('class="finding priority"') <= 3


def test_html_escapes_transcript_question_evidence_and_model_feedback():
    analyzed = make_analyzed()
    attack = "<script>alert(1)</script>"
    analyzed["turns"][0]["answer"]["text"] = attack
    group = analyzed["analysis"]["content_by_question"][0]
    group["coverage"][0].update(
        covered=True, evidence=attack, evidence_verified=True
    )
    group["strengths"] = [attack]
    custom_bank = {
        "intro": BankQuestion("intro", "test", attack, ("a point",))
    }

    html = render_report(analyzed, custom_bank, "fixed")

    assert "<script" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "http://" not in html
    assert "https://" not in html


def test_transcript_highlights_low_confidence_and_unrated_group_renders():
    analyzed = make_analyzed()
    analyzed["analysis"]["content_by_question"][0].update(
        rated=False, reason="unknown_question"
    )

    html = render_report(analyzed, BANK, "fixed")

    assert '<mark title="Recognition probability: 0.40">useful </mark>' in html
    assert "Not rated: the question could not be identified (unknown_question)." in html


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (39.9, "Needs work"),
        (40, "Developing"),
        (69.9, "Developing"),
        (70, "Strong"),
    ],
)
def test_content_band_boundaries(score, band):
    analyzed = make_analyzed(score=score)

    html = render_report(analyzed, BANK, "fixed")

    assert f"<strong>{band}</strong>" in html


def test_missing_content_analysis_renders_and_does_not_mutate_input():
    analyzed = make_analyzed(content=False)
    original = copy.deepcopy(analyzed)

    html = render_report(analyzed, BANK, "fixed")

    assert "Content analysis was not run for this session." in html
    assert "Content analysis was not run.</p>" in html
    assert analyzed == original


def test_no_answered_questions_renders_short_message():
    analyzed = make_analyzed(content=False, answered=False)

    html = render_report(analyzed, BANK, "fixed")

    assert "No answered questions are available for this session." in html


def test_none_metrics_render_as_dashes_and_charts_have_labelled_gaps():
    analyzed = make_analyzed()
    set_metric(analyzed, "words_per_minute", None)
    set_metric(analyzed, "pause_ratio", None, "audio_delivery")
    set_metric(analyzed, "level_std_db", None, "audio_delivery")
    set_metric(analyzed, "speech_seconds", None)
    set_metric(analyzed, "total_pause_seconds", None, "audio_delivery")
    set_metric(analyzed, "speech_span_seconds", None, "audio_delivery")
    set_metric(analyzed, "voiced_seconds", None, "audio_delivery")

    html = render_report(analyzed, BANK, "fixed")
    charts = (
        words_per_minute_chart(analyzed),
        audio_pause_ratio_chart(analyzed),
        level_variation_chart(analyzed),
    )

    assert "<td>-</td>" in html
    assert all("no data" in chart for chart in charts)
    assert all(len(ET.fromstring(chart).findall(".//g")) == 1 for chart in charts)


def test_charts_are_valid_svg_with_one_entry_per_question_group():
    analyzed = make_analyzed(turn_count=3)
    groups = analyzed["analysis"]["content_by_question"]
    groups.extend(
        [
            {"main_question_id": "strength", "turn_indexes": [2]},
            {"main_question_id": "conflict", "turn_indexes": [3]},
        ]
    )

    for chart in (
        words_per_minute_chart(analyzed),
        audio_pause_ratio_chart(analyzed),
        level_variation_chart(analyzed),
    ):
        root = ET.fromstring(chart)
        assert root.tag == "svg"
        assert len(root.findall(".//g[@class='bar-entry']")) == 3
        assert root.find("title") is not None
        assert root.attrib["aria-label"]


def test_group_delivery_uses_pooled_words_per_minute():
    turns = [
        {"analysis": {"fluency": {"word_count": 17, "speech_seconds": 5.42}}},
        {"analysis": {"fluency": {"word_count": 37, "speech_seconds": 16.32}}},
    ]

    result = aggregate_group_delivery(turns)

    assert result["words_per_minute"] == pytest.approx(149.0, abs=0.1)
    assert result["total_words"] == 54
    assert result["speech_seconds"] == pytest.approx(21.74)


def test_turn_without_speech_seconds_is_excluded_from_pooled_pace():
    turns = [
        {"analysis": {"fluency": {"word_count": 100, "speech_seconds": None}}},
        {"analysis": {"fluency": {"word_count": 10, "speech_seconds": 4.0}}},
    ]

    result = aggregate_group_delivery(turns)

    assert result["total_words"] == 110
    assert result["words_per_minute"] == pytest.approx(150.0)


def test_group_pause_ratio_is_pooled_and_level_variation_is_weighted():
    turns = [
        {
            "analysis": {
                "audio_delivery": {
                    "pause_count": 2,
                    "longest_pause_seconds": 1.5,
                    "total_pause_seconds": 1.0,
                    "speech_span_seconds": 4.0,
                    "voiced_seconds": 1.0,
                    "level_std_db": 4.0,
                }
            }
        },
        {
            "analysis": {
                "audio_delivery": {
                    "pause_count": 1,
                    "longest_pause_seconds": 2.0,
                    "total_pause_seconds": 3.0,
                    "speech_span_seconds": 6.0,
                    "voiced_seconds": 3.0,
                    "level_std_db": 12.0,
                }
            }
        },
    ]

    result = aggregate_group_delivery(turns)

    assert result["pause_count"] == 3
    assert result["longest_pause_seconds"] == 2.0
    assert result["pause_ratio"] == pytest.approx(4 / 10)
    assert result["level_std_db"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("field", "category", "sentence"),
    [
        ("reason", "too_short_or_empty", "the answer was too short to rate"),
        ("error", "unknown_question", "the question could not be identified"),
        ("error", "llm_error", "the language model was unavailable or returned unusable output"),
        ("error", "invalid_output", "the language model was unavailable or returned unusable output"),
        ("error", "unexpected", "the answer could not be rated"),
    ],
)
def test_unrated_categories_render_reader_facing_reason(field, category, sentence):
    analyzed = make_analyzed()
    group = analyzed["analysis"]["content_by_question"][0]
    group.update(rated=False, reason=None, error=None)
    group[field] = category

    html = render_report(analyzed, BANK, "fixed")

    assert f"Not rated: {sentence}" in html
    assert category in html


def test_unrated_too_short_group_creates_content_priority():
    analyzed = make_analyzed()
    group = analyzed["analysis"]["content_by_question"][0]
    group.update(rated=False, reason="too_short_or_empty")

    finding = next(
        item for item in build_findings(analyzed, BANK)
        if item.area == "content"
    )

    assert finding.severity == 2
    assert finding.text == (
        "On the introduction question, your answer was too short to rate; "
        "give a fuller answer with a concrete example."
    )


def test_findings_use_topic_labels_and_never_embed_question_text():
    analyzed = make_analyzed()
    findings = build_findings(analyzed, BANK)

    assert findings
    assert all("On the introduction question," in finding.text for finding in findings)
    assert all("Tell me about yourself." not in finding.text for finding in findings)


def test_hedge_strength_requires_minimum_words_per_group_and_in_total():
    analyzed = make_analyzed(turn_count=2)
    groups = analyzed["analysis"]["content_by_question"]
    groups[:] = [
        {**groups[0], "main_question_id": "intro", "turn_indexes": [1]},
        {**groups[0], "main_question_id": "strength", "turn_indexes": [2]},
    ]
    for turn, words in zip(analyzed["turns"], [10, 10]):
        turn["analysis"]["fluency"].update(word_count=words, hedge_count=0)

    findings = build_findings(analyzed, BANK)
    assert not any("kept hedging" in finding.text for finding in findings)

    for turn in analyzed["turns"]:
        turn["analysis"]["fluency"].update(word_count=20, hedge_count=0)
    findings = build_findings(analyzed, BANK)
    assert any("kept hedging" in finding.text for finding in findings)


def test_summary_spreads_findings_across_groups_with_two_per_group_limit():
    findings = [
        Finding("priority", 3, "content", "first", "", 1),
        Finding("priority", 2, "content", "second", "", 1),
        Finding("priority", 1, "content", "third", "", 1),
        Finding("priority", 1, "content", "other group", "", 2),
    ]
    document = []

    _append_summary(document, findings)

    selected = "".join(document)
    assert selected.count("<p>first</p>") == 1
    assert selected.count("<p>second</p>") == 1
    assert selected.count("<p>third</p>") == 0
    assert selected.count("<p>other group</p>") == 1


def test_summary_may_return_fewer_findings_when_fewer_exist():
    document = []
    _append_summary(
        document,
        [Finding("priority", 1, "content", "single", "", 1)],
    )

    assert "".join(document).count("<div class=\"finding priority\">") == 1


def test_cards_show_topic_without_internal_question_id():
    html = render_report(make_analyzed(), BANK, "fixed")

    assert '<p class="muted">Topic: introduction</p>' in html
    assert "Question ID" not in html


def test_score_variation_note_and_content_run_range():
    analyzed = make_analyzed()
    html = render_report(analyzed, BANK, "fixed")
    assert (
        "Approximate: scores from a language model can vary by about 10 points "
        "between runs."
    ) in html

    analyzed["analysis"]["content_runs"] = [
        {
            "content_by_question": [
                {"main_question_id": "intro", "content_score": 68.1}
            ]
        },
        {
            "content_by_question": [
                {"main_question_id": "intro", "content_score": 77.8}
            ]
        },
    ]
    html = render_report(analyzed, BANK, "fixed")
    assert "range 68.1-77.8 over 2 runs" in html
    assert "Approximate: scores from a language model" not in html


def test_cli_writes_default_and_explicit_output_deterministically(tmp_path, monkeypatch, capsys):
    analyzed = make_analyzed()
    source = tmp_path / "input.analyzed.json"
    source.write_text(json.dumps(analyzed), encoding="utf-8")
    monkeypatch.setattr("scripts.make_report.ROOT", tmp_path)

    assert main([str(source), "--generated-at", "fixed"]) == 0
    default_path = tmp_path / "data" / "reports" / "input.analyzed.report.html"
    first = default_path.read_text(encoding="utf-8")
    assert str(default_path) in capsys.readouterr().out
    assert main([str(source), "--generated-at", "fixed"]) == 0
    assert default_path.read_text(encoding="utf-8") == first

    explicit = tmp_path / "custom" / "feedback.html"
    assert main([str(source), "--out", str(explicit), "--generated-at", "fixed"]) == 0
    assert explicit.read_text(encoding="utf-8") == first


def test_cli_fails_cleanly_for_bad_input(tmp_path, capsys):
    invalid = tmp_path / "bad.json"
    invalid.write_text("{", encoding="utf-8")

    assert main([str(invalid)]) == 1
    captured = capsys.readouterr()
    assert "Could not read analyzed session" in captured.err
    assert "Traceback" not in captured.err


def test_group_grammar_rate_pools_errors_over_checked_words():
    turns = [
        {
            "analysis": {
                "grammar": {
                    "status": "ok",
                    "word_count_checked": 100,
                    "error_count": 2,
                    "issues": [
                        {"category": "article"},
                        {"category": "article"},
                    ],
                }
            }
        },
        {
            "analysis": {
                "grammar": {
                    "status": "ok",
                    "word_count_checked": 300,
                    "error_count": 1,
                    "issues": [{"category": "verb_tense"}],
                }
            }
        },
        {
            "analysis": {
                "grammar": {
                    "status": "too_short",
                    "word_count_checked": 4,
                    "error_count": 0,
                    "issues": [],
                }
            }
        },
    ]

    result = aggregate_group_delivery(turns)

    assert result["grammar_word_count_checked"] == 400
    assert result["grammar_error_count"] == 3
    assert result["grammar_errors_per_100_words"] == 0.75
    assert result["grammar_category_counts"] == {"article": 2, "verb_tense": 1}


def test_grammar_priority_uses_most_frequent_category_and_speaker_example():
    analyzed = make_analyzed(content=False)
    analyzed["turns"][0]["analysis"]["grammar"] = {
        "status": "ok",
        "word_count_checked": 30,
        "error_count": 2,
        "errors_per_100_words": 6.67,
        "issues": [
            {
                "category": "article",
                "span_text": "engineer",
                "suggestion": "an engineer",
                "explanation": "A singular count noun needs an article.",
                "word_index": 3,
            },
            {
                "category": "article",
                "span_text": "designer",
                "suggestion": "a designer",
                "explanation": "A singular count noun needs an article.",
                "word_index": 8,
            },
        ],
    }

    finding = next(
        item
        for item in build_findings(analyzed, BANK)
        if item.area == "grammar" and item.kind == "priority"
    )

    assert "article use" in finding.text
    assert "most frequent" in finding.text
    assert '"engineer" -> "an engineer"' in finding.evidence


def test_grammar_block_escapes_examples_and_shows_pooled_feedback():
    analyzed = make_analyzed()
    analyzed["turns"][0]["analysis"]["grammar"] = {
        "status": "ok",
        "word_count_checked": 10,
        "error_count": 1,
        "errors_per_100_words": 10.0,
        "issues": [
            {
                "category": "article",
                "span_text": "<script>alert(1)</script>",
                "suggestion": "an engineer",
                "explanation": "Use the correct article.",
                "word_index": 3,
            }
        ],
    }

    html = render_report(analyzed, BANK, "fixed")

    assert "<h4>Grammar</h4>" in html
    assert "10 detected errors per 100 checked words" in html
    assert "Top categories: article use (1)." in html
    assert "<script" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (
            "grammar_unavailable",
            "Install spaCy and its en_core_web_sm model to enable offline grammar checks.",
        ),
        (
            "too_short",
            "Grammar analysis was skipped because all answers had fewer than 5 checked words.",
        ),
    ],
)
def test_grammar_unavailable_and_short_states_render_as_sentences(status, message):
    analyzed = make_analyzed()
    analyzed["turns"][0]["analysis"]["grammar"] = {
        "status": status,
        "word_count_checked": 0 if status == "too_short" else 8,
        "error_count": 0,
        "errors_per_100_words": None,
        "issues": [],
    }

    html = render_report(analyzed, BANK, "fixed")

    assert message in html
