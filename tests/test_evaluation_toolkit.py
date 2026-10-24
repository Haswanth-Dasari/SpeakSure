"""Tests for blind rating sheets and grammar evaluation utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.eval_content import _load_ratings, main as eval_content_main
from scripts.eval_grammar import evaluate_cases, spans_overlap
from scripts.make_rating_sheet import build_rating_sheet, main as rating_sheet_main
from scripts.review_grammar_flags import main as review_main, score_review


def test_rating_sheet_is_blind_and_csv_skeleton_uses_question_ids(tmp_path):
    session_path = tmp_path / "blind.json"
    session_path.write_text(
        json.dumps(
            {
                "session_id": "fake-session",
                "turns": [
                    {
                        "index": 1,
                        "question": "Follow-up question text",
                        "answer": {"text": "A detailed answer to read."},
                        "analysis": {"content_ref": {"main_question_id": "intro"}},
                    }
                ],
                "analysis": {
                    "content_by_question": [
                        {
                            "main_question_id": "intro",
                            "turn_indexes": [1],
                            "content_score": 987.654,
                            "reason": "SCORE_SECRET",
                            "strengths": ["FINDING_SECRET"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "ratings"

    assert rating_sheet_main([str(session_path), "--out-dir", str(out_dir)]) == 0

    sheet = (out_dir / "fake-session.sheet.md").read_text(encoding="utf-8")
    assert "Tell me about yourself." in sheet
    assert "A detailed answer to read." in sheet
    assert "Relevance: _  Specificity: _  Clarity: _" in sheet
    assert "Relevance | Does it answer the question asked?" in sheet
    assert "987.654" not in sheet
    assert "SCORE_SECRET" not in sheet
    assert "FINDING_SECRET" not in sheet
    with (out_dir / "fake-session.ratings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    assert rows == [
        {
            "session": "fake-session",
            "question_id": "intro",
            "relevance": "",
            "specificity": "",
            "clarity": "",
            "note": "",
        }
    ]


def test_content_ratings_skip_blanks_and_reject_out_of_range_with_row(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "session,question_id,relevance,specificity,clarity\n"
        "session-a,intro,4,,5\n"
        "session-a,strength,4,3,5\n",
        encoding="utf-8",
    )
    rows, skipped = _load_ratings(ratings)
    assert len(rows) == 1
    assert skipped == 1
    ratings.write_text(
        "session,question_id,relevance,specificity,clarity\n"
        "session-a,intro,6,3,5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"row 2 relevance must be from 1 to 5"):
        _load_ratings(ratings)


def test_content_evaluator_merges_ratings_csv_files(tmp_path, capsys):
    analyzed = tmp_path / "session-a.json"
    analyzed.write_text(
        json.dumps(
            {
                "session_id": "session-a",
                "analysis": {
                    "content_by_question": [
                        {
                            "main_question_id": question_id,
                            "rated": True,
                            "content_score": 60,
                            "relevance": 3,
                            "specificity": 3,
                            "clarity": 3,
                        }
                        for question_id in ("intro", "strength")
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(
        "session,question_id,relevance,specificity,clarity\n"
        "session-a,intro,3,3,3\n",
        encoding="utf-8",
    )
    second.write_text(
        "session,question_id,relevance,specificity,clarity\n"
        "session-a,strength,4,4,4\n",
        encoding="utf-8",
    )

    result = eval_content_main(
        [str(first), str(analyzed), "--merge", str(second)]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert "Matched rows (n): 2" in output
    assert "Skipped rows with blank ratings: 0" in output


def test_grammar_metrics_match_hand_computed_example_and_require_span_overlap():
    cases = [
        {
            "id": "a1",
            "text": "I met engineer today after class.",
            "expected_categories": "article",
            "kind": "error",
            "error_span": "engineer",
        },
        {
            "id": "t1",
            "text": "Yesterday I go to class before lunch.",
            "expected_categories": "verb_tense",
            "kind": "error",
            "error_span": "go",
        },
        {
            "id": "c1",
            "text": "We work with teams every day.",
            "expected_categories": "none",
            "kind": "correct",
            "error_span": "",
        },
    ]

    def fake_analyzer(answer):
        """Produce deterministic category and span output for metric checks."""
        if "engineer" in answer.text:
            issues = [
                {"category": "article", "span_text": "engineer"},
                {"category": "subject_verb_agreement", "span_text": "today"},
            ]
        elif "Yesterday" in answer.text:
            issues = [{"category": "verb_tense", "span_text": "go"}]
        else:
            issues = [{"category": "article", "span_text": "teams"}]
        return {"issues": issues}

    result = evaluate_cases(cases, fake_analyzer)
    article = result["category_metrics"]["article"]
    tense = result["category_metrics"]["verb_tense"]
    # Hand calculation: article TP=1, FP=1 => P=.5/R=1/F1=2/3;
    # tense TP=1; SVA has one FP; micro TP=2, FP=2, FN=0.
    assert article["precision"] == pytest.approx(0.5)
    assert article["recall"] == 1.0
    assert article["f1"] == pytest.approx(2 / 3)
    assert tense["precision"] == tense["recall"] == tense["f1"] == 1.0
    assert result["micro"]["precision"] == pytest.approx(0.5)
    assert result["micro"]["recall"] == 1.0
    assert result["micro"]["f1"] == pytest.approx(2 / 3)
    assert result["false_positive_rate"] == 1.0
    assert spans_overlap("Yesterday I go to class.", "go", "go")
    assert not spans_overlap("Yesterday I go to class.", "go", "Yesterday")


def test_span_mismatch_counts_as_false_positive_and_false_negative():
    case = {
        "id": "span",
        "text": "Yesterday I go to class before lunch.",
        "expected_categories": "verb_tense",
        "kind": "error",
        "error_span": "go",
    }

    result = evaluate_cases(
        [case],
        lambda _answer: {
            "issues": [{"category": "verb_tense", "span_text": "Yesterday"}]
        },
    )

    assert result["category_metrics"]["verb_tense"]["fp"] == 1
    assert result["category_metrics"]["verb_tense"]["fn"] == 1
    assert result["category_metrics"]["verb_tense"]["tp"] == 0


def test_grammar_cases_have_five_each_and_at_least_80_rows():
    path = Path("data/eval/grammar_cases.csv")
    with path.open("r", encoding="utf-8") as source:
        lines = [line for line in source if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert len(rows) >= 80
    for category in ("article", "subject_verb_agreement", "verb_tense"):
        selected = [
            row for row in rows if row["expected_categories"] == category
        ]
        assert len(selected) == 5
        assert all(row["error_span"] in row["text"] for row in selected)
    assert sum(row["kind"] == "correct" for row in rows) >= 35


def test_review_csv_round_trip_and_per_category_precision(tmp_path):
    session = tmp_path / "analyzed.json"
    session.write_text(
        json.dumps(
            {
                "session_id": "real-session",
                "turns": [
                    {
                        "index": 7,
                        "answer": {
                            "text": "Yesterday I go to class after lunch."
                        },
                        "analysis": {
                            "grammar": {
                                "issues": [
                                    {
                                        "category": "verb_tense",
                                        "span_text": "go",
                                        "suggestion": "went",
                                        "explanation": "The time expression suggests past tense.",
                                    }
                                ]
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    review_csv = tmp_path / "review.csv"
    assert review_main([str(session), "--out", str(review_csv)]) == 0
    with review_csv.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["session"] == "real-session"
    assert rows[0]["turn"] == "7"
    assert "[[go]]" in rows[0]["span_text"]
    assert rows[0]["verdict"] == ""
    rows[0]["verdict"] = "correct"
    rows.append({**rows[0], "verdict": "incorrect"})
    rows.append({**rows[0], "verdict": "unsure"})
    with review_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metrics = score_review(review_csv)
    assert metrics["verb_tense"]["correct"] == 1
    assert metrics["verb_tense"]["incorrect"] == 1
    assert metrics["verb_tense"]["unsure"] == 1
    assert metrics["verb_tense"]["n"] == 2
    assert metrics["verb_tense"]["precision"] == 0.5
    assert review_main(["--score", "--score-file", str(review_csv)]) == 0

