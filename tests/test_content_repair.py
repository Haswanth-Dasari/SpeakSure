"""Tests for the lenient repair step that runs before strict content parsing."""

from __future__ import annotations

import json

import pytest

from speaksure.analysis.content import parse_content_result, repair_content_json

HINTS = ("a genuine weakness", "self-awareness")


def _reply(**overrides: object) -> dict[str, object]:
    reply: dict[str, object] = {
        "relevance": 3,
        "specificity": 2,
        "clarity": 4,
        "coverage": [
            {"hint": HINTS[0], "covered": True, "evidence": "I get nervous"},
            {"hint": HINTS[1], "covered": False, "evidence": ""},
        ],
        "star": None,
        "strengths": ["Honest"],
        "improvements": ["Add steps"],
        "reason": "Short answer.",
    }
    reply.update(overrides)
    return reply


def test_repair_fixes_presentation_slips_only() -> None:
    reply = _reply(
        relevance=3.0,
        extra_key=1,
        strengths=["a", "b", "c"],
        coverage=[
            {"hint": "genuine weakness", "covered": True, "evidence": " ".join(["w"] * 20)},
            {"hint": "self awareness", "covered": False, "evidence": "leftover"},
        ],
    )
    with pytest.raises(ValueError):
        parse_content_result(json.dumps(reply), HINTS, False)
    parsed = parse_content_result(
        repair_content_json(json.dumps(reply), HINTS, False), HINTS, False
    )
    assert parsed["relevance"] == 3
    assert parsed["strengths"] == ["a", "b"]
    assert [item["hint"] for item in parsed["coverage"]] == list(HINTS)
    assert len(parsed["coverage"][0]["evidence"].split()) == 15
    assert parsed["coverage"][1]["evidence"] == ""


def test_repair_leaves_unusable_output_unchanged() -> None:
    assert repair_content_json("not json", HINTS, False) == "not json"
    partial = json.dumps({"relevance": 3})
    assert repair_content_json(partial, HINTS, False) == partial
    with pytest.raises(ValueError):
        parse_content_result(repair_content_json(partial, HINTS, False), HINTS, False)


def test_repair_does_not_change_scores_or_coverage_decisions() -> None:
    reply = _reply(relevance=5, clarity=1)
    parsed = parse_content_result(
        repair_content_json(json.dumps(reply), HINTS, False), HINTS, False
    )
    assert (parsed["relevance"], parsed["clarity"]) == (5, 1)
    assert [item["covered"] for item in parsed["coverage"]] == [True, False]
