"""Tests for exporting and scoring the written review's claims."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.eval.review_narrative import collect_claims, main, score_review, write_rows

NARRATIVE = {
    "language": [{"quote": "I have no disagreements", "better": "I have not had any disagreements", "why": "more natural"}],
    "went_well": [{"point": "Clear project story", "quote": "built a coach"}],
    "work_on": [{"point": "Add a real example", "tip": "Prepare one story", "quote": ""}],
}


def write_analyzed(path: Path, narrative: dict | None) -> Path:
    analysis = {"narrative": narrative} if narrative else {}
    path.write_text(json.dumps({"session_id": "abcdef1234567890", "analysis": analysis}), encoding="utf-8")
    return path


def test_claims_are_exported_per_kind_and_sessions_without_review_are_skipped(tmp_path: Path) -> None:
    good = write_analyzed(tmp_path / "a.analyzed.json", NARRATIVE)
    plain = write_analyzed(tmp_path / "b.analyzed.json", None)
    rows = collect_claims([good, plain])
    assert [r["kind"] for r in rows] == ["language", "went_well", "work_on"]
    assert rows[0]["session"] == "abcdef12" and "better" in rows[0]["claim"]
    assert "[Tip: Prepare one story]" in rows[2]["claim"]
    assert all(r["verdict"] == "" for r in rows)


def test_precision_ignores_unsure_and_is_per_kind(tmp_path: Path) -> None:
    out = tmp_path / "review.csv"
    write_rows(collect_claims([write_analyzed(tmp_path / "a.analyzed.json", NARRATIVE)]), out)
    with out.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    for row, verdict in zip(rows, ["incorrect", "correct", "unsure"]):
        row["verdict"] = verdict
    with out.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    results = score_review(out)
    assert results["language"]["precision"] == 0.0 and results["language"]["n"] == 1
    assert results["went_well"]["precision"] == 1.0
    assert results["work_on"]["n"] == 0 and results["work_on"]["unsure"] == 1


def test_scoring_rejects_blank_or_unknown_verdicts(tmp_path: Path) -> None:
    out = tmp_path / "review.csv"
    write_rows(collect_claims([write_analyzed(tmp_path / "a.analyzed.json", NARRATIVE)]), out)
    with pytest.raises(ValueError):
        score_review(out)  # verdicts are still blank


def test_cli_export_and_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    analyzed = write_analyzed(tmp_path / "a.analyzed.json", NARRATIVE)
    out = tmp_path / "out.csv"
    assert main([str(analyzed), "--out", str(out)]) == 0 and out.exists()
    assert "Wrote 3 claims" in capsys.readouterr().out
    assert main(["--out", str(out)]) == 1
