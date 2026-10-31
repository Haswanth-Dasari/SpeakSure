"""Create blind reading sheets and CSV skeletons for analyzed sessions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.question_bank import HR_QUESTIONS  # noqa: E402

RATING_FIELDS = ("session", "question_id", "relevance", "specificity", "clarity", "note")
_RUBRIC = """| Scale | What to rate | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Relevance | Does it answer the question asked? | Not at all | Mostly off-topic | Partly answers | Mostly answers | Directly answers |
| Specificity | Concrete examples, numbers, outcomes | None | Very vague | Some detail | Concrete detail | Specific evidence and outcomes |
| Clarity | Easy to follow, well organised | Very hard | Often unclear | Understandable | Clear and organised | Exceptionally clear and organised |
"""


def _groups(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Return content groups, or reconstruct groups from per-turn references."""
    analysis = session.get("analysis", {})
    groups = analysis.get("content_by_question") if isinstance(analysis, dict) else None
    if isinstance(groups, list):
        return [item for item in groups if isinstance(item, dict)]
    grouped: dict[str, dict[str, Any]] = {}
    for turn in session.get("turns", []):
        if not isinstance(turn, dict):
            continue
        turn_analysis = turn.get("analysis", {})
        reference = turn_analysis.get("content_ref", {}) if isinstance(turn_analysis, dict) else {}
        asked = turn_analysis.get("asked", {}) if isinstance(turn_analysis, dict) else {}
        question_id = reference.get("main_question_id") or asked.get("main_question_id")
        if not isinstance(question_id, str):
            continue
        group = grouped.setdefault(
            question_id, {"main_question_id": question_id, "turn_indexes": []}
        )
        group["turn_indexes"].append(turn.get("index"))
    return list(grouped.values())


def _question_text(question_id: str, turns: list[dict[str, Any]], indexes: list[int]) -> str:
    """Find the bank question text or fall back to the group's first turn."""
    bank = {question.id: question.text for question in HR_QUESTIONS}
    if question_id in bank:
        return bank[question_id]
    by_index = {turn.get("index"): turn for turn in turns}
    for index in indexes:
        turn = by_index.get(index)
        if isinstance(turn, dict) and isinstance(turn.get("question"), str):
            return turn["question"]
    return "Question text unavailable."


def build_rating_sheet(session: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Build a blind Markdown reading sheet and empty rating rows."""
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Analyzed session has no valid session_id.")
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        raise ValueError("Analyzed session turns must be a list.")
    by_index = {
        turn.get("index"): turn for turn in turns if isinstance(turn, dict)
    }
    lines = [
        "# Blind answer rating sheet",
        "",
        "Rate each answer group independently. Do not consult automated feedback while rating.",
        "",
        "## Rubric (1 = lowest, 5 = highest)",
        "",
        _RUBRIC.rstrip(),
        "",
    ]
    rows: list[dict[str, str]] = []
    for group in _groups(session):
        question_id = group.get("main_question_id")
        if not isinstance(question_id, str):
            continue
        indexes = group.get("turn_indexes", [])
        if not isinstance(indexes, list):
            indexes = []
        lines.extend(
            [
                f"## Question: {_question_text(question_id, turns, indexes)}",
                "",
                f"Question ID: `{question_id}`",
                "",
            ]
        )
        for attempt, index in enumerate(indexes, start=1):
            turn = by_index.get(index)
            answer_data = turn.get("answer") if isinstance(turn, dict) else None
            answer_text = (
                answer_data.get("text")
                if isinstance(answer_data, dict) and isinstance(answer_data.get("text"), str)
                else ""
            )
            lines.extend(
                [
                    f"**Answer {attempt} (turn {index}):**",
                    "",
                    answer_text or "_No transcript recorded._",
                    "",
                ]
            )
        lines.extend(
            [
                "Relevance: _  Specificity: _  Clarity: _",
                "",
                "---",
                "",
            ]
        )
        rows.append(
            {
                "session": session_id,
                "question_id": question_id,
                "relevance": "",
                "specificity": "",
                "clarity": "",
                "note": "",
            }
        )
    return "\n".join(lines), rows


def write_session_outputs(session_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write non-overwriting blind sheet and ratings skeleton for one session."""
    with session_path.open("r", encoding="utf-8") as source:
        session = json.load(source)
    if not isinstance(session, dict):
        raise ValueError(f"{session_path} must contain a JSON object.")
    sheet, rows = build_rating_sheet(session)
    session_id = str(session.get("session_id", session_path.stem))
    sheet_path = out_dir / f"{session_id}.sheet.md"
    ratings_path = out_dir / f"{session_id}.ratings.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if sheet_path.exists():
        print(f"Sheet already exists; skipped: {sheet_path}")
    else:
        sheet_path.write_text(sheet, encoding="utf-8")
        print(f"Sheet written: {sheet_path}")
    if ratings_path.exists():
        print(f"Ratings CSV already exists; skipped: {ratings_path}")
    else:
        with ratings_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=RATING_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Ratings CSV written: {ratings_path}")
    return sheet_path, ratings_path


def main(argv: Sequence[str] | None = None) -> int:
    """Create one blind sheet and ratings CSV skeleton per analyzed session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyzed_json", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval/ratings"))
    args = parser.parse_args(argv)
    try:
        for session_path in args.analyzed_json:
            write_session_outputs(session_path, args.out_dir)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not create rating sheet: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
