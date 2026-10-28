"""Export analyzer flags from real sessions or score human verdicts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

_FIELDS = ("session", "turn", "category", "span_text", "suggestion", "verdict", "note")
_VERDICTS = {"correct", "incorrect", "unsure"}


def _surrounding_sentence(text: str, span: str) -> str:
    """Return the sentence containing a flag span, or the full transcript."""
    pieces = text.replace("!", ".").replace("?", ".").split(".")
    for piece in pieces:
        if span.casefold() in piece.casefold():
            sentence = piece.strip() + "."
            start = sentence.casefold().find(span.casefold())
            if start >= 0:
                return (
                    sentence[:start]
                    + "[["
                    + sentence[start : start + len(span)]
                    + "]]"
                    + sentence[start + len(span) :]
                )
            return sentence
    return text.strip()


def collect_flags(session_paths: Sequence[Path]) -> list[dict[str, str]]:
    """Collect all saved grammar issues with the answer sentence as context."""
    rows: list[dict[str, str]] = []
    for path in session_paths:
        with path.open("r", encoding="utf-8") as source:
            session = json.load(source)
        if not isinstance(session, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        session_id = str(session.get("session_id") or path.stem)
        turns = session.get("turns", [])
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            answer = turn.get("answer")
            text = answer.get("text", "") if isinstance(answer, dict) else ""
            analysis = turn.get("analysis", {})
            grammar = analysis.get("grammar", {}) if isinstance(analysis, dict) else {}
            issues = grammar.get("issues", []) if isinstance(grammar, dict) else []
            for issue in issues if isinstance(issues, list) else []:
                if not isinstance(issue, dict):
                    continue
                span = str(issue.get("span_text", ""))
                rows.append(
                    {
                        "session": session_id,
                        "turn": str(turn.get("index", "")),
                        "category": str(issue.get("category", "")),
                        "span_text": _surrounding_sentence(str(text), span),
                        "suggestion": str(issue.get("suggestion", "")),
                        "verdict": "",
                        "note": str(issue.get("explanation", "")),
                    }
                )
    return rows


def score_review(path: Path) -> dict[str, dict[str, int | float]]:
    """Compute human-confirmed precision per issue category."""
    counts: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Review CSV is missing: " + ", ".join(sorted(missing)))
        for line_number, row in enumerate(reader, start=2):
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict not in _VERDICTS:
                raise ValueError(
                    f"Review CSV row {line_number} verdict must be correct, incorrect, or unsure."
                )
            category = (row.get("category") or "").strip()
            values = counts.setdefault(category, {"correct": 0, "incorrect": 0, "unsure": 0})
            values[verdict] += 1
    return {
        category: {
            **values,
            "n": values["correct"] + values["incorrect"],
            "precision": (
                values["correct"] / (values["correct"] + values["incorrect"])
                if values["correct"] + values["incorrect"]
                else 0.0
            ),
        }
        for category, values in sorted(counts.items())
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Export grammar flags or score already reviewed verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, nargs="*")
    parser.add_argument(
        "--out", type=Path, default=Path("data/eval/grammar_flags_review.csv")
    )
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score-file", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.score:
            results = score_review(args.score_file or args.out)
            print("Category | Correct | Incorrect | Unsure | n | Precision")
            for category, metrics in results.items():
                print(
                    f"{category} | {metrics['correct']} | {metrics['incorrect']} | "
                    f"{metrics['unsure']} | {metrics['n']} | {metrics['precision']:.3f}"
                )
            return 0
        if not args.sessions:
            parser.error("provide one or more analyzed session JSON files, or use --score")
        if args.out.exists():
            print(f"Review file already exists; skipped: {args.out}")
            return 0
        rows = collect_flags(args.sessions)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Review file written: {args.out} ({len(rows)} flags)")
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not process grammar review: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
