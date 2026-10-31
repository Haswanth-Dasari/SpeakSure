"""Export the written review's claims for human checking, or score the verdicts.

The report's opening review is written by a language model from computed facts.
This tool measures how trustworthy it is: for every language note, strength and
improvement in saved analyzed sessions, a person marks the claim as correct,
incorrect or unsure, and precision is reported per kind of claim.

Export:  python scripts/eval/review_narrative.py data/analysis/*.analyzed.json
Score:   python scripts/eval/review_narrative.py --score
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

FIELDS = ("session", "kind", "claim", "evidence", "verdict", "note")
VERDICTS = {"correct", "incorrect", "unsure"}
KINDS = ("language", "went_well", "work_on")


def collect_claims(analyzed_paths: Sequence[Path]) -> list[dict[str, str]]:
    """Return one review row per claim in each analyzed session."""
    rows: list[dict[str, str]] = []
    for path in analyzed_paths:
        with path.open("r", encoding="utf-8") as source:
            data: Any = json.load(source)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object.")
        analysis = data.get("analysis")
        narrative = analysis.get("narrative") if isinstance(analysis, dict) else None
        if not isinstance(narrative, dict):
            continue
        session = str(data.get("session_id") or path.stem)[:8]
        for item in narrative.get("language", []) or []:
            if isinstance(item, dict):
                rows.append(
                    {
                        "session": session,
                        "kind": "language",
                        "claim": f"You said \"{item.get('quote', '')}\" -> better: \"{item.get('better', '')}\"",
                        "evidence": str(item.get("why", "")),
                        "verdict": "",
                        "note": "",
                    }
                )
        for kind in ("went_well", "work_on"):
            for item in narrative.get(kind, []) or []:
                if isinstance(item, dict):
                    claim = str(item.get("point", ""))
                    if kind == "work_on" and item.get("tip"):
                        claim += f"  [Tip: {item['tip']}]"
                    rows.append(
                        {
                            "session": session,
                            "kind": kind,
                            "claim": claim,
                            "evidence": str(item.get("quote", "")),
                            "verdict": "",
                            "note": "",
                        }
                    )
    return rows


def write_rows(rows: Sequence[dict[str, str]], path: Path) -> None:
    """Write review rows as a UTF-8 CSV that opens correctly in Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def score_review(path: Path) -> dict[str, dict[str, int | float]]:
    """Return precision per kind, ignoring 'unsure' verdicts in the denominator."""
    counts: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Review CSV is missing: " + ", ".join(sorted(missing)))
        for line_number, row in enumerate(reader, start=2):
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict not in VERDICTS:
                raise ValueError(
                    f"Row {line_number}: verdict must be correct, incorrect or unsure."
                )
            kind = (row.get("kind") or "").strip()
            values = counts.setdefault(kind, {"correct": 0, "incorrect": 0, "unsure": 0})
            values[verdict] += 1
    results: dict[str, dict[str, int | float]] = {}
    for kind in sorted(counts):
        values = counts[kind]
        decided = values["correct"] + values["incorrect"]
        results[kind] = {
            **values,
            "n": decided,
            "precision": values["correct"] / decided if decided else 0.0,
        }
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Export claims for review or print precision from a completed CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyzed", type=Path, nargs="*", help="*.analyzed.json files")
    parser.add_argument("--out", type=Path, default=Path("data/eval/narrative_review.csv"))
    parser.add_argument("--score", action="store_true", help="score the completed CSV")
    args = parser.parse_args(argv)
    try:
        if args.score:
            results = score_review(args.out)
            print("Kind | Correct | Incorrect | Unsure | n | Precision")
            for kind, metrics in results.items():
                print(
                    f"{kind} | {metrics['correct']} | {metrics['incorrect']} | "
                    f"{metrics['unsure']} | {metrics['n']} | {metrics['precision']:.2f}"
                )
            return 0
        if not args.analyzed:
            print("Pass one or more *.analyzed.json files to export.", file=sys.stderr)
            return 1
        rows = collect_claims(args.analyzed)
        write_rows(rows, args.out)
        print(f"Wrote {len(rows)} claims to {args.out}. Fill the verdict column, then run with --score.")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not complete: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
