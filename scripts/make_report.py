"""Generate a standalone HTML feedback report from an analyzed session."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speaksure.question_bank import bank_for_session
from speaksure.report import render_report


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, render the report, and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Create a self-contained HTML report from an analyzed session."
    )
    parser.add_argument("input", type=Path, help="analyzed session JSON file")
    parser.add_argument("--out", type=Path, help="output HTML path")
    parser.add_argument(
        "--generated-at",
        help="report generation timestamp (defaults to current UTC time)",
    )
    args = parser.parse_args(argv)

    try:
        with args.input.open("r", encoding="utf-8") as input_file:
            analyzed = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read analyzed session: {error}", file=sys.stderr)
        return 1
    if not isinstance(analyzed, dict) or not isinstance(analyzed.get("turns"), list):
        print("Invalid analyzed session: expected a JSON object with a turns list.", file=sys.stderr)
        return 1

    output_path = args.out
    if output_path is None:
        output_path = ROOT / "data" / "reports" / f"{args.input.stem}.report.html"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()
    bank_by_id = bank_for_session(analyzed.get("metadata"))
    try:
        report = render_report(analyzed, bank_by_id, generated_at)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Could not generate report: {error}", file=sys.stderr)
        return 1
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
