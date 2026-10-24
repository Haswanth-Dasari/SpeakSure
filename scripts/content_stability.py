"""Summarize repeated per-question content-score and rubric-component spread."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


CONTENT_LOW = 40.0
CONTENT_HIGH = 70.0
COMPONENTS = ("relevance", "coverage", "specificity", "clarity", "star")


def band(score: float) -> str:
    """Return the report's band for a content score."""
    if score < CONTENT_LOW:
        return "Needs work"
    if score < CONTENT_HIGH:
        return "Developing"
    return "Strong"


def _group_key(group: dict[str, Any]) -> tuple[object, tuple[object, ...]]:
    """Identify a question group consistently across saved runs."""
    indexes = group.get("turn_indexes", [])
    return (
        group.get("main_question_id"),
        tuple(indexes) if isinstance(indexes, list) else (),
    )


def _spread(values: list[float]) -> dict[str, float | int]:
    """Calculate population spread statistics for a non-empty numeric sample."""
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "standard_deviation": statistics.pstdev(values),
        "range": max(values) - min(values),
    }


def stability_rows(analyzed: dict[str, Any]) -> list[dict[str, object]]:
    """Return score and rubric-component spread rows for all saved question runs."""
    analysis = analyzed.get("analysis", {})
    runs = analysis.get("content_runs")
    if not isinstance(runs, list):
        raise ValueError("Analyzed session does not contain an analysis.content_runs list.")
    groups_by_run: list[
        dict[tuple[object, tuple[object, ...]], dict[str, Any]]
    ] = []
    all_keys: set[tuple[object, tuple[object, ...]]] = set()
    for run in runs:
        groups = run.get("content_by_question", []) if isinstance(run, dict) else []
        by_key = {
            _group_key(group): group
            for group in groups
            if isinstance(group, dict)
        }
        all_keys.update(by_key)
        groups_by_run.append(by_key)

    rows: list[dict[str, object]] = []
    for key in all_keys:
        groups = [run.get(key) for run in groups_by_run]
        question_id = str(key[0] or "unknown")
        scored = [
            (run_index, group)
            for run_index, group in enumerate(groups, start=1)
            if group is not None
            and group.get("rated") is True
            and isinstance(group.get("content_score"), (int, float))
        ]
        scores = [float(group["content_score"]) for _, group in scored]
        if scores:
            median_score = statistics.median(scores)
            median_band = band(
                min(scored, key=lambda item: abs(float(item[1]["content_score"]) - median_score))[1][
                    "content_score"
                ]
            )
            score_spread = _spread(scores)
            score_spread.update(
                {
                    "question_id": question_id,
                    "metric": "content_score",
                    "band_disagreements": sum(
                        band(score) != median_band for score in scores
                    ),
                }
            )
            rows.append(score_spread)
        for component in COMPONENTS:
            values = [
                float(component_values[component])
                for _, group in enumerate(groups, start=1)
                if group is not None
                and group.get("rated") is True
                and isinstance(
                    component_values := group.get("content_components"), dict
                )
                and isinstance(component_values.get(component), (int, float))
            ]
            if values:
                component_spread = _spread(values)
                component_spread.update(
                    {
                        "question_id": question_id,
                        "metric": component,
                        "band_disagreements": None,
                    }
                )
                rows.append(component_spread)
    return rows


def markdown_table(rows: list[dict[str, object]]) -> str:
    """Format spread rows as a Markdown table."""
    headers = (
        "Question",
        "Metric",
        "n",
        "Min",
        "Max",
        "Mean",
        "Std dev",
        "Range",
        "Band disagreements",
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = (
            str(row["question_id"]),
            str(row["metric"]),
            str(row["n"]),
            f"{row['min']:.3f}",
            f"{row['max']:.3f}",
            f"{row['mean']:.3f}",
            f"{row['standard_deviation']:.3f}",
            f"{row['range']:.3f}",
            "-" if row["band_disagreements"] is None else str(row["band_disagreements"]),
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Load one analyzed session, print stability statistics, and write Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyzed_json", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.analyzed_json.open("r", encoding="utf-8") as source:
            analyzed = json.load(source)
        rows = stability_rows(analyzed)
        table = markdown_table(rows)
        output_path = args.analyzed_json.with_name(
            f"{args.analyzed_json.stem}.content_stability.md"
        )
        output_path.write_text(
            "# Content score stability\n\n"
            "Band disagreements are counted against the observed run nearest the median score.\n\n"
            + table
            + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not analyze content stability: {error}", file=sys.stderr)
        return 1
    print(table)
    print(f"Markdown: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
