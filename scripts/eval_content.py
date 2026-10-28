"""Compare student rubric ratings with saved model content scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

CONTENT_LOW = 40.0
CONTENT_HIGH = 70.0
RATING_COLUMNS = ("relevance", "specificity", "clarity")


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks, assigning tied values their mean rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    """Compute Spearman's rank correlation using average ranks for ties."""
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal lengths.")
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = statistics.mean(left_ranks)
    right_mean = statistics.mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left_ranks)
    right_ss = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def _band(score: float) -> str:
    """Map a 0-100 score into the same bands used by the report."""
    if score < CONTENT_LOW:
        return "Needs work"
    if score < CONTENT_HIGH:
        return "Developing"
    return "Strong"


def _load_ratings(path: Path) -> tuple[list[dict[str, str]], int]:
    """Load complete rating rows and count rows with any blank rating."""
    with path.open("r", encoding="utf-8-sig", newline="") as rating_file:
        reader = csv.DictReader(rating_file)
        required = {"session", "question_id", *RATING_COLUMNS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Ratings CSV is missing required column(s): "
                + ", ".join(sorted(missing))
            )
        rows = []
        skipped = 0
        for line_number, source in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in source.items()}
            if any(not row.get(column) for column in RATING_COLUMNS):
                skipped += 1
                continue
            for column in RATING_COLUMNS:
                try:
                    rating = int(row[column])
                except (KeyError, ValueError):
                    raise ValueError(
                        f"Ratings CSV line {line_number} has an invalid {column} rating."
                    ) from None
                if not 1 <= rating <= 5:
                    raise ValueError(
                        f"Ratings CSV row {line_number} {column} must be from 1 to 5; "
                        f"got {row[column]!r}."
                    )
            rows.append(row)
    return rows, skipped


def _session_aliases(path: Path, analyzed: dict[str, Any]) -> set[str]:
    """Return the session id and filename aliases used to match ratings."""
    session_id = analyzed.get("session_id")
    return {value for value in (path.name, path.stem, session_id) if isinstance(value, str)}


def evaluate_ratings(
    ratings: Sequence[dict[str, str]],
    analyzed_files: Sequence[Path],
) -> dict[str, Any]:
    """Match student ratings with model groups and compute agreement metrics."""
    sessions: dict[str, dict[str, Any]] = {}
    for path in analyzed_files:
        with path.open("r", encoding="utf-8") as analyzed_file:
            analyzed = json.load(analyzed_file)
        for alias in _session_aliases(path, analyzed):
            sessions[alias] = analyzed

    matched: list[dict[str, float]] = []
    unrated: list[dict[str, str]] = []
    for row in ratings:
        session_value = row["session"]
        session_path = Path(session_value)
        analyzed = (
            sessions.get(session_value)
            or sessions.get(session_path.name)
            or sessions.get(session_path.stem)
        )
        if analyzed is None:
            unrated.append(
                {
                    "session": row["session"],
                    "question_id": row["question_id"],
                    "reason": "analyzed session file not provided or session ID did not match",
                }
            )
            continue
        groups = analyzed.get("analysis", {}).get("content_by_question", [])
        group = next(
            (
                item
                for item in groups
                if isinstance(item, dict)
                and item.get("main_question_id") == row["question_id"]
            ),
            None,
        )
        if group is None:
            unrated.append(
                {
                    "session": row["session"],
                    "question_id": row["question_id"],
                    "reason": "question group not found in analyzed session",
                }
            )
            continue
        if group.get("rated") is not True or not isinstance(
            group.get("content_score"), (int, float)
        ):
            reason = group.get("reason") or group.get("error") or "model group is unrated"
            unrated.append(
                {
                    "session": row["session"],
                    "question_id": row["question_id"],
                    "reason": str(reason),
                }
            )
            continue
        values = {
            column: float(row[column])
            for column in RATING_COLUMNS
        }
        student_mean = statistics.mean(values.values())
        matched.append(
            {
                "student_mean": student_mean,
                "student_scaled": (student_mean - 1) / 4 * 100,
                "model_score": float(group["content_score"]),
                "band_agreement": float(
                    _band((student_mean - 1) / 4 * 100)
                    == _band(float(group["content_score"]))
                ),
                **{
                    f"student_{column}": values[column]
                    for column in RATING_COLUMNS
                },
                **{
                    f"model_{column}": float(group[column])
                    for column in RATING_COLUMNS
                    if isinstance(group.get(column), (int, float))
                },
            }
        )

    student_means = [row["student_mean"] for row in matched]
    model_scores = [row["model_score"] for row in matched]
    differences = [
        abs(row["student_scaled"] - row["model_score"]) for row in matched
    ]
    component_correlations: dict[str, dict[str, float | int | None]] = {}
    for column in RATING_COLUMNS:
        component_rows = [
            row for row in matched if f"model_{column}" in row
        ]
        component_correlations[column] = {
            "n": len(component_rows),
            "spearman": spearman_correlation(
                [row[f"student_{column}"] for row in component_rows],
                [row[f"model_{column}"] for row in component_rows],
            ),
        }
    return {
        "n": len(matched),
        "overall_spearman": spearman_correlation(student_means, model_scores),
        "component_correlations": component_correlations,
        "mean_absolute_difference": statistics.mean(differences) if differences else None,
        "band_agreements": sum(int(row["band_agreement"]) for row in matched),
        "unrated": unrated,
    }


def _display_number(value: object) -> str:
    """Render absent correlations and differences as an explicit dash."""
    return "-" if value is None else f"{float(value):.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    """Read ratings and analyzed sessions, then print E3 agreement statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ratings_csv", type=Path)
    parser.add_argument("analyzed_json", type=Path, nargs="+")
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        default=[],
        metavar="CSV",
        help="additional ratings CSV files (put this option after analyzed JSON paths)",
    )
    args = parser.parse_args(argv)
    try:
        rating_paths = [args.ratings_csv, *args.merge]
        ratings: list[dict[str, str]] = []
        skipped = 0
        for rating_path in rating_paths:
            file_rows, file_skipped = _load_ratings(rating_path)
            ratings.extend(file_rows)
            skipped += file_skipped
        result = evaluate_ratings(
            ratings,
            args.analyzed_json,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not evaluate content ratings: {error}", file=sys.stderr)
        return 1

    print(f"Matched rows (n): {result['n']}")
    print(f"Skipped rows with blank ratings: {skipped}")
    if result["n"] < 5:
        print("Warning: correlations are not meaningful with fewer than 5 matched rows.")
    print(f"Overall Spearman: {_display_number(result['overall_spearman'])}")
    for component, metrics in result["component_correlations"].items():
        print(
            f"{component.title()} Spearman (n={metrics['n']}): "
            f"{_display_number(metrics['spearman'])}"
        )
    print(
        "Mean absolute difference (student rescaled to 0-100): "
        f"{_display_number(result['mean_absolute_difference'])}"
    )
    print(f"Band agreements: {result['band_agreements']} / {result['n']}")
    print(f"Unrated/excluded rows: {len(result['unrated'])}")
    for item in result["unrated"]:
        print(
            f"  {item['session']} / {item['question_id']}: {item['reason']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
