"""Evaluate the rule-based spoken grammar analyzer on a labelled CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.analysis.grammar import (  # noqa: E402
    GRAMMAR_ANALYZER_VERSION,
    analyze_grammar,
)
from speaksure.speech_to_text import TranscriptionResult  # noqa: E402

CATEGORIES = ("article", "subject_verb_agreement", "verb_tense")
_CSV_FIELDS = ("id", "text", "expected_categories", "kind", "error_span", "note")


def _read_cases(path: Path) -> list[dict[str, str]]:
    """Read labelled cases while allowing explanatory comment header lines."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        content = "".join(line for line in source if not line.startswith("#"))
    reader = csv.DictReader(content.splitlines())
    missing = set(_CSV_FIELDS) - set(reader.fieldnames or ())
    if missing:
        raise ValueError("Grammar cases CSV is missing: " + ", ".join(sorted(missing)))
    return [
        {key: (value or "").strip() for key, value in row.items()}
        for row in reader
    ]


def _span_bounds(text: str, span_text: str) -> tuple[int, int] | None:
    """Find a literal case-insensitive annotation or analyzer span in its text."""
    if not span_text:
        return None
    match = re.search(re.escape(span_text), text, flags=re.IGNORECASE)
    return (match.start(), match.end()) if match else None


def spans_overlap(text: str, expected_span: str, predicted_span: str) -> bool:
    """Return whether the annotated and predicted literal spans overlap."""
    expected = _span_bounds(text, expected_span)
    predicted = _span_bounds(text, predicted_span)
    return bool(
        expected
        and predicted
        and expected[0] < predicted[1]
        and predicted[0] < expected[1]
    )


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Calculate one category's precision, recall, and F1 with zero-safe division."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_cases(
    cases: Sequence[dict[str, str]],
    analyzer: Callable[[TranscriptionResult], dict[str, object]] = analyze_grammar,
) -> dict[str, Any]:
    """Compute span-aware per-category metrics, confusion counts, and FPR."""
    counts = {category: Counter() for category in CATEGORIES}
    confusion: Counter[tuple[str, str]] = Counter()
    correct_cases = 0
    false_positive_cases = 0
    results: list[dict[str, object]] = []
    for case in cases:
        text = case["text"]
        expected = [
            item.strip()
            for item in case["expected_categories"].split(";")
            if item.strip() and item.strip().lower() != "none"
        ]
        if any(category not in CATEGORIES for category in expected):
            raise ValueError(f"Unknown expected category in case {case.get('id')}.")
        if case["kind"] == "correct":
            expected = []
            correct_cases += 1
        answer = TranscriptionResult(text, (), 0.0, "en", 1.0, "evaluation")
        analysis = analyzer(answer)
        issues_value = analysis.get("issues", [])
        issues = [item for item in issues_value if isinstance(item, dict)] if isinstance(issues_value, list) else []
        annotated_span = case.get("error_span", "")
        if annotated_span and _span_bounds(text, annotated_span) is None:
            raise ValueError(f"Case {case.get('id')} error_span is not found in its text.")

        matched_issue_indexes: set[int] = set()
        matched_expected: set[str] = set()
        for category in expected:
            for issue_index, issue in enumerate(issues):
                issue_category = issue.get("category")
                issue_span = str(issue.get("span_text", ""))
                overlap = (
                    spans_overlap(text, annotated_span, issue_span)
                    if annotated_span
                    else True
                )
                if (
                    issue_index not in matched_issue_indexes
                    and issue_category == category
                    and overlap
                ):
                    matched_issue_indexes.add(issue_index)
                    matched_expected.add(category)
                    counts[category]["tp"] += 1
                    break
            if category not in matched_expected:
                counts[category]["fn"] += 1

        for issue_index, issue in enumerate(issues):
            category = issue.get("category")
            if category not in CATEGORIES:
                continue
            if issue_index not in matched_issue_indexes:
                counts[category]["fp"] += 1

        expected_label = ";".join(expected) if expected else "none"
        overlapping_labels = sorted(
            {
                str(issue.get("category"))
                for issue in issues
                if not annotated_span
                or spans_overlap(text, annotated_span, str(issue.get("span_text", "")))
            }
        )
        predicted_label = ";".join(overlapping_labels) if overlapping_labels else "none"
        confusion[(expected_label, predicted_label)] += 1
        if case["kind"] == "correct" and issues:
            false_positive_cases += 1
        results.append(
            {
                "id": case.get("id", ""),
                "expected_categories": expected_label,
                "predicted_categories": ";".join(
                    sorted({str(issue.get("category")) for issue in issues})
                )
                or "none",
                "matched": len(matched_issue_indexes),
                "issue_count": len(issues),
            }
        )

    metrics = {
        category: _precision_recall_f1(
            counts[category]["tp"],
            counts[category]["fp"],
            counts[category]["fn"],
        )
        for category in CATEGORIES
    }
    micro = _precision_recall_f1(
        sum(value["tp"] for value in metrics.values()),
        sum(value["fp"] for value in metrics.values()),
        sum(value["fn"] for value in metrics.values()),
    )
    return {
        "analyzer_version": GRAMMAR_ANALYZER_VERSION,
        "category_metrics": metrics,
        "false_positive_rate": false_positive_cases / correct_cases if correct_cases else 0.0,
        "false_positive_sentences": false_positive_cases,
        "correct_sentences": correct_cases,
        "micro": micro,
        "confusion": [
            {"expected": key[0], "predicted": key[1], "count": value}
            for key, value in sorted(confusion.items())
        ],
        "cases": results,
    }


def _format_table(result: dict[str, Any]) -> str:
    """Render per-category and micro metrics as a Markdown table."""
    rows = [
        "| Category | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, metrics in result["category_metrics"].items():
        rows.append(
            f"| {category} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |"
        )
    micro = result["micro"]
    rows.append(
        f"| **micro average** | {micro['tp']} | {micro['fp']} | {micro['fn']} | "
        f"{micro['precision']:.3f} | {micro['recall']:.3f} | {micro['f1']:.3f} |"
    )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run E2 and write non-overwriting per-case CSV and Markdown results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases_csv", nargs="?", type=Path, default=Path("data/eval/grammar_cases.csv")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/eval/grammar_results.csv")
    )
    args = parser.parse_args(argv)
    markdown_path = args.out.with_suffix(".md")
    if args.out.exists() or markdown_path.exists():
        print(
            f"Results already exist; refusing to overwrite: {args.out} or {markdown_path}",
            file=sys.stderr,
        )
        return 1
    try:
        result = evaluate_cases(_read_cases(args.cases_csv))
        table = _format_table(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            f"# Grammar evaluation (E2)\n\nAnalyzer version: {result['analyzer_version']}\n\n"
            f"{table}\n\nFalse-positive rate on correct sentences: "
            f"{result['false_positive_rate']:.3f} "
            f"({result['false_positive_sentences']}/{result['correct_sentences']}).\n\n"
            "## Confusion (expected vs predicted)\n\n"
            "| Expected | Predicted | Count |\n|---|---|---:|\n"
            + "\n".join(
                f"| {item['expected']} | {item['predicted']} | {item['count']} |"
                for item in result["confusion"]
            )
            + "\n",
            encoding="utf-8",
        )
        with args.out.open("w", encoding="utf-8", newline="") as output:
            output.write(f"# analyzer_version={result['analyzer_version']}\n")
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "id",
                    "expected_categories",
                    "predicted_categories",
                    "matched",
                    "issue_count",
                ),
            )
            writer.writeheader()
            writer.writerows(result["cases"])
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as error:
        print(f"Could not evaluate grammar cases: {error}", file=sys.stderr)
        if markdown_path.exists() and not args.out.exists():
            markdown_path.unlink()
        return 1
    print(f"Grammar analyzer version: {result['analyzer_version']}")
    print(table)
    print(
        f"False-positive rate on correct sentences: {result['false_positive_rate']:.3f} "
        f"({result['false_positive_sentences']}/{result['correct_sentences']})"
    )
    print("Confusion (expected -> predicted):")
    for item in result["confusion"]:
        print(f"  {item['expected']} -> {item['predicted']}: {item['count']}")
    print(f"CSV results: {args.out}")
    print(f"Markdown results: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
