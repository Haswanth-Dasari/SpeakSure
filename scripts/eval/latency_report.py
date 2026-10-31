"""Summarize per-turn response latency from saved session JSON files."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


STAGE_NAMES = ("end_wait_s", "upload_s", "stt_s", "llm_s", "tts_s")
TIMING_ALIASES = {
    "stt_s": "stt_seconds",
    "llm_s": "llm_seconds",
    "tts_s": "tts_synthesis_seconds",
}


def _seconds(value: object) -> float | None:
    """Return finite non-negative seconds, or None when no measurement exists."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _stage_value(timing: dict[str, Any], stage: str) -> float | None:
    """Read a current or legacy timing key from a turn."""
    value = timing.get(stage)
    if stage not in timing and stage in TIMING_ALIASES:
        value = timing.get(TIMING_ALIASES[stage])
    return _seconds(value)


def read_turn_timings(
    paths: Sequence[Path],
) -> list[dict[str, float | None]]:
    """Load per-turn timing values from validated session JSON objects."""
    results: list[dict[str, float | None]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read session JSON {path}: {error}") from error
        if not isinstance(data, dict) or not isinstance(data.get("turns"), list):
            raise ValueError(f"{path} is not a session JSON file with a turns list.")
        for turn_number, turn in enumerate(data["turns"], start=1):
            if not isinstance(turn, dict):
                raise ValueError(f"{path}: turn {turn_number} must be an object.")
            analysis = turn.get("analysis", {})
            timing = analysis.get("timing", {}) if isinstance(analysis, dict) else {}
            if not isinstance(timing, dict):
                continue
            results.append(
                {
                    stage: _stage_value(timing, stage)
                    for stage in STAGE_NAMES
                    if stage != "upload_s" or "upload_s" in timing
                }
            )
    return results


def _percentile_90(values: list[float]) -> float:
    """Return the nearest-rank 90th percentile."""
    ordered = sorted(values)
    rank = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return ordered[rank]


def _summary(values: list[float]) -> tuple[float, float, float] | None:
    """Return median, nearest-rank p90, and maximum for available values."""
    if not values:
        return None
    return statistics.median(values), _percentile_90(values), max(values)


def _total_seconds(turn: dict[str, float | None]) -> float | None:
    """Combine measured end-wait, upload, STT, LLM, and TTS stages."""
    required = ("end_wait_s", "stt_s", "llm_s", "tts_s")
    if any(turn.get(stage) is None for stage in required):
        return None
    upload = turn.get("upload_s")
    if "upload_s" in turn and upload is None:
        return None
    return sum(float(turn[stage]) for stage in required) + (upload or 0.0)


def render_latency_report(
    turns: list[dict[str, float | None]],
) -> str:
    """Render stage and end-of-answer-to-reply latency summaries as Markdown."""
    rows = {
        stage: [
            float(value)
            for turn in turns
            if (value := turn.get(stage)) is not None
        ]
        for stage in STAGE_NAMES
    }
    rows["total_s"] = [
        total
        for turn in turns
        if (total := _total_seconds(turn)) is not None
    ]
    lines = [
        "# Turn latency report",
        "",
        "Total latency is measured from the end of learner speech to the start "
        "of interviewer playback: end wait + browser upload (when measured) + "
        "STT + LLM + TTS synthesis. Percentiles use the nearest-rank method.",
        "",
        "| Stage | Median (s) | P90 (s) | Maximum (s) | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage, values in rows.items():
        summary = _summary(values)
        if summary is None:
            lines.append(f"| {stage} | n/a | n/a | n/a | 0 |")
        else:
            median, percentile, maximum = summary
            lines.append(
                f"| {stage} | {median:.3f} | {percentile:.3f} | "
                f"{maximum:.3f} | {len(values)} |"
            )
    return "\n".join(lines) + "\n"


def _session_paths(input_path: Path) -> list[Path]:
    """Resolve one session JSON file or a directory containing session JSON."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.rglob("*.json"))
        if paths:
            return paths
    raise ValueError(f"No session JSON files found at {input_path}.")


def main(argv: Sequence[str] | None = None) -> int:
    """Print a latency summary and save it beside the input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="a session JSON file or directory")
    args = parser.parse_args(argv)
    try:
        paths = _session_paths(args.input)
        report = render_latency_report(read_turn_timings(paths))
        output_path = (
            args.input / "latency_report.md"
            if args.input.is_dir()
            else args.input.parent / "latency_report.md"
        )
        output_path.write_text(report, encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Latency report failed: {error}", file=sys.stderr)
        return 1
    print(report, end="")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
