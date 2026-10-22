"""Dependency-free inline SVG charts for interview delivery metrics."""

from __future__ import annotations

from html import escape
from statistics import mean
from typing import Any


def small_bar_chart(
    analyzed: dict[str, Any],
    metric: str,
    title: str,
    *,
    source: str = "fluency",
) -> str:
    """Render one SVG bar or labelled gap for each question group."""
    turns = analyzed.get("turns", [])
    groups = analyzed.get("analysis", {}).get("content_by_question", [])
    values = [
        (
            str(group.get("main_question_id") or f"Question {index + 1}"),
            _group_value(group, turns, metric, source),
        )
        for index, group in enumerate(groups)
    ]
    if not values:
        values = []
    width = max(240, 88 * len(values) + 32)
    height = 170
    top = 30
    baseline = 124
    max_value = max((value for _, value in values if value is not None), default=0.0)
    usable_height = baseline - top
    bar_width = min(46, max(18, 54 // max(1, len(values))))
    parts = [
        f'<svg role="img" '
        f'aria-label="{escape(title, quote=True)}" viewBox="0 0 {width} {height}">',
        f"<title>{escape(title)}</title>",
        '<line x1="12" y1="124" x2="100%" y2="124" stroke="#64748b" stroke-width="1"/>',
    ]
    for index, (label, value) in enumerate(values):
        center = 44 + index * 88
        safe_label = escape(label)
        if value is None:
            parts.extend(
                [
                    f'<g class="bar-entry" aria-label="{safe_label}: no data">',
                    f'<text x="{center}" y="82" text-anchor="middle" class="value">no data</text>',
                    f'<text x="{center}" y="145" text-anchor="middle" class="question">{safe_label}</text>',
                    "</g>",
                ]
            )
            continue
        bar_height = 0.0 if max_value <= 0 else usable_height * value / max_value
        y = baseline - bar_height
        value_label = f"{value:.2f}".rstrip("0").rstrip(".")
        parts.extend(
            [
                f'<g class="bar-entry" aria-label="{safe_label}: {escape(value_label)}">',
                f'<rect x="{center - bar_width / 2:g}" y="{y:.2f}" width="{bar_width}" '
                f'height="{bar_height:.2f}" fill="#2563eb"/>',
                f'<text x="{center}" y="{max(top - 5, y - 5):.2f}" text-anchor="middle" class="value">'
                f"{escape(value_label)}</text>",
                f'<text x="{center}" y="145" text-anchor="middle" class="question">{safe_label}</text>',
                "</g>",
            ]
        )
    if not values:
        parts.append('<text x="16" y="82" class="value">No question groups</text>')
    parts.extend(
        [
            '<style>.value{font:12px sans-serif;fill:#0f172a}.question{font:11px sans-serif;fill:#334155}</style>',
            "</svg>",
        ]
    )
    return "".join(parts)


def words_per_minute_chart(analyzed: dict[str, Any]) -> str:
    """Render a question-group chart of mean transcript words per minute."""
    return small_bar_chart(analyzed, "words_per_minute", "Words per minute")


def audio_pause_ratio_chart(analyzed: dict[str, Any]) -> str:
    """Render a question-group chart of mean audio pause ratio."""
    return small_bar_chart(
        analyzed, "pause_ratio", "Audio pause ratio", source="audio_delivery"
    )


def level_variation_chart(analyzed: dict[str, Any]) -> str:
    """Render a question-group chart of mean audio level variation in dB."""
    return small_bar_chart(
        analyzed, "level_std_db", "Audio level variation (dB)", source="audio_delivery"
    )


def _group_value(
    group: dict[str, Any],
    turns: list[dict[str, Any]],
    metric: str,
    source: str,
) -> float | None:
    """Calculate the mean available metric for the turns in one group."""
    metric_values: list[float] = []
    for turn_index in group.get("turn_indexes", []):
        if not isinstance(turn_index, int) or not 1 <= turn_index <= len(turns):
            continue
        metrics = turns[turn_index - 1].get("analysis", {}).get(source, {})
        value = metrics.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metric_values.append(float(value))
    return mean(metric_values) if metric_values else None
