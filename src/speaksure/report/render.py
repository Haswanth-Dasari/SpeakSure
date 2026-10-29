"""Render an analyzed interview as a self-contained, escaped HTML report."""

from __future__ import annotations

from html import escape
from typing import Any, Mapping

from ..analysis.question_labels import resolve_asked_questions
from ..question_bank import BankQuestion
from ..session_log import InterviewSession
from .aggregate import aggregate_group_delivery
from .charts import audio_pause_ratio_chart, level_variation_chart, words_per_minute_chart
from .coaching import (
    CONTENT_HIGH,
    CONTENT_LOW,
    GRAMMAR_CATEGORY_LABELS,
    MAX_FINDINGS_PER_QUESTION,
    Finding,
    TOPIC_LABELS,
    build_findings,
)

# Word probabilities below this threshold receive visual transcript highlighting.
LOW_CONFIDENCE_WORD_PROBABILITY = 0.5
SCORE_VARIATION_NOTE = (
    "Approximate: scores from a language model can vary by about 10 points between runs."
)

_UNRATED_SENTENCES = {
    "too_short_or_empty": "the answer was too short to rate",
    "unknown_question": "the question could not be identified",
    "llm_error": (
        "the language model was unavailable or returned unusable output for this question"
    ),
    "invalid_output": (
        "the language model was unavailable or returned unusable output for this question"
    ),
}

_CSS = """
body{font:16px/1.5 system-ui,sans-serif;color:#172033;margin:0;background:#f4f6fa}
main{max-width:1100px;margin:auto;padding:1.5rem}
header,.card,section{background:#fff;border:1px solid #dbe2ea;border-radius:10px;padding:1rem 1.25rem;margin:1rem 0}
h1,h2,h3{line-height:1.2}h1{margin-top:0}
.metadata,.summary-grid,.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.75rem}
.badge{display:inline-block;border-radius:1rem;background:#e2e8f0;padding:.15rem .65rem;font-weight:600}
.badge.low{background:#fee2e2}.badge.medium{background:#fef3c7}.badge.high{background:#dcfce7}
.finding{padding:.5rem .75rem;margin:.5rem 0;border-left:4px solid #2563eb;background:#f8fafc}
.finding.priority{border-color:#b45309}.finding.strength{border-color:#15803d}
.transcript{white-space:pre-wrap;background:#f8fafc;padding:.75rem;border-radius:6px}
mark{background:#fde68a}table{border-collapse:collapse;width:100%;margin:.75rem 0;display:block;overflow-x:auto}
th,td{border:1px solid #cbd5e1;padding:.4rem .55rem;text-align:left;vertical-align:top}
th{background:#eaf0f6}ul{padding-left:1.4rem}.charts figure{margin:.2rem;overflow-x:auto}
svg{max-width:100%;height:auto}.muted{color:#475569}
footer{margin:2rem 0;color:#475569}
@media print{body{background:#fff}.card,section,header{break-inside:avoid;box-shadow:none}main{max-width:none;padding:.4in}}
"""


def render_report(
    analyzed: dict[str, Any],
    bank_by_id: Mapping[str, BankQuestion],
    generated_at: str,
) -> str:
    """Render a complete standalone HTML5 feedback report."""
    session_analysis = analyzed.get("analysis", {})
    content_groups = session_analysis.get("content_by_question")
    content_available = isinstance(content_groups, list)
    if not content_available:
        content_groups = _infer_groups(analyzed, bank_by_id)
    groups = content_groups
    findings = build_findings(analyzed, bank_by_id)
    turns = analyzed.get("turns", [])
    metadata = analyzed.get("metadata", {})
    session_id = _as_text(analyzed.get("session_id", ""))
    date = _as_text(analyzed.get("created_at", ""))[:10] or "Unknown"
    role = _as_text(metadata.get("role", "Not specified"))
    planned = analyzed.get("planned_questions")
    question_count = str(planned) if isinstance(planned, int) else str(len(groups))

    document = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\';">',
        "<title>Mock interview feedback</title>",
        f"<style>{_CSS}</style></head><body><main>",
        "<header><h1>Mock interview feedback</h1>",
        '<div class="metadata">',
        _metadata_item("Date", date),
        _metadata_item("Role", role),
        _metadata_item("Questions", question_count),
        _metadata_item("Session", session_id[:8] or "Unknown"),
        _metadata_item("STT model", _stt_model_label(metadata)),
        _metadata_item(
            "LLM model",
            _model_value(metadata.get("llm_model") or _content_model(groups)),
        ),
        _metadata_item(
            "TTS model",
            _model_value(metadata.get("tts_model", metadata.get("tts_voice"))),
        ),
        "</div></header>",
        "<section><h2>How to read this report</h2>",
        "<p>Scores are approximate guides.</p>",
        "<p>Delivery numbers are speech measurements, not emotion detection.</p>",
        "<p>Names and technical terms may be misrecognised by speech recognition.</p>",
        "</section>",
        "<section><h2>Summary</h2>",
    ]
    if not content_available:
        document.append("<p>Content analysis was not run for this session.</p>")
    _append_summary(document, findings)
    document.append("</section><section><h2>Question-by-question feedback</h2>")
    answered_groups = [
        group
        for group in groups
        if any(_has_answer(turns[index - 1]) for index in group.get("turn_indexes", [])
               if isinstance(index, int) and 1 <= index <= len(turns))
    ]
    if not answered_groups:
        document.append("<p>No answered questions are available for this session.</p>")
    else:
        for index, group in enumerate(answered_groups, start=1):
            document.append(
                _render_group(
                    group,
                    index,
                    turns,
                    bank_by_id,
                    content_available,
                    session_analysis,
                )
            )
    document.extend(
        [
            "</section><section><h2>Delivery across the interview</h2>",
            "<p>These charts show delivery signals, not a complete evaluation.</p>",
            '<div class="charts">',
            f"<figure><figcaption>Words per minute</figcaption>{words_per_minute_chart(_chart_input(analyzed, groups))}</figure>",
            f"<figure><figcaption>Audio pause ratio</figcaption>{audio_pause_ratio_chart(_chart_input(analyzed, groups))}</figure>",
            f"<figure><figcaption>Level variation</figcaption>{level_variation_chart(_chart_input(analyzed, groups))}</figure>",
            "</div></section>",
            "<section><h2>Method and limitations</h2><ul>",
            "<li>Speech recognition can misrecognise names, technical terms and hesitations; low-confidence words are highlighted.</li>",
            "<li>Scores from a language model vary a little between runs.</li>",
            "<li>Pauses are measured from the audio, not from word timestamps.</li>",
            "<li>Filler words may be under-detected because recognition tends to drop them.</li>",
            "<li>Grammar feedback uses a small offline rule set; it may miss context or reflect transcript recognition errors.</li>",
            "<li>No emotion is detected.</li>",
            "<li>Thresholds in the coaching rules are heuristics.</li>",
            "</ul></section>",
            f"<footer>Generated at {escape(generated_at)}</footer>",
            "</main></body></html>",
        ]
    )
    return "".join(document)


def _append_summary(document: list[str], findings: list[Finding]) -> None:
    """Append up to three prioritized findings of each kind with group diversity."""
    for kind, heading in (("strength", "Strengths"), ("priority", "Priorities")):
        candidates = [finding for finding in findings if finding.kind == kind]
        selected: list[Finding] = []
        selected_indexes: set[int] = set()
        group_counts: dict[int | str, int] = {}
        for index, finding in enumerate(candidates):
            if len(selected) == 3:
                break
            key: int | str = (
                finding.group_key
                if finding.group_key is not None
                else f"ungrouped:{index}"
            )
            if group_counts.get(key, 0) >= MAX_FINDINGS_PER_QUESTION:
                continue
            selected.append(finding)
            selected_indexes.add(index)
            group_counts[key] = group_counts.get(key, 0) + 1
        if len(selected) < 3:
            for index, finding in enumerate(candidates):
                if len(selected) == 3:
                    break
                if index not in selected_indexes:
                    selected.append(finding)
        document.append(f"<h3>{heading}</h3>")
        if not selected:
            document.append("<p>None identified.</p>")
            continue
        for finding in selected:
            document.append(_finding_html(finding))


def _finding_html(finding: Finding) -> str:
    """Render a finding with safely escaped text and evidence."""
    return (
        f'<div class="finding {finding.kind}"><p>{escape(finding.text)}</p>'
        f"<p><strong>Evidence:</strong> {escape(finding.evidence)}</p></div>"
    )


def _render_group(
    group: dict[str, Any],
    display_index: int,
    turns: list[dict[str, Any]],
    bank_by_id: Mapping[str, BankQuestion],
    content_available: bool,
    session_analysis: dict[str, Any],
) -> str:
    """Render question, transcripts, rubric results, and delivery metrics."""
    turn_indexes = [
        index
        for index in group.get("turn_indexes", [])
        if isinstance(index, int) and 1 <= index <= len(turns)
    ]
    question_id = group.get("main_question_id")
    question = bank_by_id.get(question_id) if isinstance(question_id, str) else None
    question_text = question.text if question is not None else "Question not identified"
    topic = _topic_label(question_id)
    reliability = _as_text(group.get("asr_reliability", "unknown")).lower()
    badge_class = reliability if reliability in {"low", "medium", "high"} else ""
    html_parts = [
        f'<article class="card"><h3>Question {display_index}: {escape(question_text)}</h3>',
        f'<p class="muted">Topic: {escape(topic)}</p>',
        f'<p><span class="badge {badge_class}">Recognition confidence: {escape(reliability)}</span></p>',
    ]
    labels = _turn_labels(group, turns)
    for index in turn_indexes:
        turn = turns[index - 1]
        if labels.get(index, False):
            html_parts.append(
                f"<p><strong>Follow-up question asked:</strong> "
                f"{escape(_as_text(turn.get('question', '')))}</p>"
            )
        answer = turn.get("answer")
        html_parts.append(f"<h4>Answer {index}</h4><p class=\"transcript\">{_transcript_html(answer)}</p>")

    if content_available:
        if group.get("rated") is True:
            for field_name, label in (
                ("relevance", "Relevance"),
                ("specificity", "Specificity"),
                ("clarity", "Clarity"),
            ):
                html_parts.append(
                    f"<p>{label}: {_rating(group.get(field_name))} / 5</p>"
                )
            score = _number(group.get("content_score"))
            score_variation = _score_variation(session_analysis, question_id)
            variation_text = (
                score_variation
                if score_variation is not None
                else SCORE_VARIATION_NOTE
                if not isinstance(session_analysis.get("content_runs"), list)
                else ""
            )
            sample_scores = group.get("content_score_samples")
            if isinstance(sample_scores, list) and sample_scores:
                shown = ", ".join(_format_number(_number(item)) for item in sample_scores)
                variation_text = (
                    f"Median of {len(sample_scores)} ratings (scores: {shown})."
                )
            html_parts.append(
                f"<p>Content score: {_format_number(score)} / 100 "
                f"(<strong>{_content_band(score)}</strong>)"
                f"{f' <span class=\"muted\">{escape(variation_text)}</span>' if variation_text else ''}</p>"
            )
            html_parts.append("<h4>Rubric hints</h4><ul>")
            for entry in group.get("coverage", []):
                covered = entry.get("covered") is True
                status = "Covered" if covered else "Not covered"
                html_parts.append(
                    f"<li><strong>{status}</strong>: {escape(_as_text(entry.get('hint', '')))}"
                )
                evidence = entry.get("evidence")
                if isinstance(evidence, str) and evidence:
                    html_parts.append(f"<br>Evidence: {escape(evidence)}")
                if entry.get("evidence_verified") is False:
                    html_parts.append("<br>quote not found in transcript")
                html_parts.append("</li>")
            html_parts.append("</ul>")
            star = group.get("star")
            if question_id in {"teamwork", "conflict", "challenge", "mistake"} and isinstance(star, dict):
                html_parts.append("<h4>STAR checklist</h4><ul>")
                for part in ("situation", "task", "action", "result"):
                    status = "Covered" if star.get(part) is True else "Not covered"
                    html_parts.append(f"<li>{escape(part.title())}: {status}</li>")
                html_parts.append("</ul>")
            html_parts.extend(
                [
                    _render_feedback_list("Strengths", group.get("strengths", [])),
                    _render_feedback_list("Improvements", group.get("improvements", [])),
                ]
            )
        else:
            category = _reason_category(group)
            sentence = _UNRATED_SENTENCES.get(category, "the answer could not be rated")
            category_note = f" ({escape(category)})" if category else ""
            html_parts.append(
                f"<p>Not rated: {escape(sentence)}{category_note}.</p>"
            )
    else:
        html_parts.append("<p>Content analysis was not run.</p>")

    html_parts.append(_grammar_block(turn_indexes, turns))
    html_parts.append(_group_metrics_table(turn_indexes, turns))
    html_parts.append(_answer_metrics_table(turn_indexes, turns))
    html_parts.append("</article>")
    return "".join(html_parts)


def _grammar_block(
    turn_indexes: list[int], turns: list[dict[str, Any]]
) -> str:
    """Render pooled grammar feedback and readable unavailable states."""
    group_turns = [turns[index - 1] for index in turn_indexes]
    results = [
        turn.get("analysis", {}).get("grammar")
        for turn in group_turns
        if isinstance(turn.get("analysis", {}).get("grammar"), dict)
    ]
    if not results:
        return (
            "<h4>Grammar</h4><p>Grammar analysis results were not saved "
            "for these answers.</p>"
        )

    statuses = [result.get("status") for result in results]
    aggregate = aggregate_group_delivery(group_turns)
    if aggregate["grammar_word_count_checked"]:
        count = int(aggregate["grammar_error_count"])
        checked = int(aggregate["grammar_word_count_checked"])
        rate = _number(aggregate["grammar_errors_per_100_words"])
        parts = [
            "<h4>Grammar</h4>",
            f"<p>{_format_number(rate)} detected errors per 100 checked words "
            f"({count} issue(s) across {checked} checked words).</p>",
        ]
        if count == 0:
            parts.append("<p>No grammar issues were detected in the checked words.</p>")
            parts.append("<p>Top categories: none.</p>")
        else:
            category_counts = aggregate["grammar_category_counts"]
            top_categories = sorted(
                category_counts.items(), key=lambda item: (-item[1], item[0])
            )[:3]
            category_text = ", ".join(
                f"{escape(GRAMMAR_CATEGORY_LABELS.get(category, category.replace('_', ' ')))} ({amount})"
                for category, amount in top_categories
            )
            parts.append(f"<p>Top categories: {category_text}.</p>")
            examples = [
                issue
                for result in results
                if result.get("status") == "ok"
                for issue in result.get("issues", [])
                if isinstance(issue, dict)
            ][:3]
            if examples:
                parts.append("<ul>")
                for issue in examples:
                    category = issue.get("category")
                    label = GRAMMAR_CATEGORY_LABELS.get(
                        category, category.replace("_", " ")
                    ) if isinstance(category, str) else "Grammar"
                    parts.append(
                        f"<li><strong>{escape(label)}</strong>: "
                        f"&quot;{escape(_as_text(issue.get('span_text')))}&quot; "
                        f"-> &quot;{escape(_as_text(issue.get('suggestion')))}&quot;. "
                        f"{escape(_as_text(issue.get('explanation')))}</li>"
                    )
                parts.append("</ul>")
        if "grammar_unavailable" in statuses:
            parts.append(
                "<p>Some answers could not be checked because the offline "
                "grammar model was unavailable.</p>"
            )
        if "too_short" in statuses:
            parts.append(
                "<p>Short answers were skipped because they had fewer than "
                "5 checked words.</p>"
            )
        return "".join(parts)

    if "grammar_unavailable" in statuses:
        message = next(
            (
                result.get("message")
                for result in results
                if result.get("status") == "grammar_unavailable"
                and isinstance(result.get("message"), str)
                and result.get("message")
            ),
            "Install spaCy and its en_core_web_sm model to enable offline grammar checks.",
        )
        if "too_short" in statuses:
            message = (
                f"{message} Short answers were also skipped because they had "
                "fewer than 5 checked words."
            )
        return f"<h4>Grammar</h4><p>{escape(message)}</p>"
    if statuses and all(status == "too_short" for status in statuses):
        return (
            "<h4>Grammar</h4><p>Grammar analysis was skipped because all "
            "answers had fewer than 5 checked words.</p>"
        )
    return (
        "<h4>Grammar</h4><p>No grammar results were available for these "
        "answers.</p>"
    )


def _answer_metrics_table(turn_indexes: list[int], turns: list[dict[str, Any]]) -> str:
    """Render delivery measurements for each answer in a question group."""
    columns = (
        ("Words", "fluency", "word_count"),
        ("Words per minute", "fluency", "words_per_minute"),
        ("Articulation rate", "audio_delivery", "articulation_rate_wpm"),
        ("Pause count", "audio_delivery", "pause_count"),
        ("Longest pause (s)", "audio_delivery", "longest_pause_seconds"),
        ("Pause ratio", "audio_delivery", "pause_ratio"),
        ("Fillers", "fluency", "filler_count"),
        ("Hedges", "fluency", "hedge_count"),
        ("Thinking time (s)", "fluency", "thinking_time_seconds"),
    )
    parts = ['<table><caption>Delivery by answer</caption><thead><tr><th scope="col">Turn</th>']
    parts.extend(f'<th scope="col">{label}</th>' for label, _, _ in columns)
    parts.append("</tr></thead><tbody>")
    for index in turn_indexes:
        analysis = turns[index - 1].get("analysis", {})
        parts.append(f"<tr><th scope=\"row\">{index}</th>")
        for _, source, metric in columns:
            value = analysis.get(source, {}).get(metric)
            parts.append(f"<td>{escape(_format_number(_number(value)))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _group_metrics_table(turn_indexes: list[int], turns: list[dict[str, Any]]) -> str:
    """Render pooled delivery metrics while retaining the per-answer table."""
    group_turns = [turns[index - 1] for index in turn_indexes]
    metrics = aggregate_group_delivery(group_turns)
    columns = (
        ("Total words", "total_words"),
        ("Words per minute", "words_per_minute"),
        ("Pause count", "pause_count"),
        ("Longest pause (s)", "longest_pause_seconds"),
        ("Pause ratio", "pause_ratio"),
        ("Level variation (dB)", "level_std_db"),
        ("Fillers per 100 words", "fillers_per_100_words"),
        ("Hedges per 100 words", "hedges_per_100_words"),
        ("Thinking time (s)", "thinking_time_seconds"),
    )
    cells = "".join(
        f"<td>{escape(_format_number(_number(metrics.get(key))))}</td>"
        for _, key in columns
    )
    headers = "".join(f"<th scope=\"col\">{label}</th>" for label, _ in columns)
    return (
        '<table><caption>Group delivery summary</caption><thead><tr>'
        f"{headers}</tr></thead><tbody><tr>{cells}</tr></tbody></table>"
    )


def _transcript_html(answer: object) -> str:
    """Render transcript text with low-confidence timed words highlighted."""
    if not isinstance(answer, dict):
        return ""
    words = answer.get("words")
    if not isinstance(words, list) or not words:
        return escape(_as_text(answer.get("text", "")))
    rendered: list[str] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = _as_text(word.get("text", ""))
        probability = _number(word.get("probability"))
        escaped_text = escape(text)
        if probability is not None and probability < LOW_CONFIDENCE_WORD_PROBABILITY:
            rendered.append(
                f'<mark title="Recognition probability: {probability:.2f}">{escaped_text}</mark>'
            )
        else:
            rendered.append(escaped_text)
    return "".join(rendered) if rendered else escape(_as_text(answer.get("text", "")))


def _render_feedback_list(title: str, values: object) -> str:
    """Render escaped model-provided feedback as a list."""
    if not isinstance(values, list):
        values = []
    items = "".join(f"<li>{escape(_as_text(value))}</li>" for value in values)
    return f"<h4>{title}</h4><ul>{items or '<li>None provided.</li>'}</ul>"


def _turn_labels(group: dict[str, Any], turns: list[dict[str, Any]]) -> dict[int, bool]:
    """Return the stored follow-up status for each turn in a group."""
    labels: dict[int, bool] = {}
    for index in group.get("turn_indexes", []):
        if not isinstance(index, int) or not 1 <= index <= len(turns):
            continue
        asked = turns[index - 1].get("analysis", {}).get("asked", {})
        labels[index] = asked.get("is_follow_up") is True
    return labels


def _infer_groups(
    analyzed: dict[str, Any], bank_by_id: Mapping[str, BankQuestion]
) -> list[dict[str, Any]]:
    """Build display-only consecutive question groups when content analysis is absent."""
    turns = analyzed.get("turns", [])
    labels = resolve_asked_questions(InterviewSession.from_dict(analyzed), bank_by_id)
    groups: list[dict[str, Any]] = []
    for index, label in enumerate(labels, start=1):
        question_id = label.get("main_question_id")
        if groups and groups[-1]["main_question_id"] == question_id:
            groups[-1]["turn_indexes"].append(index)
        else:
            groups.append(
                {
                    "main_question_id": question_id,
                    "turn_indexes": [index],
                    "label_source": label.get("source", "unknown"),
                    "rated": False,
                    "reason": (
                        "unknown_question"
                        if question_id is None
                        else "Content analysis was not run."
                    ),
                    "asr_reliability": "unknown",
                }
            )
    return groups


def _chart_input(
    analyzed: dict[str, Any], groups: list[dict[str, Any]]
) -> dict[str, Any]:
    """Provide the display groups to charts without mutating the input."""
    return {
        "turns": analyzed.get("turns", []),
        "analysis": {"content_by_question": groups},
    }


def _metadata_item(label: str, value: str) -> str:
    """Render one safely escaped metadata value."""
    return f"<div><strong>{label}:</strong> {escape(value)}</div>"


def _model_value(value: object) -> str:
    """Return a model name or a clear unavailable marker."""
    return _as_text(value) if isinstance(value, str) and value else "Not available"


def _as_text(value: object) -> str:
    """Convert supported JSON scalar values to text without invoking custom reprs."""
    return value if isinstance(value, str) else ""


def _has_answer(turn: dict[str, Any]) -> bool:
    """Return whether a turn has a non-empty transcript."""
    answer = turn.get("answer")
    return isinstance(answer, dict) and bool(
        isinstance(answer.get("text"), str) and answer["text"].strip()
    )


def _number(value: object) -> float | None:
    """Return an optional JSON number as float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _format_number(value: float | None) -> str:
    """Format a metric, using a dash for unavailable measurements."""
    if value is None:
        return "-"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _topic_label(question_id: object) -> str:
    """Return a reader-facing topic label for a question group."""
    if isinstance(question_id, str) and question_id:
        return TOPIC_LABELS.get(question_id, question_id)
    return "unknown"


def _reason_category(group: dict[str, Any]) -> str:
    """Return an unrated reason or error category."""
    for key in ("reason", "error"):
        value = group.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _score_variation(
    session_analysis: dict[str, Any], question_id: object
) -> str | None:
    """Format the available per-question score range across saved content runs."""
    runs = session_analysis.get("content_runs")
    if not isinstance(runs, list):
        return None
    scores: list[float] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        groups = run.get("content_by_question")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict) or group.get("main_question_id") != question_id:
                continue
            score = _number(group.get("content_score"))
            if score is not None:
                scores.append(score)
                break
    if not scores:
        return None
    return (
        f"range {_format_number(min(scores))}-{_format_number(max(scores))} "
        f"over {len(scores)} runs"
    )


def _rating(value: object) -> str:
    """Format a rubric score or a dash."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "-"


def _stt_model_label(metadata: Mapping[str, Any]) -> str:
    """Describe the live model and, when used, the larger report-pass model."""
    live = _model_value(metadata.get("stt_model"))
    report = metadata.get("report_stt_model")
    if isinstance(report, str) and report and report != metadata.get("stt_model"):
        return f"{live} (live), {_as_text(report)} (this report)"
    return live


def _content_model(groups: Any) -> str | None:
    """Return the model name recorded by the first rated content result, if any."""
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, Mapping) and isinstance(group.get("model"), str):
                return group["model"] or None
    return None


def _content_band(score: float | None) -> str:
    """Return the text band for a question content score."""
    if score is None:
        return "Not available"
    if score < CONTENT_LOW:
        return "Needs work"
    if score < CONTENT_HIGH:
        return "Developing"
    return "Strong"
