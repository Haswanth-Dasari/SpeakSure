"""Analyze a saved interview session using transcript and audio heuristics."""

import argparse
import copy
import math
import sys
from pathlib import Path
from statistics import median
from time import sleep
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.analysis.session_analysis import analyze_session  # noqa: E402
from speaksure.llm_client import (  # noqa: E402
    GroqAnalysisLLMConfig,
    LLMError,
    build_analysis_llm_client,
)
from speaksure.session_log import InterviewSession, SessionFormatError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Load, analyze, and print a session while saving to a separate output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--audio-root", type=Path, default=Path("."))
    parser.add_argument("--content", action="store_true")
    parser.add_argument("--role", default="software engineering fresher")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be from 1 to 10.")
    if args.repeat > 1 and not args.content:
        parser.error("--repeat requires --content.")
    if not math.isfinite(args.pause_seconds) or args.pause_seconds < 0:
        parser.error("--pause-seconds must be a finite non-negative number.")

    try:
        source = args.session
        session = InterviewSession.load(source)
        output_path = args.out or (
            Path("data/analysis") / f"{source.stem}.analyzed.json"
        )
        if output_path.resolve() == source.resolve():
            raise ValueError("Output path must not overwrite the source session.")
        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"Analyzed output already exists: {output_path}. "
                "Choose another --out path or pass --force to overwrite."
            )
        llm = None
        if args.content:
            analysis_config = GroqAnalysisLLMConfig.from_env()
            if not analysis_config.api_key:
                raise ValueError(
                    "Content analysis requires GROQ_API_KEY in the project-root .env file."
                )
            if not analysis_config.model.strip():
                raise ValueError("Content analysis model is not configured.")
            llm = build_analysis_llm_client(analysis_config)
        analyzed_runs = []
        for run_index in range(args.repeat):
            analyzed_runs.append(
                analyze_session(
                    session,
                    audio_root=args.audio_root,
                    llm=llm,
                    role=args.role,
                )
            )
            if run_index + 1 < args.repeat and args.pause_seconds:
                sleep(args.pause_seconds)
        analyzed = analyzed_runs[-1]
        if args.repeat > 1:
            _select_median_content_run(analyzed, analyzed_runs)
        if args.content and args.repeat == 1 and any(
            group.get("error") == "llm_error"
            for group in analyzed.analysis.get("content_by_question", [])
        ):
            raise LLMError(
                "Content analysis could not use the configured model. "
                "Check GROQ_API_KEY, LLM_ANALYSIS_MODEL, and network access."
            )
        saved_path = analyzed.save_to(output_path)
    except (OSError, SessionFormatError, ValueError, LLMError) as error:
        print(f"Could not analyze session: {error}", file=sys.stderr)
        return 1

    _print_report(analyzed)
    if args.content:
        _print_content_report(analyzed)
    print(f"Analysis saved: {saved_path}")
    return 0


def _select_median_content_run(
    analyzed: InterviewSession, analyzed_runs: Sequence[InterviewSession]
) -> None:
    """Store report-compatible runs and select each question's median-score group."""
    runs = [
        {
            "content_by_question": copy.deepcopy(
                run.analysis.get("content_by_question", [])
            )
        }
        for run in analyzed_runs
    ]
    analyzed.analysis["content_runs"] = runs
    source_groups = analyzed_runs[0].analysis.get("content_by_question", [])
    selected_groups: list[dict[str, Any]] = []
    for source_group in source_groups:
        key = _content_group_key(source_group)
        candidates = [
            (run_index, group)
            for run_index, run in enumerate(runs, start=1)
            for group in run["content_by_question"]
            if _content_group_key(group) == key
        ]
        rated = [
            (run_index, group)
            for run_index, group in candidates
            if group.get("rated") is True
            and isinstance(group.get("content_score"), (int, float))
        ]
        if rated:
            target_score = median(
                float(group["content_score"]) for _, group in rated
            )
            run_index, selected = min(
                rated,
                key=lambda item: abs(float(item[1]["content_score"]) - target_score),
            )
        elif candidates:
            run_index, selected = candidates[0]
        else:
            continue
        selected_group = copy.deepcopy(selected)
        selected_group["selected_run_index"] = run_index
        selected_groups.append(selected_group)

    analyzed.analysis["content_by_question"] = selected_groups
    _refresh_content_summary(analyzed, selected_groups)


def _content_group_key(group: dict[str, Any]) -> tuple[object, tuple[object, ...]]:
    """Identify corresponding groups by question label and turn indexes."""
    turn_indexes = group.get("turn_indexes", [])
    return (
        group.get("main_question_id"),
        tuple(turn_indexes) if isinstance(turn_indexes, list) else (),
    )


def _refresh_content_summary(
    analyzed: InterviewSession, groups: list[dict[str, Any]]
) -> None:
    """Recompute the selected-run aggregate fields used by existing reports."""
    rated = [group for group in groups if group.get("rated") is True]

    def average(field: str) -> float | None:
        values = [
            float(group[field])
            for group in rated
            if isinstance(group.get(field), (int, float))
        ]
        return sum(values) / len(values) if values else None

    sources: dict[str, int] = {"recorded": 0, "inferred": 0, "unknown": 0}
    feedback_by_question: dict[str, dict[str, list[str]]] = {}
    for group in groups:
        source = str(group.get("label_source", "unknown"))
        sources[source] = sources.get(source, 0) + 1
    for group in rated:
        question_id = str(group.get("main_question_id"))
        feedback = feedback_by_question.setdefault(
            question_id, {"strengths": [], "improvements": []}
        )
        feedback["strengths"].extend(group.get("strengths", []))
        feedback["improvements"].extend(group.get("improvements", []))
    analyzed.analysis["content_summary"] = {
        "rated_questions": len(rated),
        "unrated_questions": len(groups) - len(rated),
        "mean_content_score": average("content_score"),
        "mean_relevance": average("relevance"),
        "mean_specificity": average("specificity"),
        "mean_clarity": average("clarity"),
        "mean_coverage_ratio": average("coverage_ratio"),
        "content_score_by_question": [
            group.get("content_score") if group.get("rated") is True else None
            for group in groups
        ],
        "feedback_by_question_id": feedback_by_question,
        "label_sources": sources,
    }


def _print_report(session: InterviewSession) -> None:
    """Print per-turn audio metrics, pause comparison, and session summaries."""
    headers = (
        "turn",
        "words",
        "speech_s",
        "wpm",
        "audio_pauses",
        "audio_longest",
        "audio_pause_ratio",
        "artic_wpm",
        "level_std_db",
        "fillers",
        "hedges",
        "mean_prob",
    )
    print(" ".join(headers))
    for turn in session.turns:
        fluency = turn.analysis.get("fluency", {})
        quality = turn.analysis.get("transcript_quality", {})
        delivery = turn.analysis.get("audio_delivery", {})
        values = (
            turn.index,
            fluency.get("word_count"),
            fluency.get("speech_seconds"),
            fluency.get("words_per_minute"),
            delivery.get("pause_count") if delivery.get("has_audio") else None,
            delivery.get("longest_pause_seconds") if delivery.get("has_audio") else None,
            delivery.get("pause_ratio") if delivery.get("has_audio") else None,
            delivery.get("articulation_rate_wpm") if delivery.get("has_audio") else None,
            delivery.get("level_std_db") if delivery.get("has_audio") else None,
            fluency.get("filler_count"),
            fluency.get("hedge_count"),
            quality.get("mean_word_probability"),
        )
        print(" ".join(_display(value) for value in values))
        flags = delivery.get("quality_flags", [])
        if flags:
            print(f"  quality_flags: {', '.join(flags)}")

    print("ASR-timestamp pauses vs audio pauses")
    print("turn asr_pauses audio_pauses")
    delivery_summary = session.analysis.get("delivery_summary", {})
    for comparison in delivery_summary.get("asr_vs_audio_pauses", []):
        print(
            " ".join(
                _display(comparison.get(key))
                for key in ("turn", "asr_pause_count", "audio_pause_count")
            )
        )

    summary = session.analysis.get("fluency_summary", {})
    summary_fields = (
        "answered_turns",
        "unanswered_turns",
        "mean_words_per_minute",
        "mean_pause_ratio",
        "total_fillers",
        "total_hedges",
        "total_immediate_repetitions",
        "mean_thinking_time_seconds",
        "mean_transcript_probability",
    )
    for field in summary_fields:
        print(f"{field}: {_display(summary.get(field))}")
    print("Delivery summary")
    for field in (
        "audio_turns",
        "mean_pause_ratio_audio",
        "total_pause_seconds_audio",
        "total_pause_count_audio",
        "mean_articulation_rate_wpm",
        "mean_level_std_db",
        "pause_ratio_audio_by_turn",
        "level_std_db_by_turn",
    ):
        print(f"{field}: {_display(delivery_summary.get(field))}")

    print("Grammar by turn")
    for turn in session.turns:
        grammar = turn.analysis.get("grammar", {})
        print(
            f"turn {turn.index}: status={_display(grammar.get('status'))} "
            f"words={_display(grammar.get('word_count_checked'))} "
            f"errors={_display(grammar.get('error_count'))} "
            f"errors_per_100_words={_display(grammar.get('errors_per_100_words'))}"
        )
        for issue in grammar.get("issues", []):
            print(
                f"  {issue.get('category')}: {issue.get('span_text')} -> "
                f"{issue.get('suggestion')} (word {issue.get('word_index')})"
            )


def _display(value: object) -> str:
    """Render missing values as a dash and numbers compactly."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_content_report(session: InterviewSession) -> None:
    """Print question-group scores, feedback, evidence, and label-source counts."""
    print("Content analysis")
    print(
        "question_id turns relevance specificity clarity coverage "
        "content_score asr_reliability"
    )
    groups = session.analysis.get("content_by_question", [])
    for content in groups:
        turn_indexes = content.get("turn_indexes", [])
        turns_text = "+".join(str(index) for index in turn_indexes) or "-"
        if content.get("rated") is not True:
            print(
                f"{_display(content.get('main_question_id'))} {turns_text} "
                f"- - - - - {_display(content.get('asr_reliability'))}"
            )
            continue
        coverage = content.get("coverage", [])
        verified_count = sum(
            bool(entry.get("evidence_verified")) for entry in coverage
        )
        print(
            " ".join(
                _display(value)
                for value in (
                    content.get("main_question_id"),
                    turns_text,
                    content.get("relevance"),
                    content.get("specificity"),
                    content.get("clarity"),
                    f"{verified_count}/{len(coverage)}",
                    content.get("content_score"),
                    content.get("asr_reliability"),
                )
            )
        )
        print(f"  strengths: {'; '.join(content['strengths']) or '-'}")
        print(f"  improvements: {'; '.join(content['improvements']) or '-'}")
        failed_evidence = [
            entry
            for entry in coverage
            if entry.get("covered") and not entry.get("evidence_verified")
        ]
        for entry in failed_evidence:
            print(
                "  unverified evidence: "
                f"{entry['hint']} -> {entry['evidence'] or '(empty quote)'}"
            )
    label_sources = session.analysis.get("content_summary", {}).get(
        "label_sources", {}
    )
    ordered_sources = sorted(
        ("recorded", "inferred", "unknown"),
        key=lambda source: (
            label_sources.get(source, 0) == 0,
            -label_sources.get(source, 0),
        ),
    )
    print(
        "labels: "
        + ", ".join(
            f"{label_sources.get(source, 0)} {source}"
            for source in ordered_sources
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
