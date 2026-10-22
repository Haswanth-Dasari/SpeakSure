"""Analyze a saved interview session using transcript and audio heuristics."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.analysis.session_analysis import analyze_session  # noqa: E402
from speaksure.session_log import InterviewSession, SessionFormatError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Load, analyze, and print a session while saving to a separate output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--audio-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    try:
        source = args.session
        session = InterviewSession.load(source)
        analyzed = analyze_session(session, audio_root=args.audio_root)
        output_path = args.out or (
            Path("data/analysis") / f"{source.stem}.analyzed.json"
        )
        if output_path.resolve() == source.resolve():
            raise ValueError("Output path must not overwrite the source session.")
        saved_path = analyzed.save_to(output_path)
    except (OSError, SessionFormatError, ValueError) as error:
        print(f"Could not analyze session: {error}", file=sys.stderr)
        return 1

    _print_report(analyzed)
    print(f"Analysis saved: {saved_path}")
    return 0


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


def _display(value: object) -> str:
    """Render missing values as a dash and numbers compactly."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
