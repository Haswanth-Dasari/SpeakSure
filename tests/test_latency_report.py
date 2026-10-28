import json
from pathlib import Path

from scripts.latency_report import (
    main,
    read_turn_timings,
    render_latency_report,
)


def _session_file(path: Path, timings: list[dict[str, float | None]]) -> Path:
    """Write fake session JSON containing the provided turn measurements."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "turns": [
                    {"analysis": {"timing": timing}} for timing in timings
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_latency_report_summarizes_stages_and_total_answer_to_reply_time() -> None:
    """Summarize measured pipeline stages and combined response latency."""
    turns = [
        {
            "end_wait_s": 1.0,
            "upload_s": 1.0,
            "stt_s": 2.0,
            "llm_s": 2.0,
            "tts_s": 3.0,
        },
        {
            "end_wait_s": 3.0,
            "upload_s": 2.0,
            "stt_s": 4.0,
            "llm_s": 3.0,
            "tts_s": 5.0,
        },
    ]

    report = render_latency_report(turns)

    assert "| end_wait_s | 2.000 | 3.000 | 3.000 | 2 |" in report
    assert "| total_s | 13.000 | 17.000 | 17.000 | 2 |" in report


def test_latency_reader_supports_legacy_timing_keys_and_omits_unknown_totals(
    tmp_path: Path,
) -> None:
    """Read legacy timing names without presenting incomplete totals as complete."""
    session = _session_file(
        tmp_path / "session.json",
        [
            {
                "stt_seconds": 1.25,
                "llm_seconds": 0.5,
                "tts_synthesis_seconds": 0.75,
            }
        ],
    )

    turns = read_turn_timings([session])
    report = render_latency_report(turns)

    assert turns[0]["stt_s"] == 1.25
    assert "| stt_s | 1.250 | 1.250 | 1.250 | 1 |" in report
    assert "| total_s | n/a | n/a | n/a | 0 |" in report


def test_latency_cli_writes_markdown_beside_input_directory(tmp_path: Path) -> None:
    """Write the generated Markdown report beside the supplied sessions folder."""
    timings = [
        {
            "end_wait_s": 0.5,
            "stt_s": 1.0,
            "llm_s": 0.25,
            "tts_s": 0.75,
        }
    ]
    _session_file(tmp_path / "one.json", timings)

    assert main([str(tmp_path)]) == 0

    report_path = tmp_path / "latency_report.md"
    assert report_path.is_file()
    assert "| total_s | 2.500 | 2.500 | 2.500 | 1 |" in report_path.read_text(
        encoding="utf-8"
    )
