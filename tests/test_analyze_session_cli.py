"""CLI tests for loading and writing analyzed sessions."""

import json
from pathlib import Path

from scripts.analyze_session import main
from src.speaksure.session_log import InterviewSession
from src.speaksure.speech_to_text import TranscriptionResult


def test_cli_writes_analysis_separately_and_leaves_original_untouched(
    tmp_path: Path, monkeypatch, capsys
):
    """The CLI writes an analyzed exact-path copy and preserves source bytes."""
    source = InterviewSession.new("hr", "standard", 1)
    source.add_turn(
        "Tell me about yourself.",
        answer=TranscriptionResult("I am a developer.", (), 1.0, "en", 1.0, "typed"),
    )
    input_path = source.save(tmp_path / "sessions")
    original_bytes = input_path.read_bytes()
    output_path = tmp_path / "nested" / "custom.json"
    monkeypatch.chdir(tmp_path)

    result = main([str(input_path), "--out", str(output_path)])

    captured = capsys.readouterr()
    assert result == 0
    assert input_path.read_bytes() == original_bytes
    assert output_path.is_file()
    analyzed = InterviewSession.load(output_path)
    assert "fluency" in analyzed.turns[0].analysis
    assert "transcript_quality" in analyzed.turns[0].analysis
    assert "turn words speech_s wpm pauses longest_pause pause_ratio fillers hedges repeats mean_prob" in captured.out
    assert "Analysis saved:" in captured.out


def test_cli_missing_or_invalid_session_returns_error_without_traceback(
    tmp_path: Path, capsys
):
    """Missing and invalid source paths produce concise CLI errors."""
    missing = tmp_path / "missing.json"
    assert main([str(missing)]) == 1
    first = capsys.readouterr()
    assert "Could not analyze session:" in first.err
    assert "Traceback" not in first.err

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad json", encoding="utf-8")
    assert main([str(invalid)]) == 1
    second = capsys.readouterr()
    assert "Corrupt session JSON" in second.err
    assert "Traceback" not in second.err
