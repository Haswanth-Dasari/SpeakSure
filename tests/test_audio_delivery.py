"""Synthetic WAV tests for audio-delivery analysis."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_session import main
from src.speaksure.analysis.audio_delivery import analyze_audio_delivery
from src.speaksure.analysis.session_analysis import analyze_session
from src.speaksure.session_log import InterviewSession
from src.speaksure.speech_to_text import TranscriptionResult


SAMPLE_RATE = 16000


def write_wav(
    path: Path,
    parts: list[tuple[float, float]],
    *,
    channels: int = 1,
) -> Path:
    """Write specified tone/silence segments to a synthetic 16-bit WAV."""
    segments = []
    offset = 0
    for duration, amplitude in parts:
        count = round(duration * SAMPLE_RATE)
        if amplitude:
            times = np.arange(count, dtype=np.float64) / SAMPLE_RATE
            samples = amplitude * np.sin(2.0 * np.pi * 440.0 * times)
            samples = np.rint(samples * 32767).astype("<i2")
        else:
            samples = np.zeros(count, dtype="<i2")
        segments.append(samples)
        offset += count
    mono = np.concatenate(segments) if segments else np.array([], dtype="<i2")
    if channels == 2:
        pcm = np.column_stack((mono, mono)).astype("<i2").tobytes()
    else:
        pcm = mono.tobytes()
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)
    return path


def test_audio_pauses_and_speech_span_from_silence_runs(tmp_path: Path):
    """Only interior silence is measured, with long pauses identified."""
    path = write_wav(
        tmp_path / "pauses.wav",
        [(1.0, 0.2), (0.8, 0.0), (1.0, 0.2), (1.8, 0.0), (1.0, 0.2)],
    )

    metrics = analyze_audio_delivery(path, 20)

    assert metrics["has_audio"] is True
    assert metrics["pause_count"] == 2
    assert metrics["long_pause_count"] == 1
    assert metrics["total_pause_seconds"] == pytest.approx(2.6, abs=0.1)
    assert metrics["longest_pause_seconds"] == pytest.approx(1.8, abs=0.1)
    assert metrics["speech_span_seconds"] == pytest.approx(5.6, abs=0.1)
    assert metrics["pause_ratio"] == pytest.approx(0.46, abs=0.03)


def test_leading_trailing_silence_excluded_and_short_gap_bridged(tmp_path: Path):
    """Edge silence is excluded and an enclosed sub-0.2-second gap is bridged."""
    path = write_wav(
        tmp_path / "edges.wav",
        [(0.6, 0.0), (0.6, 0.2), (0.15, 0.0), (0.6, 0.2), (0.6, 0.0)],
    )

    metrics = analyze_audio_delivery(path, 10)

    assert metrics["pause_count"] == 0
    assert metrics["speech_span_seconds"] == pytest.approx(1.35, abs=0.1)
    assert metrics["voiced_seconds"] == pytest.approx(metrics["speech_span_seconds"])
    assert metrics["articulation_rate_wpm"] is None


def test_level_statistics_and_quality_flags(tmp_path: Path):
    """Stable tones measure level variation and quiet/clipped conditions."""
    constant = write_wav(tmp_path / "constant.wav", [(2.0, 0.2)])
    quiet = write_wav(tmp_path / "quiet.wav", [(1.5, 0.002)])
    clipped = write_wav(tmp_path / "clipped.wav", [(1.5, 1.0)])

    constant_metrics = analyze_audio_delivery(constant, 10)
    quiet_metrics = analyze_audio_delivery(quiet, 10)
    clipped_metrics = analyze_audio_delivery(clipped, 10)

    assert constant_metrics["level_std_db"] == pytest.approx(0.0, abs=0.5)
    assert constant_metrics["mean_level_db"] == pytest.approx(-17.0, abs=1.0)
    assert "too_quiet" in quiet_metrics["quality_flags"]
    assert "clipped" in clipped_metrics["quality_flags"]
    assert clipped_metrics["clipping_ratio"] > 0.001
    assert "very_short" in analyze_audio_delivery(
        write_wav(tmp_path / "short.wav", [(0.5, 0.2)]), 5
    )["quality_flags"]


@pytest.mark.parametrize("word_count", [0, 4])
def test_articulation_rate_requires_five_words(tmp_path: Path, word_count: int):
    """Audio articulation rate is unavailable for fewer than five words."""
    path = write_wav(tmp_path / f"rate-{word_count}.wav", [(3.0, 0.2)])

    assert analyze_audio_delivery(path, word_count)["articulation_rate_wpm"] is None


def test_unavailable_and_invalid_wav_inputs_return_errors(tmp_path: Path):
    """Missing, empty, unreadable, and stereo audio never escapes as an exception."""
    empty = tmp_path / "empty.wav"
    empty.touch()
    stereo = write_wav(tmp_path / "stereo.wav", [(1.0, 0.2)], channels=2)

    for path in (None, tmp_path / "missing.wav", empty, stereo):
        result = analyze_audio_delivery(path, 5)
        assert result["has_audio"] is False
        assert isinstance(result["audio_error"], str)
        assert result["audio_error"]


def test_session_analysis_adds_audio_metrics_without_mutating_input(tmp_path: Path):
    """Per-turn audio results and delivery summaries preserve prior analysis."""
    session = InterviewSession.new("hr", "standard", 2, {"existing": True})
    audio = write_wav(
        tmp_path / "answer.wav",
        [(1.0, 0.2), (0.6, 0.0), (1.0, 0.2)],
    )
    answer = TranscriptionResult("one two three four five", (), 2.0, "en", 1.0, "test")
    session.add_turn(
        "Question",
        answer=answer,
        audio_path=audio.name,
        analysis={"timing": {"stt_seconds": 1.0}},
    )
    session.add_turn("No audio", answer=answer, audio_path=None)
    before = json.loads(json.dumps(session.to_dict()))

    result = analyze_session(session, audio_root=tmp_path)

    assert session.to_dict() == before
    assert result.turns[0].analysis["timing"] == {"stt_seconds": 1.0}
    assert result.turns[0].analysis["audio_delivery"]["has_audio"] is True
    assert result.turns[1].analysis["audio_delivery"]["has_audio"] is False
    assert result.analysis["delivery_summary"]["audio_turns"] == 1
    assert result.analysis["delivery_summary"]["pause_ratio_audio_by_turn"][1] is None
    assert result.analysis["delivery_summary"]["level_std_db_by_turn"][1] is None
    assert result.analysis["delivery_summary"]["asr_vs_audio_pauses"][1] == {
        "turn": 2,
        "asr_pause_count": 0,
        "audio_pause_count": None,
    }
    assert result.analysis["fluency_summary"]["analysis_versions"] == {
        "fluency": 1,
        "transcript_quality": 1,
        "audio_delivery": 1,
    }


def test_cli_prints_audio_and_pause_comparison_tables(tmp_path: Path, monkeypatch, capsys):
    """CLI supports audio roots and writes the analyzed output as before."""
    audio = write_wav(tmp_path / "answer.wav", [(2.0, 0.2)])
    session = InterviewSession.new("hr", "standard", 1)
    session.add_turn(
        "Question",
        answer=TranscriptionResult("one two three four five", (), 2.0, "en", 1.0, "test"),
        audio_path=str(audio),
    )
    source = session.save(tmp_path / "sessions")
    output = tmp_path / "analysis" / "result.json"
    monkeypatch.chdir(tmp_path)

    assert main([str(source), "--out", str(output), "--audio-root", "."]) == 0
    out = capsys.readouterr().out
    assert "audio_pauses audio_longest audio_pause_ratio artic_wpm level_std_db" in out
    assert "ASR-timestamp pauses vs audio pauses" in out
    assert "audio_turns: 1" in out
    assert output.is_file()
