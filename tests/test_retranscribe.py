"""Tests for the accurate second transcription pass used by the report."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.speaksure.retranscribe import choose_transcript, retranscribe_session
from src.speaksure.session_log import InterviewSession
from src.speaksure.speech_to_text import TranscriptionResult


def result(text: str) -> TranscriptionResult:
    return TranscriptionResult(text, (), 4.0, "en", 1.0, "cpu")


class FakeProvider:
    def __init__(self, replies: dict[str, object], device: str = "cuda") -> None:
        self.replies = replies
        self.actual_device = device
        self.warmed = False
        self.hotwords: list[object] = []

    def warmup(self) -> None:
        self.warmed = True

    def transcribe_detailed(self, audio: bytes, **kwargs):
        self.hotwords.append(kwargs.get("hotwords"))
        reply = self.replies[kwargs["filename"]]
        if isinstance(reply, Exception):
            raise reply
        return reply


def build_session(tmp_path: Path) -> InterviewSession:
    session = InterviewSession.new("hr", "standard", 4)
    for name, text in (
        ("t1.wav", "I have full strike development skills and many projects at college"),
        ("t2.wav", "I study computer science at the institute and I like building tools"),
        ("t3.wav", "I enjoy solving problems with a team and learning new things daily"),
    ):
        (tmp_path / name).write_bytes(b"RIFF")
        session.add_turn("Q", answer=result(text), audio_path=str(tmp_path / name))
    session.add_turn("Q4", answer=None, audio_path=None)
    return session


def test_better_transcript_replaces_live_and_original_is_untouched(tmp_path: Path) -> None:
    session = build_session(tmp_path)
    provider = FakeProvider(
        {
            "t1.wav": result("I have full stack development skills and many projects at college"),
            "t2.wav": result("I study computer science at the institute and I like building tools"),
            "t3.wav": result("I enjoy solving problems with a team and learning new things daily"),
        }
    )
    new = retranscribe_session(
        session, provider, report_model="large-v3", live_model="small", hotwords="Haswanth"
    )
    assert provider.warmed
    assert provider.hotwords == ["Haswanth"] * 3
    assert "full stack" in new.turns[0].answer.text
    assert "full strike" in session.turns[0].answer.text  # input not mutated
    record = new.turns[0].analysis["transcript_passes"]
    assert record["used"] == "report"
    assert "full strike" in record["live_text"]
    assert new.metadata["report_stt_model"] == "large-v3"
    assert new.turns[3].answer is None


def test_bad_report_output_and_errors_keep_live_text(tmp_path: Path) -> None:
    session = build_session(tmp_path)
    provider = FakeProvider(
        {
            "t1.wav": result("ok"),  # collapsed
            "t2.wav": RuntimeError("boom"),
            "t3.wav": result("word " * 60),  # repetition-like growth
        }
    )
    new = retranscribe_session(session, provider, report_model="large-v3", live_model="small")
    for index in range(3):
        assert new.turns[index].answer.text == session.turns[index].answer.text
        assert new.turns[index].analysis["transcript_passes"]["used"] == "live"
    assert "report_stt_model" not in new.metadata


def test_missing_audio_is_left_alone(tmp_path: Path) -> None:
    session = InterviewSession.new("hr", "standard", 1)
    session.add_turn("Q", answer=result("hello there friend of mine"), audio_path=str(tmp_path / "gone.wav"))
    new = retranscribe_session(session, FakeProvider({}), report_model="m", live_model="s")
    assert new.turns[0].analysis["transcript_passes"]["reason"] == "audio file not found"


def test_cpu_report_model_is_refused() -> None:
    session = InterviewSession.new("hr", "standard", 1)
    with pytest.raises(RuntimeError):
        retranscribe_session(session, FakeProvider({}, device="cpu"), report_model="large-v3", live_model="small")


def test_choose_transcript_rules() -> None:
    assert choose_transcript("one two", "three four five")[0] is True  # short live text: accept
    assert choose_transcript("a b c d e f g h", "")[0] is False
    assert choose_transcript("a b c d e f g h", "a b")[0] is False
    assert choose_transcript("a b c d e f g h", "a b c d e f g h i")[0] is True


def test_live_text_is_kept_when_it_has_more_of_the_candidates_names() -> None:
    live = "My name is Haswanth Dasari and I study at IIITDM Kurnool"
    report = "My name is Ashwan Dasari and I study at IIITDM Kurnool"
    terms = ["Haswanth", "IIITDM Kurnool"]
    keep_live, reason = choose_transcript(live, report, terms)
    assert keep_live is False and "names" in reason
    assert choose_transcript(report, live, terms)[0] is True  # the better one wins either way
    assert choose_transcript(live, report, [])[0] is True  # no hints: no preference
    assert choose_transcript(live, live.replace("Dasari", "Dasare"), terms)[0] is True  # equal hits


def test_session_pass_applies_the_name_rule_per_turn(tmp_path: Path) -> None:
    session = InterviewSession.new("hr", "standard", 2)
    for name, text in (
        ("a.wav", "My name is Haswanth Dasari and I build speech tools every day"),
        ("b.wav", "I use Flask for the backend and I like writing clean code daily"),
    ):
        (tmp_path / name).write_bytes(b"RIFF")
        session.add_turn("Q", answer=result(text), audio_path=str(tmp_path / name))
    provider = FakeProvider(
        {
            "a.wav": result("My name is Ashwan Dasari and I build speech tools every day"),
            "b.wav": result("I use Flask for the backend and I like writing clean code every day"),
        }
    )
    new = retranscribe_session(
        session, provider, report_model="large-v3", live_model="medium", hotwords="Haswanth, Flask"
    )
    assert "Haswanth" in new.turns[0].answer.text  # kept live: it had the name
    assert new.turns[0].analysis["transcript_passes"]["used"] == "live"
    assert "every day" in new.turns[1].answer.text  # report text used when hints tie
    assert new.turns[1].analysis["transcript_passes"]["used"] == "report"
