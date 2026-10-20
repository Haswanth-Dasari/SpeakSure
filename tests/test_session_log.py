import json
from datetime import datetime, timezone

import pytest

from src.speaksure.session_log import InterviewSession, SessionFormatError
from src.speaksure.speech_to_text import TranscriptWord, TranscriptionResult


def make_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="Hello there.",
        words=(
            TranscriptWord("Hello", 0.12345, 0.45678, 0.98765),
            TranscriptWord("there.", 0.5, 0.9, 0.87654),
        ),
        duration_seconds=1.23456,
        language="en",
        language_probability=0.99876,
        device="cpu",
    )


def make_session() -> InterviewSession:
    session = InterviewSession.new("HR interview", "Intermediate", 3)
    session.add_turn(
        "Tell me about yourself.",
        answer=make_result(),
        thinking_time_seconds=0.12345,
        answer_duration_seconds=1.23456,
    )
    return session


def test_session_full_round_trip_includes_transcription_words():
    original = make_session()

    restored = InterviewSession.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
    assert restored.turns[0].answer.words[0] == TranscriptWord(
        "Hello", 0.123, 0.457, 0.988
    )


def test_add_turn_assigns_sequential_one_based_indices():
    session = InterviewSession.new("HR interview", "Intermediate", 3)

    turns = [session.add_turn(f"Question {index}") for index in range(1, 4)]

    assert [turn.index for turn in turns] == [1, 2, 3]


def test_unknown_schema_version_raises_session_format_error():
    data = make_session().to_dict()
    data["schema_version"] = 99

    with pytest.raises(SessionFormatError, match="schema_version"):
        InterviewSession.from_dict(data)


def test_corrupt_json_raises_session_format_error(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text('{"schema_version": ', encoding="utf-8")

    with pytest.raises(SessionFormatError, match="Corrupt session JSON"):
        InterviewSession.load(path)


def test_missing_key_raises_session_format_error():
    data = make_session().to_dict()
    del data["mode"]

    with pytest.raises(SessionFormatError, match="mode"):
        InterviewSession.from_dict(data)


def test_to_dict_rounds_float_values_to_three_decimals():
    data = make_session().to_dict()

    assert data["turns"][0]["thinking_time_seconds"] == 0.123
    assert data["turns"][0]["answer"]["words"][0]["end"] == 0.457
    assert data["turns"][0]["answer"]["language_probability"] == 0.999
    assert data["turns"][0]["answer_duration_seconds"] == 1.235


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path):
    path = make_session().save(tmp_path)

    assert path.is_file()
    assert path.parent == tmp_path
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


def test_turn_without_answer_survives_round_trip():
    session = InterviewSession.new("Technical interview", "Advanced", 1)
    session.add_turn("Describe your project.", answer=None)

    restored = InterviewSession.from_dict(session.to_dict())

    assert restored.turns[0].answer is None
    assert restored.to_dict() == session.to_dict()


def test_new_session_uses_uuid_hex_and_utc_timestamp():
    session = InterviewSession.new("Everyday conversation", "Beginner", 2)

    assert len(session.session_id) == 32
    assert session.session_id == session.session_id.lower()
    assert datetime.fromisoformat(session.created_at).utcoffset() == timezone.utc.utcoffset(
        datetime.fromisoformat(session.created_at)
    )
