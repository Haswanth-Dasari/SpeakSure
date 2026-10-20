"""Serializable interview session data and atomic JSON persistence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

if TYPE_CHECKING:
    from .speech_to_text import TranscriptionResult


class SessionFormatError(ValueError):
    """Raised when session data is missing, malformed, or unsupported."""


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _validate_utc_timestamp(value: object, field_name: str) -> str:
    """Validate and return a timestamp string with a zero UTC offset."""
    if not isinstance(value, str):
        raise SessionFormatError(f"{field_name} must be a UTC ISO-8601 string.")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SessionFormatError(f"{field_name} must be a valid ISO-8601 timestamp.") from None
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise SessionFormatError(f"{field_name} must use the UTC timezone.")
    return value


def _required(data: dict[str, Any], key: str, expected_type: type) -> Any:
    """Return a required value after checking its type."""
    if key not in data:
        raise SessionFormatError(f"Missing required key: {key}.")
    value = data[key]
    if expected_type is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected_type is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, expected_type)
    if not valid:
        raise SessionFormatError(f"{key} has the wrong type; expected {expected_type.__name__}.")
    return value


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    """Read an optional finite number from serialized data."""
    if key not in data:
        raise SessionFormatError(f"Missing required key: {key}.")
    value = data[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionFormatError(f"{key} has the wrong type; expected a number or null.")
    number = float(value)
    if not math.isfinite(number):
        raise SessionFormatError(f"{key} must be a finite number.")
    return number


def _round_floats(value: Any) -> Any:
    """Round every float in a JSON-compatible value to three decimals."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, tuple):
        return [_round_floats(item) for item in value]
    return value


def _transcription_to_dict(answer: TranscriptionResult) -> dict[str, Any]:
    """Convert a transcription result and its words into plain data."""
    return {
        "text": answer.text,
        "words": [
            {
                "text": word.text,
                "start": word.start,
                "end": word.end,
                "probability": word.probability,
            }
            for word in answer.words
        ],
        "duration_seconds": answer.duration_seconds,
        "language": answer.language,
        "language_probability": answer.language_probability,
        "device": answer.device,
    }


def _transcription_from_dict(data: object) -> TranscriptionResult:
    """Validate and reconstruct a transcription result and its words."""
    from .speech_to_text import TranscriptWord, TranscriptionResult

    if not isinstance(data, dict):
        raise SessionFormatError("answer must be an object or null.")
    required_keys = (
        "text",
        "words",
        "duration_seconds",
        "language",
        "language_probability",
        "device",
    )
    for key in required_keys:
        if key not in data:
            raise SessionFormatError(f"Missing required key: answer.{key}.")
    words_data = data["words"]
    if not isinstance(words_data, list):
        raise SessionFormatError("answer.words has the wrong type; expected list.")
    words = []
    for index, word_data in enumerate(words_data):
        prefix = f"answer.words[{index}]"
        if not isinstance(word_data, dict):
            raise SessionFormatError(f"{prefix} has the wrong type; expected object.")
        for key in ("text", "start", "end", "probability"):
            if key not in word_data:
                raise SessionFormatError(f"Missing required key: {prefix}.{key}.")
        text = word_data["text"]
        if not isinstance(text, str):
            raise SessionFormatError(f"{prefix}.text has the wrong type; expected str.")
        numeric_values = {}
        for key in ("start", "end", "probability"):
            value = word_data[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SessionFormatError(
                    f"{prefix}.{key} has the wrong type; expected number."
                )
            numeric_values[key] = float(value)
            if not math.isfinite(numeric_values[key]):
                raise SessionFormatError(f"{prefix}.{key} must be finite.")
        words.append(
            TranscriptWord(
                text=text,
                start=numeric_values["start"],
                end=numeric_values["end"],
                probability=numeric_values["probability"],
            )
        )
    for key in ("text", "language", "device"):
        if not isinstance(data[key], str):
            raise SessionFormatError(f"answer.{key} has the wrong type; expected str.")
    numeric_values = {}
    for key in ("duration_seconds", "language_probability"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SessionFormatError(
                f"answer.{key} has the wrong type; expected number."
            )
        numeric_values[key] = float(value)
        if not math.isfinite(numeric_values[key]):
            raise SessionFormatError(f"answer.{key} must be finite.")
    return TranscriptionResult(
        text=data["text"],
        words=tuple(words),
        duration_seconds=numeric_values["duration_seconds"],
        language=data["language"],
        language_probability=numeric_values["language_probability"],
        device=data["device"],
    )


@dataclass
class Turn:
    """One interview question and its answer, timing, and derived analysis."""

    index: int
    question: str
    asked_at: str
    answer: TranscriptionResult | None = None
    audio_path: str | None = None
    thinking_time_seconds: float | None = None
    answer_duration_seconds: float | None = None
    analysis: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterviewSession:
    """An interview session containing ordered turns and analysis metadata."""

    session_id: str
    created_at: str
    mode: str
    difficulty: str
    planned_questions: int
    metadata: dict[str, Any] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[int] = 1

    @classmethod
    def new(
        cls,
        mode: str,
        difficulty: str,
        planned_questions: int,
        metadata: dict[str, Any] | None = None,
    ) -> InterviewSession:
        """Create a session with a UUID and the current UTC timestamp."""
        return cls(
            session_id=uuid4().hex,
            created_at=_utc_now(),
            mode=mode,
            difficulty=difficulty,
            planned_questions=planned_questions,
            metadata={} if metadata is None else metadata,
        )

    def add_turn(
        self,
        question: str,
        *,
        answer: TranscriptionResult | None = None,
        audio_path: str | None = None,
        thinking_time_seconds: float | None = None,
        answer_duration_seconds: float | None = None,
        analysis: dict[str, Any] | None = None,
        asked_at: str | None = None,
    ) -> Turn:
        """Append and return the next numbered turn."""
        turn = Turn(
            index=len(self.turns) + 1,
            question=question,
            asked_at=_utc_now() if asked_at is None else asked_at,
            answer=answer,
            audio_path=audio_path,
            thinking_time_seconds=thinking_time_seconds,
            answer_duration_seconds=answer_duration_seconds,
            analysis={} if analysis is None else analysis,
        )
        self.turns.append(turn)
        return turn

    def to_dict(self) -> dict[str, Any]:
        """Return the full session as JSON-compatible data."""
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "mode": self.mode,
            "difficulty": self.difficulty,
            "planned_questions": self.planned_questions,
            "metadata": self.metadata,
            "turns": [
                {
                    "index": turn.index,
                    "question": turn.question,
                    "asked_at": turn.asked_at,
                    "answer": (
                        None if turn.answer is None else _transcription_to_dict(turn.answer)
                    ),
                    "audio_path": turn.audio_path,
                    "thinking_time_seconds": turn.thinking_time_seconds,
                    "answer_duration_seconds": turn.answer_duration_seconds,
                    "analysis": turn.analysis,
                }
                for turn in self.turns
            ],
            "analysis": self.analysis,
        }
        return _round_floats(data)

    @classmethod
    def from_dict(cls, data: object) -> InterviewSession:
        """Validate serialized session data and reconstruct all nested records."""
        if not isinstance(data, dict):
            raise SessionFormatError("Session data must be an object.")
        if "schema_version" not in data:
            raise SessionFormatError("Missing required key: schema_version.")
        version = data["schema_version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise SessionFormatError("schema_version has the wrong type; expected int.")
        if version != cls.SCHEMA_VERSION:
            raise SessionFormatError(f"Unknown schema_version: {version}.")
        for key in (
            "session_id",
            "created_at",
            "mode",
            "difficulty",
            "planned_questions",
            "metadata",
            "turns",
            "analysis",
        ):
            if key not in data:
                raise SessionFormatError(f"Missing required key: {key}.")
        for key in ("session_id", "mode", "difficulty"):
            if not isinstance(data[key], str):
                raise SessionFormatError(f"{key} has the wrong type; expected str.")
        created_at = _validate_utc_timestamp(data["created_at"], "created_at")
        planned_questions = _required(data, "planned_questions", int)
        if planned_questions < 0:
            raise SessionFormatError("planned_questions must not be negative.")
        metadata = _required(data, "metadata", dict)
        analysis = _required(data, "analysis", dict)
        turns_data = _required(data, "turns", list)
        turns = []
        for position, turn_data in enumerate(turns_data, start=1):
            prefix = f"turns[{position - 1}]"
            if not isinstance(turn_data, dict):
                raise SessionFormatError(f"{prefix} has the wrong type; expected object.")
            for key in (
                "index",
                "question",
                "asked_at",
                "answer",
                "audio_path",
                "thinking_time_seconds",
                "answer_duration_seconds",
                "analysis",
            ):
                if key not in turn_data:
                    raise SessionFormatError(f"Missing required key: {prefix}.{key}.")
            index = _required(turn_data, "index", int)
            question = _required(turn_data, "question", str)
            asked_at = _validate_utc_timestamp(turn_data["asked_at"], f"{prefix}.asked_at")
            answer_data = turn_data["answer"]
            answer = (
                None if answer_data is None else _transcription_from_dict(answer_data)
            )
            audio_path = turn_data["audio_path"]
            if audio_path is not None and not isinstance(audio_path, str):
                raise SessionFormatError(f"{prefix}.audio_path must be a string or null.")
            thinking_time = _optional_float(turn_data, "thinking_time_seconds")
            answer_duration = _optional_float(turn_data, "answer_duration_seconds")
            turn_analysis = _required(turn_data, "analysis", dict)
            turns.append(
                Turn(
                    index=index,
                    question=question,
                    asked_at=asked_at,
                    answer=answer,
                    audio_path=audio_path,
                    thinking_time_seconds=thinking_time,
                    answer_duration_seconds=answer_duration,
                    analysis=turn_analysis,
                )
            )
        return cls(
            session_id=data["session_id"],
            created_at=created_at,
            mode=data["mode"],
            difficulty=data["difficulty"],
            planned_questions=planned_questions,
            metadata=metadata,
            turns=turns,
            analysis=analysis,
        )

    def save(self, directory: str | os.PathLike[str]) -> Path:
        """Atomically save the session as UTF-8 JSON and return its path."""
        target_directory = Path(directory)
        target_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = target_directory / f"session_{timestamp}_{self.session_id[:8]}.json"
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target_directory
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> InterviewSession:
        """Load and validate a session JSON file."""
        try:
            with Path(path).open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except json.JSONDecodeError as error:
            raise SessionFormatError(f"Corrupt session JSON: {error.msg}.") from None
        return cls.from_dict(data)
