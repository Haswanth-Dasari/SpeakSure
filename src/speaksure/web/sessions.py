"""Thread-safe registry and disk helpers for local web interview sessions."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..interviewer import Interviewer
from ..session_log import InterviewSession

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class WebSession:
    """An active interview and its web-only lifecycle and file state."""

    session: InterviewSession
    interviewer: Interviewer
    current_question: str
    current_question_id: str
    question_index: int
    total_main_questions: int
    session_path: Path
    spoken_lines: dict[str, str] = field(default_factory=dict)
    spoken_line_count: int = 0
    state: str = "active"
    message: str = ""
    report_path: Path | None = None
    empty_attempts: int = 0
    listen_attempts: int = 0
    transcription_errors: list[dict[str, object]] = field(default_factory=list)
    current_is_follow_up: bool = False
    content_requested: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class WebSessionRegistry:
    """Store active sessions behind a lock shared by the local API threads."""

    def __init__(self) -> None:
        """Initialize an empty registry and its synchronization lock."""
        self._sessions: dict[str, WebSession] = {}
        self._lock = threading.RLock()

    def add(self, entry: WebSession) -> None:
        """Register an active session after validating its identifier."""
        session_id = entry.session.session_id
        if not is_session_id(session_id):
            raise ValueError("Session identifiers must be 32 lowercase hexadecimal characters.")
        with self._lock:
            self._sessions[session_id] = entry

    def get(self, session_id: str) -> WebSession | None:
        """Return a registered session, or None when it is not active here."""
        if not is_session_id(session_id):
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def snapshot(self) -> list[WebSession]:
        """Return a stable shallow snapshot of active session entries."""
        with self._lock:
            return list(self._sessions.values())


def is_session_id(value: str) -> bool:
    """Validate a session identifier without interpreting it as a file path."""
    return SESSION_ID_PATTERN.fullmatch(value) is not None


def create_session_record(
    question_count: int, metadata: dict[str, Any]
) -> InterviewSession:
    """Create a session using the existing serializable session model."""
    return InterviewSession.new(
        mode="hr",
        difficulty="standard",
        planned_questions=question_count,
        metadata=metadata,
    )
