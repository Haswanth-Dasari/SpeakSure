"""Deterministic configuration and session logic for SpeakSure."""

from typing import TypedDict


DURATIONS = (10, 15, 30, 60)
PRACTICE_MODES = (
    "HR interview",
    "Technical interview",
    "Team collaboration meeting",
    "Everyday conversation",
)
DIFFICULTIES = ("Beginner", "Intermediate", "Advanced")
DEFAULT_CONFIGURATION = {
    "duration_minutes": 10,
    "practice_mode": "HR interview",
    "difficulty": "Intermediate",
}
INTRODUCTORY_QUESTIONS = {
    "HR interview": "Tell me about yourself and what brings you to this opportunity.",
    "Technical interview": "Can you describe a technical project you enjoyed working on?",
    "Team collaboration meeting": "What helps you contribute effectively during a team discussion?",
    "Everyday conversation": "What is something interesting you have done recently?",
}
SCRIPTED_QUESTION_BANK = {
    "HR interview": (
        "What strengths would you bring to this role?",
        "Can you describe a challenge you handled at work?",
        "What kind of team helps you do your best work?",
    ),
    "Technical interview": (
        "What was the most difficult problem you solved in that project?",
        "How did you decide which tools or technologies to use?",
        "What would you improve if you worked on it again?",
    ),
    "Team collaboration meeting": (
        "How do you make space for different opinions in a discussion?",
        "What do you do when a team disagrees about priorities?",
        "Can you share an example of helping a teammate?",
    ),
    "Everyday conversation": (
        "What did you enjoy most about that experience?",
        "Who did you share it with?",
        "What is something you would like to try next?",
    ),
}


class Configuration(TypedDict):
    duration_minutes: int
    practice_mode: str
    difficulty: str


class Session(TypedDict):
    status: str
    configuration: Configuration
    transcript: list[str]


def get_default_configuration() -> Configuration:
    """Return a fresh copy of the default session configuration."""
    return dict(DEFAULT_CONFIGURATION)  # type: ignore[return-value]


def get_scripted_question(practice_mode: str, question_index: int) -> str:
    """Return a mode-specific prompt, wrapping to the opening question after the bank."""
    if practice_mode not in INTRODUCTORY_QUESTIONS:
        raise ValueError(f"Practice mode must be one of {PRACTICE_MODES}.")
    if question_index < 0:
        raise ValueError("Question index cannot be negative.")
    questions = (
        INTRODUCTORY_QUESTIONS[practice_mode],
        *SCRIPTED_QUESTION_BANK[practice_mode],
    )
    return questions[question_index % len(questions)]


def validate_configuration(configuration: dict[str, object]) -> Configuration:
    """Validate and normalize a session configuration, raising ValueError if invalid."""
    duration = configuration.get("duration_minutes")
    mode = configuration.get("practice_mode")
    difficulty = configuration.get("difficulty")
    if isinstance(duration, bool) or duration not in DURATIONS:
        raise ValueError(f"Duration must be one of {DURATIONS} minutes.")
    if mode not in PRACTICE_MODES:
        raise ValueError(f"Practice mode must be one of {PRACTICE_MODES}.")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Difficulty must be one of {DIFFICULTIES}.")
    return {
        "duration_minutes": int(duration),
        "practice_mode": str(mode),
        "difficulty": str(difficulty),
    }


def create_session(configuration: dict[str, object]) -> Session:
    """Validate configuration and create a new active text prototype session."""
    return {
        "status": "active_text_prototype",
        "configuration": validate_configuration(configuration),
        "transcript": [],
    }


def add_answer(session: Session, answer: str) -> Session:
    """Append a non-empty answer to an active session, preserving prior turns."""
    if session.get("status") != "active_text_prototype":
        raise ValueError("Answers can only be added to an active session.")
    cleaned = answer.strip()
    if not cleaned:
        raise ValueError("Answer cannot be blank.")
    return {
        **session,
        "transcript": [*session["transcript"], cleaned],
    }


def complete_session(session: Session) -> Session:
    """Mark an active session as completed."""
    if session.get("status") != "active_text_prototype":
        raise ValueError("Only an active session can be completed.")
    return {**session, "status": "completed"}
