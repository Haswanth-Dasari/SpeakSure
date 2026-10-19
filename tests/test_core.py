import pytest

from src.speaksure.core import (
    DIFFICULTIES,
    DURATIONS,
    PRACTICE_MODES,
    add_answer,
    complete_session,
    create_session,
    get_default_configuration,
    get_scripted_question,
    INTRODUCTORY_QUESTIONS,
    SCRIPTED_QUESTION_BANK,
    validate_configuration,
)


def test_default_configuration() -> None:
    assert get_default_configuration() == {
        "duration_minutes": 10,
        "practice_mode": "HR interview",
        "difficulty": "Intermediate",
    }


@pytest.mark.parametrize("duration", DURATIONS)
@pytest.mark.parametrize("mode", PRACTICE_MODES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_valid_configuration_values(duration: int, mode: str, difficulty: str) -> None:
    config = validate_configuration(
        {"duration_minutes": duration, "practice_mode": mode, "difficulty": difficulty}
    )
    assert config == {"duration_minutes": duration, "practice_mode": mode, "difficulty": difficulty}


@pytest.mark.parametrize(
    "overrides",
    [
        {"duration_minutes": 12},
        {"duration_minutes": True},
        {"practice_mode": "Debate"},
        {"difficulty": "Expert"},
        {"difficulty": None},
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict[str, object]) -> None:
    config = {**get_default_configuration(), **overrides}
    with pytest.raises(ValueError):
        validate_configuration(config)


def test_starting_session_validates_configuration_and_sets_state() -> None:
    session = create_session(get_default_configuration())
    assert session == {
        "status": "active_text_prototype",
        "configuration": get_default_configuration(),
        "transcript": [],
    }


def test_answers_append_without_losing_previous_turns() -> None:
    session = create_session(get_default_configuration())
    session = add_answer(session, "  First answer.  ")
    session = add_answer(session, "Second answer.")
    assert session["transcript"] == ["First answer.", "Second answer."]


def test_blank_answer_is_rejected_without_mutating_session() -> None:
    session = create_session(get_default_configuration())
    with pytest.raises(ValueError, match="blank"):
        add_answer(session, "  \n")
    assert session["transcript"] == []


def test_completing_session_preserves_transcript_and_configuration() -> None:
    session = add_answer(create_session(get_default_configuration()), "I enjoy solving problems.")
    completed = complete_session(session)
    assert completed["status"] == "completed"
    assert completed["transcript"] == ["I enjoy solving problems."]
    assert completed["configuration"] == get_default_configuration()


def test_cannot_answer_after_session_is_completed() -> None:
    completed = complete_session(create_session(get_default_configuration()))
    with pytest.raises(ValueError, match="active"):
        add_answer(completed, "Too late")


def test_cannot_complete_session_twice() -> None:
    completed = complete_session(create_session(get_default_configuration()))
    with pytest.raises(ValueError, match="active"):
        complete_session(completed)


@pytest.mark.parametrize("mode", PRACTICE_MODES)
def test_scripted_questions_progress_for_each_mode_and_wrap(mode: str) -> None:
    questions = [
        get_scripted_question(mode, index)
        for index in range(len(SCRIPTED_QUESTION_BANK[mode]) + 1)
    ]
    assert questions[0] == INTRODUCTORY_QUESTIONS[mode]
    assert len(set(questions)) == len(questions)
    assert get_scripted_question(mode, len(questions)) == questions[0]


def test_scripted_question_rejects_invalid_mode_or_index() -> None:
    with pytest.raises(ValueError, match="Practice mode"):
        get_scripted_question("Debate", 0)
    with pytest.raises(ValueError, match="index"):
        get_scripted_question("HR interview", -1)
