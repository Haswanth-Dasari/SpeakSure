"""Fake-collaborator tests for half-duplex live interview orchestration."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from src.speaksure.audio_capture import CapturedAnswer, SAMPLE_RATE
from src.speaksure.interviewer import InterviewPlan, InterviewerTurn
from src.speaksure.live_interview import (
    LiveInterviewRunner,
    LiveSettings,
    SpokenTiming,
    format_transcript_flags,
)
from src.speaksure.question_bank import select_questions
from src.speaksure.session_log import InterviewSession
from src.speaksure.speech_to_text import TranscriptWord, TranscriptionResult


def transcript(
    text: str = "A useful answer",
    probabilities: tuple[float, ...] = (0.9, 0.9, 0.9),
) -> TranscriptionResult:
    """Build a transcription result whose word confidences match its text."""
    words = tuple(
        TranscriptWord(word, index * 0.2, (index + 1) * 0.2, probability)
        for index, (word, probability) in enumerate(
            zip(text.split(), probabilities)
        )
    )
    return TranscriptionResult(
        text=text,
        words=words,
        duration_seconds=1.0,
        language="en",
        language_probability=1.0,
        device="fake",
    )


def captured(
    *,
    stop_reason: str = "silence",
    samples: np.ndarray | None = None,
    thinking: float = 1.0,
    answer_duration: float = 2.0,
) -> CapturedAnswer:
    """Build a captured-answer object for runner tests."""
    return CapturedAnswer(
        samples=(
            np.array([0.1, -0.1], dtype=np.float32)
            if samples is None
            else samples
        ),
        sample_rate=SAMPLE_RATE,
        thinking_time_seconds=thinking,
        answer_duration_seconds=answer_duration,
        stop_reason=stop_reason,
        noise_rms=0.002,
        threshold=0.008,
    )


class FakeInterviewer:
    """Provide scripted interviewer replies and expose the planned bank questions."""

    def __init__(self, replies: list[InterviewerTurn], question_count: int = 3):
        """Set the interview plan and ordered replies for the runner."""
        self.plan = SimpleNamespace(
            questions=select_questions(question_count, seed=0)
        )
        self.replies = deque(replies)
        self.finished = False
        self.answers: list[str] = []

    @property
    def is_finished(self) -> bool:
        """Return whether a scripted closing reply has been consumed."""
        return self.finished

    def opening(self) -> InterviewerTurn:
        """Return the scripted opening turn."""
        return InterviewerTurn(
            utterance="Welcome. Tell me about yourself.",
            kind="opening",
            question_id=self.plan.questions[0].id,
            rationale="Opening",
            llm_latency_seconds=None,
            used_fallback=False,
            fallback_reason=None,
        )

    def next_turn(self, answer: str) -> InterviewerTurn:
        """Record an answer and return its scripted interviewer response."""
        self.answers.append(answer)
        reply = self.replies.popleft()
        if reply.kind == "closing":
            self.finished = True
        return reply


class FakeClock:
    """Advance in a deterministic way on every clock read."""

    def __init__(self, step: float = 0.1):
        """Initialize the clock's current value and increment."""
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        """Return the current fake time, then advance by one step."""
        current = self.value
        self.value += self.step
        return current


def reply(
    kind: str,
    utterance: str,
    *,
    question_id: str | None = "test-question",
    latency: float | None = 0.2,
    fallback: bool = False,
) -> InterviewerTurn:
    """Create a scripted interviewer response."""
    return InterviewerTurn(
        utterance=utterance,
        kind=kind,
        question_id=question_id,
        rationale=f"Reason for {kind}",
        llm_latency_seconds=latency,
        used_fallback=fallback,
        fallback_reason="test-fallback" if fallback else None,
    )


def make_session(question_count: int = 3) -> InterviewSession:
    """Create a session with test metadata and the requested question count."""
    return InterviewSession.new(
        "hr",
        "standard",
        question_count,
        metadata={"end_silence_seconds": 1.5},
    )


def test_format_transcript_flags_and_preserves_text_without_word_metadata() -> None:
    """Low-confidence words are marked, while text-only transcripts stay unchanged."""
    result = transcript("Maybe this works", (0.9, 0.3, 0.8))

    assert format_transcript_flags(result) == "Maybe [this?] works"
    text_only = TranscriptionResult("Plain transcript.", (), 0, "en", 1, "fake")
    assert format_transcript_flags(text_only) == "Plain transcript."


def test_three_question_run_speaks_in_order_saves_each_turn_and_stops_at_closing(
    tmp_path: Path,
) -> None:
    """Three questions create three turns and persist before the next listen."""
    questions = select_questions(3, seed=0)
    interviewer = FakeInterviewer(
        [
            reply("next_question", questions[1].text),
            reply("next_question", questions[2].text),
            reply("closing", "The interview is complete.", latency=None),
        ]
    )
    session = make_session()
    captures = deque([captured(), captured(), captured()])
    events: list[str] = []
    spoken: list[str] = []
    outputs: list[str] = []
    results = deque(
        [
            transcript("First candidate answer"),
            transcript("Second candidate answer"),
            transcript("Third candidate answer"),
        ]
    )

    def listen() -> CapturedAnswer:
        """Ensure each listen happens after playback and the prior save."""
        assert not speaker.active
        if len(events) >= 4:
            assert list(tmp_path.glob("session_*.json"))
        events.append("listen")
        return captures.popleft()

    def speak(text: str) -> SpokenTiming:
        """Record speech ordering and return deterministic timings."""
        speaker.active = True
        events.append("speak")
        spoken.append(text)
        speaker.active = False
        return SpokenTiming(0.2, 0.3)

    speaker = SimpleNamespace(active=False)
    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path),
        listen=listen,
        transcribe=lambda _capture: results.popleft(),
        speak=speak,
        output=outputs.append,
        sleep=lambda seconds: events.append(f"sleep:{seconds}"),
        clock=FakeClock(),
    )

    completed = runner.run()

    assert completed is session
    assert len(session.turns) == 3
    assert [turn.question for turn in session.turns] == [
        question.text for question in questions
    ]
    assert [event for event in events if event.startswith("sleep:")] == [
        "sleep:0.4",
        "sleep:0.4",
        "sleep:0.4",
    ]
    assert events.index("speak") < events.index("sleep:0.4") < events.index("listen")
    assert events.count("listen") == 3
    assert outputs[-1] == "Interviewer: The interview is complete."
    assert spoken == [
        "Welcome. Tell me about yourself.",
        questions[1].text,
        questions[2].text,
        "The interview is complete.",
    ]
    assert runner.session_path is not None and runner.session_path.is_file()
    assert all(
        turn.analysis["timing"]["tts_synthesis_seconds"] == 0.2
        for turn in session.turns
    )
    assert session.turns[0].analysis["timing"]["stt_seconds"] == pytest.approx(0.1)
    assert session.turns[0].analysis["timing"]["llm_seconds"] == pytest.approx(0.1)
    assert session.turns[0].analysis["timing"][
        "response_gap_estimate_seconds"
    ] == pytest.approx(0.4)
    assert session.turns[0].analysis["timing"]["end_silence_seconds"] == 1.5
    assert session.turns[2].analysis["timing"]["llm_seconds"] is None


def test_no_speech_repeats_without_recording_a_turn_until_retry(tmp_path: Path) -> None:
    """A no-speech attempt triggers repeat and is omitted from session turns."""
    interviewer = FakeInterviewer(
        [reply("repeat", "Please try again.", latency=None), reply("closing", "Done.", latency=None)],
        question_count=1,
    )
    session = make_session(1)
    captures = deque(
        [
            captured(stop_reason="no_speech", samples=np.empty(0, np.float32)),
            captured(),
        ]
    )
    listens = 0

    def listen() -> CapturedAnswer:
        """Track listening and ensure the first repeat did not add a turn."""
        nonlocal listens
        listens += 1
        if listens == 2:
            assert not session.turns
        return captures.popleft()

    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path, save_audio=False),
        listen=listen,
        transcribe=lambda _capture: transcript(),
        speak=lambda _text: SpokenTiming(0.1, 0.2),
        output=lambda _text: None,
        sleep=lambda _seconds: None,
        clock=FakeClock(),
    )

    runner.run()

    assert len(session.turns) == 1
    assert session.turns[0].analysis["empty_attempts"] == 1
    assert interviewer.answers == ["", "A useful answer"]


def test_one_word_transcript_is_treated_as_empty_but_raw_result_is_logged(
    tmp_path: Path,
) -> None:
    """A one-word transcript triggers repeat and remains available for the Turn."""
    interviewer = FakeInterviewer(
        [reply("repeat", "Say more.", latency=None), reply("closing", "Done.", latency=None)],
        question_count=1,
    )
    session = make_session(1)
    results = deque([transcript("Yes", (0.2,)), transcript("A detailed answer")])

    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path, save_audio=False),
        listen=lambda: captured(),
        transcribe=lambda _capture: results.popleft(),
        speak=lambda _text: SpokenTiming(0.1, 0.2),
        output=lambda _text: None,
        sleep=lambda _seconds: None,
        clock=FakeClock(),
    )

    runner.run()

    assert interviewer.answers == ["", "A detailed answer"]
    assert session.turns[0].answer is not None
    assert session.turns[0].answer.text == "A detailed answer"
    assert session.turns[0].analysis["empty_attempts"] == 1


def test_low_confidence_words_are_recorded_in_turn_analysis(tmp_path: Path) -> None:
    """Words under the confidence threshold are persisted as structured flags."""
    interviewer = FakeInterviewer([reply("closing", "Done.", latency=None)], question_count=1)
    session = make_session(1)
    result = transcript("A possibly answer", (0.9, 0.2, 0.4))

    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path, save_audio=False),
        listen=lambda: captured(),
        transcribe=lambda _capture: result,
        speak=lambda _text: SpokenTiming(0.1, 0.2),
        output=lambda _text: None,
        sleep=lambda _seconds: None,
        clock=FakeClock(),
    )

    runner.run()

    assert session.turns[0].analysis["low_confidence_words"] == [
        {"word": "possibly", "probability": 0.2},
        {"word": "answer", "probability": 0.4},
    ]


def test_transcription_exception_apologizes_repeats_and_continues(tmp_path: Path) -> None:
    """A transcription failure is retried without creating an extra Turn."""
    interviewer = FakeInterviewer([reply("closing", "Done.", latency=None)], question_count=1)
    session = make_session(1)
    transcriptions: deque[object] = deque(
        [RuntimeError("private transcription failure"), transcript()]
    )
    spoken: list[str] = []

    def transcribe(_capture: CapturedAnswer) -> TranscriptionResult:
        """Raise the first scripted failure, then return the transcription."""
        result = transcriptions.popleft()
        if isinstance(result, Exception):
            raise result
        assert isinstance(result, TranscriptionResult)
        return result

    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path, save_audio=False),
        listen=lambda: captured(),
        transcribe=transcribe,
        speak=lambda text: spoken.append(text) or SpokenTiming(0.1, 0.2),
        output=lambda _text: None,
        sleep=lambda _seconds: None,
        clock=FakeClock(),
    )

    runner.run()

    assert len(session.turns) == 1
    assert session.turns[0].analysis["empty_attempts"] == 1
    assert any("Sorry" in message for message in spoken)
    assert interviewer.answers == ["A useful answer"]


def test_speech_failure_is_logged_and_does_not_stop_interview(tmp_path: Path) -> None:
    """A failed TTS response falls back to printed text and is logged."""
    interviewer = FakeInterviewer([reply("closing", "The end.", latency=None)], question_count=1)
    session = make_session(1)

    def speak(text: str) -> SpokenTiming:
        """Fail on the closing utterance while allowing the opening."""
        if text == "The end.":
            raise RuntimeError("speaker unavailable")
        return SpokenTiming(0.1, 0.2)

    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path, save_audio=False),
        listen=lambda: captured(),
        transcribe=lambda _capture: transcript(),
        speak=speak,
        output=lambda _text: None,
        sleep=lambda _seconds: None,
        clock=FakeClock(),
    )

    runner.run()

    assert len(session.turns) == 1
    assert session.turns[0].analysis["speech_output_failed"] is True
    assert session.turns[0].analysis["timing"]["tts_synthesis_seconds"] is None


def test_keyboard_interrupt_saves_session_and_returns_gracefully(tmp_path: Path) -> None:
    """An interrupted listen returns the session after saving it."""
    interviewer = FakeInterviewer([], question_count=1)
    session = make_session(1)
    outputs: list[str] = []

    def interrupt() -> CapturedAnswer:
        """Simulate Ctrl+C during microphone listening."""
        raise KeyboardInterrupt

    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path, save_audio=False),
        listen=interrupt,
        transcribe=lambda _capture: transcript(),
        speak=lambda _text: SpokenTiming(0.1, 0.2),
        output=outputs.append,
        sleep=lambda _seconds: None,
        clock=FakeClock(),
    )

    assert runner.run() is session
    assert outputs[-1] == "Interview stopped early."
    assert runner.session_path is not None and runner.session_path.is_file()


@pytest.mark.parametrize("save_audio", [True, False])
def test_audio_files_are_saved_only_when_enabled_and_path_is_recorded(
    tmp_path: Path, save_audio: bool
) -> None:
    """Captured WAV data is saved and referenced only when requested."""
    interviewer = FakeInterviewer([reply("closing", "Done.", latency=None)], question_count=1)
    session = make_session(1)
    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(sessions_dir=tmp_path, save_audio=save_audio),
        listen=lambda: captured(),
        transcribe=lambda _capture: transcript(),
        speak=lambda _text: SpokenTiming(0.1, 0.2),
        output=lambda _text: None,
        sleep=lambda _seconds: None,
        clock=FakeClock(),
    )

    runner.run()

    audio_path = session.turns[0].audio_path
    if save_audio:
        assert audio_path is not None
        assert Path(audio_path).is_file()
        assert Path(audio_path).parent == tmp_path / session.session_id
    else:
        assert audio_path is None
        assert not list(tmp_path.rglob("*.wav"))


def test_sleep_precedes_every_listen_and_listen_never_overlaps_speaking(
    tmp_path: Path,
) -> None:
    """The settle delay follows every speech completion before listening begins."""
    interviewer = FakeInterviewer(
        [reply("next_question", "Next?"), reply("closing", "Done.", latency=None)],
        question_count=2,
    )
    session = make_session(2)
    order: list[str] = []
    speaking = False

    def speak(_text: str) -> SpokenTiming:
        """Track active playback for the half-duplex assertion."""
        nonlocal speaking
        assert not speaking
        speaking = True
        order.append("speak-start")
        speaking = False
        order.append("speak-end")
        return SpokenTiming(0.1, 0.2)

    def listen() -> CapturedAnswer:
        """Assert the microphone only runs after speech and settle sleep."""
        assert not speaking
        assert order[-1] == "settle"
        order.append("listen")
        return captured()

    runner = LiveInterviewRunner(
        interviewer=interviewer,  # type: ignore[arg-type]
        session=session,
        settings=LiveSettings(settle_seconds=0.7, sessions_dir=tmp_path),
        listen=listen,
        transcribe=lambda _capture: transcript(),
        speak=speak,
        output=lambda _text: None,
        sleep=lambda seconds: order.append("settle")
        if seconds == 0.7
        else pytest.fail("Unexpected settle duration"),
        clock=FakeClock(),
    )

    runner.run()

    assert order.count("listen") == 2
    for index, event in enumerate(order):
        if event == "listen":
            assert order[index - 1] == "settle"
            assert order[index - 2] == "speak-end"
