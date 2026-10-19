from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.speaksure import ai_provider
from src.speaksure import speech_to_text
from src.speaksure.ai_provider import ProviderError
from src.speaksure.core import (
    INTRODUCTORY_QUESTIONS,
    PRACTICE_MODES,
    create_session,
    get_default_configuration,
    get_scripted_question,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def start_active_app(monkeypatch, *, mode: str, use_ai: bool, provider_class=None):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    if provider_class is not None:
        monkeypatch.setattr(ai_provider, "OpenAIConversationProvider", provider_class)

    app = AppTest.from_file(str(APP_PATH)).run()
    configuration = {**get_default_configuration(), "practice_mode": mode}
    app.session_state["speaksure_session"] = create_session(configuration)
    app.session_state["speaksure_questions"] = [INTRODUCTORY_QUESTIONS[mode]]
    app.session_state["speaksure_current_question"] = INTRODUCTORY_QUESTIONS[mode]
    app.session_state["speaksure_scripted_question_index"] = 0
    app.session_state["speaksure_use_ai"] = use_ai
    app.session_state["speaksure_ai_error"] = None
    app.session_state["speaksure_pending_ai"] = False
    return app.run()


def submit_answer(app, answer: str):
    app.text_area[0].set_value(answer)
    app.form_submit_button[0].click()
    return app.run()


def click_button(app, label: str):
    next(button for button in app.button if button.label == label).click()
    return app.run()


@pytest.mark.parametrize("mode", PRACTICE_MODES)
def test_app_advances_scripted_question_for_each_mode(monkeypatch, mode: str):
    app = start_active_app(monkeypatch, mode=mode, use_ai=False)

    submit_answer(app, "My response.")

    assert app.session_state["speaksure_current_question"] == get_scripted_question(mode, 1)
    assert app.session_state["speaksure_scripted_question_index"] == 1
    assert app.session_state["speaksure_session"]["transcript"] == ["My response."]


def test_retry_uses_latest_answer_without_duplicating_it(monkeypatch):
    calls = []

    class FailsOnceProvider:
        def next_question(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ProviderError("temporary failure")
            return "What did you learn from that?"

    app = start_active_app(
        monkeypatch, mode="HR interview", use_ai=True, provider_class=FailsOnceProvider
    )
    submit_answer(app, "The newest answer.")

    assert app.session_state["speaksure_session"]["transcript"] == ["The newest answer."]
    assert calls[0]["answers"] == ["The newest answer."]

    click_button(app, "Retry AI follow-up")

    assert len(calls) == 2
    assert calls[1]["answers"] == ["The newest answer."]
    assert app.session_state["speaksure_session"]["transcript"] == ["The newest answer."]
    assert app.session_state["speaksure_current_question"] == "What did you learn from that?"


def test_scripted_fallback_preserves_transcript_and_configuration(monkeypatch):
    calls = 0

    class BrokenProvider:
        def next_question(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ProviderError("temporary failure")
            return "What did you learn from the project?"

    app = start_active_app(
        monkeypatch,
        mode="Technical interview",
        use_ai=True,
        provider_class=BrokenProvider,
    )
    before = app.session_state["speaksure_session"]
    submit_answer(app, "I started by understanding the problem.")
    submit_answer(app, "I built a useful tool.")

    click_button(app, "Continue with scripted prompts")

    session = app.session_state["speaksure_session"]
    assert session["transcript"] == [
        "I started by understanding the problem.",
        "I built a useful tool.",
    ]
    assert session["configuration"] == before["configuration"]
    assert app.session_state["speaksure_current_question"] == get_scripted_question(
        "Technical interview", 2
    )
    assert app.session_state["speaksure_use_ai"] is False


def test_recognized_transcript_can_be_edited_and_submitted_once(monkeypatch):
    class FakeRecording:
        name = "recording.wav"
        type = "audio/wav"

        def getvalue(self):
            return b"mock wav audio"

    transcribed_audio = []

    class FakeTranscriber:
        def transcribe(self, audio, *, filename, content_type):
            transcribed_audio.append((audio, filename, content_type))
            return "Recognized answer."

    monkeypatch.setattr(
        speech_to_text, "create_speech_to_text_provider", lambda: FakeTranscriber()
    )
    monkeypatch.setattr(st, "audio_input", lambda *args, **kwargs: FakeRecording())
    app = start_active_app(monkeypatch, mode="HR interview", use_ai=False)

    click_button(app, "Transcribe recording")
    assert app.text_area[0].value == "Recognized answer."
    assert app.session_state["speaksure_session"]["transcript"] == []
    assert transcribed_audio == [(b"mock wav audio", "recording.wav", "audio/wav")]

    app.text_area[0].set_value("Edited and approved answer.")
    app.form_submit_button[0].click()
    app.run()

    assert app.session_state["speaksure_session"]["transcript"] == [
        "Edited and approved answer."
    ]


def test_microphone_without_recording_shows_permission_guidance(monkeypatch):
    app = start_active_app(monkeypatch, mode="HR interview", use_ai=False)

    click_button(app, "Transcribe recording")

    assert any(
        "Check browser microphone permission" in element.value
        for element in app.error
    )


def test_missing_transcription_configuration_keeps_typed_answers_available(
    monkeypatch,
):
    class FakeRecording:
        name = "recording.wav"
        type = "audio/wav"

        def getvalue(self):
            return b"wav bytes"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SPEECH_TO_TEXT_PROVIDER", "openai")
    monkeypatch.setattr(st, "audio_input", lambda *args, **kwargs: FakeRecording())
    app = start_active_app(monkeypatch, mode="HR interview", use_ai=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    click_button(app, "Transcribe recording")

    assert any(
        "Speech transcription is not configured" in element.value
        for element in app.error
    )
    assert len(app.text_area) == 1

    submit_answer(app, "Typed answer still works.")
    assert app.session_state["speaksure_session"]["transcript"] == [
        "Typed answer still works."
    ]
