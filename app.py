"""Streamlit dashboard and UI flow for SpeakSure practice sessions."""

from src.speaksure.configuration import get_api_key, get_speech_to_text_provider

import streamlit as st

from src.speaksure.core import (
    DIFFICULTIES, DURATIONS, INTRODUCTORY_QUESTIONS, PRACTICE_MODES,
    add_answer, complete_session, create_session, get_default_configuration,
    get_scripted_question,
)
from src.speaksure.ai_provider import (
    MissingConfiguration, OpenAIConversationProvider, ProviderError,
)
from src.speaksure.speech_to_text import (
    SpeechToTextError, create_speech_to_text_provider,
)

st.set_page_config(page_title="SpeakSure", page_icon="🎙️", layout="wide")
st.markdown("""<style>.block-container {max-width: 1000px; padding-top: 2.5rem;}
[data-testid="stMetric"] {background: #f4f7fb; padding: 1rem; border-radius: 12px;}</style>""", unsafe_allow_html=True)
st.title("?? SpeakSure")
st.write("Build confidence in English through realistic, interactive practice.")
st.caption("Practice by typing or speaking")

if "speaksure_session" not in st.session_state:
    st.session_state.speaksure_session = None

session = st.session_state.speaksure_session
if session is None:
    defaults = get_default_configuration()
    st.subheader("Set up a practice session")
    st.write("Choose a scenario and pace that fit the conversation you want to practice.")
    with st.form("practice_setup"):
        left, right = st.columns(2)
        with left:
            duration = st.selectbox("Session duration", DURATIONS, index=DURATIONS.index(defaults["duration_minutes"]), format_func=lambda value: f"{value} minutes")
            mode = st.selectbox("Practice mode", PRACTICE_MODES, index=PRACTICE_MODES.index(defaults["practice_mode"]))
        with right:
            difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=DIFFICULTIES.index(defaults["difficulty"]))
            ai_configured = bool(get_api_key())
            st.info("AI follow-up questions are available." if ai_configured else "AI is not configured; the session will use scripted prompts.")
        started = st.form_submit_button("Start Practice", type="primary", use_container_width=True)
    if started:
        try:
            st.session_state.speaksure_session = create_session({"duration_minutes": duration, "practice_mode": mode, "difficulty": difficulty})
            st.session_state.speaksure_questions = [INTRODUCTORY_QUESTIONS[mode]]
            st.session_state.speaksure_current_question = INTRODUCTORY_QUESTIONS[mode]
            st.session_state.speaksure_scripted_question_index = 0
            st.session_state.speaksure_use_ai = ai_configured
            st.session_state.speaksure_ai_error = None
            st.session_state.speaksure_pending_ai = False
            st.rerun()
        except ValueError as error:
            st.error(str(error))
elif session["status"] == "active_text_prototype":
    config = session["configuration"]
    st.subheader("Practice session")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Duration", f"{config['duration_minutes']} min")
    metric2.metric("Mode", config["practice_mode"])
    metric3.metric("Difficulty", config["difficulty"])
    use_ai = st.session_state.get("speaksure_use_ai", False)
    st.info("**AI conversation ? context-aware follow-ups**" if use_ai else "**Scripted conversation ? fixed practice prompts**")
    question = st.session_state.get("speaksure_current_question", INTRODUCTORY_QUESTIONS[config["practice_mode"]])
    st.markdown("#### Interviewer question")
    st.write(question)

    if session["transcript"]:
        st.markdown("#### Session transcript")
        for index, answer in enumerate(session["transcript"], start=1):
            st.markdown(f"**Your answer {index}**")
            st.write(answer)

    st.markdown("#### Speak your answer")
    selected_transcriber = get_speech_to_text_provider()
    if selected_transcriber == "openai":
        st.caption(
            "Allow microphone access, record a response, and choose Transcribe recording. "
            "Review or edit the transcript before submitting. OpenAI transcription sends audio "
            "to OpenAI and may incur charges."
        )
    elif selected_transcriber == "local":
        st.caption(
            "Allow microphone access, record a response, and choose Transcribe recording. "
            "Local transcription runs on this computer; the first use downloads model weights."
        )
    else:
        st.warning(
            "Speech transcription is unavailable because SPEECH_TO_TEXT_PROVIDER must be "
            "'local' or 'openai'. Typed answers are still available."
        )
    recording = st.audio_input("Record an answer", sample_rate=16000, key="speaksure_audio_input")
    if st.button("Transcribe recording", key="speaksure_transcribe"):
        st.session_state.speaksure_transcription_error = None
        st.session_state.speaksure_transcription_notice = None
        st.session_state.speaksure_transcription_device = None
        if recording is None:
            st.session_state.speaksure_transcription_error = (
                "No microphone recording was received. Check browser microphone permission and record audio first."
            )
        else:
            try:
                transcription_provider = create_speech_to_text_provider()
                transcript = transcription_provider.transcribe(
                    recording.getvalue(),
                    filename=recording.name or "recording.wav",
                    content_type=recording.type or "",
                )
                st.session_state.speaksure_answer = transcript
                if selected_transcriber == "local":
                    st.session_state.speaksure_transcription_device = getattr(
                        transcription_provider, "actual_device", "unknown"
                    )
                st.session_state.speaksure_transcription_notice = (
                    "Review or edit the recognized transcript below before submitting."
                )
            except SpeechToTextError as error:
                st.session_state.speaksure_transcription_error = str(error)
    if st.session_state.get("speaksure_transcription_error"):
        st.error(st.session_state.speaksure_transcription_error)
    if st.session_state.get("speaksure_transcription_notice"):
        st.success(st.session_state.speaksure_transcription_notice)
    if st.session_state.get("speaksure_transcription_device"):
        st.caption(
            f"Local model inference ran on {st.session_state.speaksure_transcription_device.upper()}."
        )

    def request_follow_up() -> None:
        try:
            latest_session = st.session_state.speaksure_session
            latest_configuration = latest_session["configuration"]
            provider = OpenAIConversationProvider()
            next_question = provider.next_question(
                practice_mode=latest_configuration["practice_mode"],
                difficulty=latest_configuration["difficulty"],
                questions=st.session_state.speaksure_questions,
                answers=latest_session["transcript"],
            )
            st.session_state.speaksure_questions.append(next_question)
            st.session_state.speaksure_current_question = next_question
            st.session_state.speaksure_ai_error = None
            st.session_state.speaksure_pending_ai = False
        except MissingConfiguration as error:
            st.session_state.speaksure_ai_error = str(error)
            st.session_state.speaksure_pending_ai = True
        except ProviderError as error:
            st.session_state.speaksure_ai_error = str(error)
            st.session_state.speaksure_pending_ai = True

    if st.session_state.get("speaksure_ai_error"):
        st.error(st.session_state.speaksure_ai_error)
        retry_col, fallback_col = st.columns(2)
        if retry_col.button("Retry AI follow-up", type="primary"):
            request_follow_up()
            st.rerun()
        if fallback_col.button("Continue with scripted prompts"):
            fallback_session = st.session_state.speaksure_session
            fallback_mode = fallback_session["configuration"]["practice_mode"]
            scripted_index = st.session_state.speaksure_scripted_question_index
            scripted_question = get_scripted_question(fallback_mode, scripted_index)
            if st.session_state.speaksure_questions[-1] != scripted_question:
                st.session_state.speaksure_questions.append(scripted_question)
            st.session_state.speaksure_current_question = scripted_question
            st.session_state.speaksure_use_ai = False
            st.session_state.speaksure_ai_error = None
            st.session_state.speaksure_pending_ai = False
            st.rerun()
    elif not st.session_state.get("speaksure_pending_ai", False):
        with st.form("answer_form", clear_on_submit=True):
            answer = st.text_area(
                "Your answer",
                placeholder="Type your response here or review the recognized transcript above.",
                height=130,
                key="speaksure_answer",
            )
            submitted = st.form_submit_button("Submit answer", type="primary")
        if submitted:
            try:
                updated = add_answer(session, answer)
                st.session_state.speaksure_session = updated
                st.session_state.speaksure_transcription_notice = None
                st.session_state.speaksure_transcription_error = None
                scripted_index = st.session_state.speaksure_scripted_question_index + 1
                st.session_state.speaksure_scripted_question_index = scripted_index
                if use_ai:
                    request_follow_up()
                else:
                    scripted_question = get_scripted_question(config["practice_mode"], scripted_index)
                    st.session_state.speaksure_questions.append(scripted_question)
                    st.session_state.speaksure_current_question = scripted_question
                st.rerun()
            except ValueError as error:
                st.warning(str(error))

    if st.button("End session", type="secondary"):
        st.session_state.speaksure_session = complete_session(st.session_state.speaksure_session)
        st.rerun()
elif session["status"] == "completed":
    config = session["configuration"]
    st.subheader("Session summary")
    st.success("Practice session completed.")
    st.write(f"You completed a **{config['duration_minutes']} minute** **{config['practice_mode']}** session at **{config['difficulty'].lower()}** difficulty.")
    st.metric("Answers submitted", len(session["transcript"]))
    if session["transcript"]:
        with st.expander("Review your transcript"):
            for index, answer in enumerate(session["transcript"], start=1):
                st.markdown(f"**Your answer {index}**")
                st.write(answer)
    if st.button("Start a new session", type="primary"):
        st.session_state.speaksure_session = None
        for key in (
            "speaksure_questions",
            "speaksure_current_question",
            "speaksure_scripted_question_index",
            "speaksure_use_ai",
            "speaksure_ai_error",
            "speaksure_pending_ai",
            "speaksure_answer",
            "speaksure_transcription_error",
            "speaksure_transcription_notice",
            "speaksure_transcription_device",
        ):
            st.session_state.pop(key, None)
        st.rerun()
