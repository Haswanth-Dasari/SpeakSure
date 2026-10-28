"""Offline FastAPI endpoint tests with fake speech and interviewer services."""

from __future__ import annotations

import io
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi.testclient import TestClient

from scripts.run_web import main as run_web_main
from src.speaksure.interviewer import InterviewPlan, InterviewerTurn
from src.speaksure.speech_to_text import (
    LocalTranscriptionConfig,
    TranscriptWord,
    TranscriptionResult,
)
from src.speaksure.text_to_speech import SpeechSynthesisConfig, SynthesizedAudio
from src.speaksure.web.app import WebServices, WebSettings, create_app


def _wav_bytes(
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
    samples: np.ndarray | None = None,
) -> bytes:
    """Create a small WAV request body with configurable PCM format."""
    output = io.BytesIO()
    if samples is None:
        samples = np.full(sample_rate // 5, 0.15, dtype=np.float32)
    clipped = np.clip(samples, -1, 1)
    if sample_width == 2:
        frames = (clipped * 32767).astype("<i2").tobytes()
    elif sample_width == 1:
        frames = ((clipped + 1) * 127.5).astype("u1").tobytes()
    else:
        raise ValueError("test helper supports 8-bit or 16-bit samples")
    if channels == 2:
        frames = np.repeat(
            np.frombuffer(frames, dtype="<i2" if sample_width == 2 else "u1")
            .reshape(-1, 1),
            2,
            axis=1,
        ).tobytes()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return output.getvalue()


class FakeInterviewer:
    """Script interviewer actions and retain the submitted answer history."""

    def __init__(self, plan: InterviewPlan, actions: list[tuple[str, str]]) -> None:
        """Set the plan and its ordered scripted interviewer actions."""
        self.plan = plan
        self.actions = deque(actions)
        self.answers: list[str] = []
        self.finished = False

    @property
    def is_finished(self) -> bool:
        """Return whether a closing action was already consumed."""
        return self.finished

    def opening(self) -> InterviewerTurn:
        """Return the first prompt without a model call."""
        question = self.plan.questions[0]
        return InterviewerTurn(
            utterance=f"Welcome. {question.text}",
            kind="opening",
            question_id=question.id,
            rationale="fake opening",
            llm_latency_seconds=None,
            used_fallback=False,
            fallback_reason=None,
        )

    def next_turn(self, answer: str) -> InterviewerTurn:
        """Return the next scripted response and record the supplied answer."""
        self.answers.append(answer)
        kind, utterance = self.actions.popleft()
        if kind == "closing":
            self.finished = True
        question_id = self.plan.questions[0].id
        return InterviewerTurn(
            utterance=utterance,
            kind=kind,  # type: ignore[arg-type]
            question_id=question_id,
            rationale="fake response",
            llm_latency_seconds=0.01,
            used_fallback=False,
            fallback_reason=None,
        )


class FakeSTT:
    """Return scripted fake transcripts and count transcription calls."""

    def __init__(self, responses: list[object] | None = None) -> None:
        """Store transcript outputs or exceptions for subsequent requests."""
        self.responses = deque(responses or [])
        self.actual_device = "cpu"
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}

    def transcribe_detailed(self, *_args: object, **_kwargs: object) -> TranscriptionResult:
        """Return one fake transcript or raise the next scripted exception."""
        self.calls += 1
        self.last_kwargs = dict(_kwargs)
        response = self.responses.popleft() if self.responses else _transcript()
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, TranscriptionResult)
        return response

    def transcribe(self, *_args: object, **_kwargs: object) -> str:
        """Implement the base speech provider interface for completeness."""
        return self.transcribe_detailed().text


class FakeTTS:
    """Synthesize short silent WAV data without loading a voice model."""

    voice_name = "fake-voice"

    def __init__(self) -> None:
        """Initialize a call counter."""
        self.calls = 0
        self.fail = False

    def synthesize(self, _text: str) -> SynthesizedAudio:
        """Return deterministic mono samples for WAV output."""
        self.calls += 1
        if self.fail:
            raise RuntimeError("fake synthesis failure")
        return SynthesizedAudio(
            samples=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            synthesis_seconds=0.001,
        )


def _transcript(text: str = "A useful candidate answer.") -> TranscriptionResult:
    """Build deterministic recognized text and word timings."""
    return TranscriptionResult(
        text=text,
        words=tuple(
            TranscriptWord(word, index * 0.1, (index + 1) * 0.1, 0.9)
            for index, word in enumerate(text.split())
        ),
        duration_seconds=0.2,
        language="en",
        language_probability=1.0,
        device="cpu",
    )


def _client(
    tmp_path: Path,
    *,
    stt_config: LocalTranscriptionConfig | None = None,
    stt_responses: list[object] | None = None,
    actions: list[tuple[str, str]] | None = None,
    analysis: object | None = None,
    llm_configured: bool = False,
    warmup_functions: dict[str, Callable[[], object]] | None = None,
) -> tuple[TestClient, FakeSTT, FakeTTS, dict[str, object]]:
    """Build a FastAPI test client with isolated files and injectable fakes."""
    settings = WebSettings(
        stt_config=stt_config or LocalTranscriptionConfig(model_size="base", device="cpu"),
        tts_config=SpeechSynthesisConfig(
            voice_name="fake-voice",
            models_dir=tmp_path,
            length_scale=1.0,
        ),
        sessions_dir=tmp_path / "sessions",
        analysis_dir=tmp_path / "analysis",
        reports_dir=tmp_path / "reports",
        cache_dir=tmp_path / "web_cache",
    )
    stt = FakeSTT(stt_responses)
    tts = FakeTTS()
    created: dict[str, object] = {}
    scripted_actions = actions or [
        ("follow_up", "Could you add one example?"),
        ("repeat", "Please say that again."),
        ("next_question", "What motivates you?"),
        ("closing", "Thank you for your time."),
    ]

    def interviewer_factory(plan: InterviewPlan) -> FakeInterviewer:
        """Return an isolated scripted fake interviewer."""
        interviewer = FakeInterviewer(plan, scripted_actions.copy())
        created["interviewer"] = interviewer
        return interviewer  # type: ignore[return-value]

    def analysis_function(session, **_kwargs):
        """Signal that the background analysis function completed."""
        created["analysis_kwargs"] = _kwargs
        if callable(analysis):
            return analysis(session, **_kwargs)
        return session

    services = WebServices(
        stt_provider=stt,  # type: ignore[arg-type]
        tts_provider=tts,  # type: ignore[arg-type]
        interviewer_factory=interviewer_factory,  # type: ignore[arg-type]
        analysis_function=analysis_function,
        report_function=lambda _session: "<!doctype html><title>Fake report</title>",
        clock=lambda: 1.0,
        sessions_dir=settings.sessions_dir,
        analysis_dir=settings.analysis_dir,
        reports_dir=settings.reports_dir,
        cache_dir=settings.cache_dir,
        llm_configured=llm_configured,
        analysis_llm_factory=lambda: created.setdefault("analysis_llm", object()),
        warmup_functions=warmup_functions,
    )
    return TestClient(create_app(settings, services)), stt, tts, created


def _create_session(client: TestClient) -> dict[str, object]:
    """Create a three-question session with browser-capture metadata."""
    response = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "candidate_name": "Student",
            "client": {
                "input_device_label": "Bluetooth headset microphone",
                "constraints": {"echoCancellation": True, "noiseSuppression": True},
            },
        },
    )
    assert response.status_code == 200
    return response.json()


def _answer(client: TestClient, session_id: str) -> dict[str, object]:
    """Submit one generated WAV through the real API validation and flow."""
    response = client.post(
        f"/api/sessions/{session_id}/answer",
        content=_wav_bytes(),
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 200
    return response.json()


def test_health_and_create_session_validate_and_store_browser_metadata(tmp_path):
    client, stt, _tts, created = _client(tmp_path)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "components": [
            {"name": "stt", "status": "ready"},
            {"name": "tts", "status": "ready"},
            {"name": "spacy", "status": "ready"},
        ],
        "stt_model": "base",
        "stt_device_setting": "cpu",
        "tts_voice": "fake-voice",
        "llm_configured": False,
    }
    stt.actual_device = "not-initialized"
    loading = client.get("/api/health").json()
    assert loading["status"] == "loading"
    assert loading["components"][0] == {"name": "stt", "status": "loading"}
    stt.actual_device = "cpu"
    invalid = client.post(
        "/api/sessions",
        json={"questions": 2, "client": {"input_device_label": "Mic"}},
    )
    assert invalid.status_code == 422
    assert set(invalid.json()) == {"error", "message"}

    created_response = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "client": {"input_device_label": "Browser mic", "constraints": {"a": 1}},
        },
    )
    assert created_response.status_code == 200
    payload = created_response.json()
    assert len(payload["session_id"]) == 32
    assert payload["question_index"] == 1
    entry = client.app.state.registry.get(payload["session_id"])
    assert entry.session.metadata["capture_source"] == "browser"
    assert entry.session.metadata["input_device"] == "Browser mic"
    assert entry.session.metadata["capture_constraints"] == {"a": 1}
    assert entry.session.metadata["role"] == "software engineering fresher"
    assert entry.session.metadata["length"] == "standard"
    assert created["interviewer"] is not None


def test_model_warmups_run_in_background_and_health_lists_each_component(tmp_path: Path) -> None:
    """Keep health in loading state until all three fake model loaders finish."""
    calls: list[str] = []
    gate = threading.Event()

    def fake_warmup(name: str) -> Callable[[], None]:
        """Build a deterministic model warmup that waits for the test gate."""
        def warmup() -> None:
            """Record that this component started loading before the gate opens."""
            calls.append(name)
            gate.wait(timeout=2)
        return warmup

    client, _stt, _tts, _created = _client(
        tmp_path,
        warmup_functions={
            name: fake_warmup(name) for name in ("stt", "tts", "spacy")
        },
    )
    with client:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(calls) < 1:
            time.sleep(0.01)
        health = client.get("/api/health").json()
        assert health["status"] == "loading"
        assert health["components"] == [
            {"name": "stt", "status": "loading"},
            {"name": "tts", "status": "loading"},
            {"name": "spacy", "status": "loading"},
        ]
        gate.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            health = client.get("/api/health").json()
            if health["status"] == "ok":
                break
            time.sleep(0.01)
    assert calls == ["stt", "tts", "spacy"]
    assert health["status"] == "ok"


def test_role_length_mapping_and_explicit_questions_precedence(tmp_path: Path) -> None:
    """Apply preset question counts while preserving explicit counts and role."""
    client, _stt, _tts, created = _client(tmp_path)
    expected_counts = {"quick": 3, "standard": 5, "full": 8}
    for length, expected_count in expected_counts.items():
        response = client.post(
            "/api/sessions",
            json={
                "length": length,
                "role": "Data scientist",
                "candidate_name": "Student",
                "client": {"input_device_label": "Mic"},
            },
        )
        assert response.status_code == 200
        entry = client.app.state.registry.get(response.json()["session_id"])
        assert entry.total_main_questions == expected_count
        assert entry.session.metadata["role"] == "Data scientist"
        assert entry.session.metadata["length"] == length
        assert created["interviewer"].plan.role == "Data scientist"

    explicit = client.post(
        "/api/sessions",
        json={
            "questions": 4,
            "length": "quick",
            "role": "Product designer",
            "client": {"input_device_label": "Mic"},
        },
    )
    assert explicit.status_code == 200
    entry = client.app.state.registry.get(explicit.json()["session_id"])
    assert entry.total_main_questions == 4
    assert entry.session.planned_questions == 4
    assert entry.session.metadata["length"] == "quick"
    assert entry.interviewer.plan.role == "Product designer"

    defaulted = client.post(
        "/api/sessions",
        json={"client": {"input_device_label": "Mic"}},
    )
    assert defaulted.status_code == 200
    default_entry = client.app.state.registry.get(defaulted.json()["session_id"])
    assert default_entry.total_main_questions == 5
    assert default_entry.session.metadata["role"] == "software engineering fresher"

    client.post(f"/api/sessions/{explicit.json()['session_id']}/finish", json={})
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if client.get(f"/api/sessions/{explicit.json()['session_id']}/status").json()["state"] == "ready":
            break
        time.sleep(0.01)
    assert created["analysis_kwargs"]["role"] == "Product designer"

    for invalid_role in ("", " " * 4, "R" * 61):
        invalid = client.post(
            "/api/sessions",
            json={
                "role": invalid_role,
                "client": {"input_device_label": "Mic"},
            },
        )
        assert invalid.status_code == 422


def test_silent_upload_uses_shared_interviewer_silence_handling(tmp_path: Path) -> None:
    """Submit silent audio as a normal answer without invoking speech recognition."""
    client, stt, _tts, created = _client(
        tmp_path,
        actions=[("repeat", "Could you say that again?")],
    )
    session_id = _create_session(client)["session_id"]
    response = client.post(
        f"/api/sessions/{session_id}/answer",
        content=_wav_bytes(samples=np.zeros(16000, dtype=np.float32)),
        headers={"content-type": "audio/wav"},
    )
    assert response.status_code == 200
    assert response.json()["interviewer"]["action"] == "repeat"
    assert stt.calls == 0
    assert created["interviewer"].answers == [""]


def test_answer_timing_metadata_is_updated_from_browser_headers(tmp_path: Path) -> None:
    """Store server timings and browser-measured upload and answer-start values."""
    client, _stt, _tts, _created = _client(tmp_path)
    session_id = _create_session(client)["session_id"]
    result = _answer(client, session_id)
    entry = client.app.state.registry.get(session_id)
    timing = entry.session.turns[0].analysis["timing"]
    assert {"end_wait_s", "upload_s", "stt_s", "llm_s", "tts_s"} <= set(timing)
    assert timing["upload_s"] is None
    assert timing["tts_s"] == 0.001

    updated = client.patch(
        f"/api/sessions/{session_id}/turns/{result['turn_index']}/timing",
        headers={
            "X-Client-Upload-Ms": "123.5",
            "X-Client-End-Wait-Ms": "840",
        },
    )
    assert updated.status_code == 200
    stored = entry.session.turns[0].analysis["timing"]
    assert stored["upload_s"] == 0.1235
    assert stored["end_wait_s"] == 0.84


def test_analysis_status_exposes_callback_stage_and_omits_unknown_stage(tmp_path: Path) -> None:
    """Expose known progress values but omit stage fields when none are reported."""
    entered = threading.Event()
    release = threading.Event()

    def analysis(session: Any, **kwargs: Any) -> Any:
        """Report a deterministic stage and wait so the status route can observe it."""
        callback = kwargs["progress_callback"]
        callback("delivery", 0.4)
        entered.set()
        release.wait(timeout=2)
        return session

    client, _stt, _tts, _created = _client(tmp_path, analysis=analysis)
    session_id = _create_session(client)["session_id"]
    active = client.get(f"/api/sessions/{session_id}/status").json()
    assert "stage" not in active
    assert "progress" not in active
    assert client.post(f"/api/sessions/{session_id}/finish", json={}).status_code == 200
    assert entered.wait(timeout=2)
    status = client.get(f"/api/sessions/{session_id}/status").json()
    assert status["stage"] == "delivery"
    assert status["progress"] == 0.4
    release.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{session_id}/status").json()
        if status["state"] == "ready":
            break
        time.sleep(0.01)
    assert status["stage"] == "report"
    assert status["progress"] == 1.0


def test_audio_serves_cached_tts_and_rejects_unknown_or_traversal_names(tmp_path):
    client, _stt, tts, _created = _client(tmp_path)
    session = _create_session(client)
    session_id = session["session_id"]
    audio_url = session["first_question"]["audio_url"]

    assert tts.calls == 1
    first = client.get(audio_url)
    second = client.get(audio_url)

    assert first.status_code == 200
    assert first.headers["content-type"] == "audio/wav"
    assert first.content[:4] == b"RIFF"
    assert second.content == first.content
    assert tts.calls == 1
    unknown = client.get(f"/api/sessions/{session_id}/audio/../../secret.wav")
    assert unknown.status_code == 404
    assert set(unknown.json()) == {"error", "message"}


def test_answer_flow_supports_followup_repeat_next_question_and_finish(tmp_path):
    client, stt, tts, created = _client(tmp_path)
    session = _create_session(client)
    session_id = session["session_id"]

    follow_up = _answer(client, session_id)
    synthesized_calls = tts.calls
    ready_audio = client.get(follow_up["interviewer"]["audio_url"])
    assert ready_audio.status_code == 200
    assert ready_audio.content[:4] == b"RIFF"
    assert tts.calls == synthesized_calls
    repeated = _answer(client, session_id)
    next_question = _answer(client, session_id)
    finished = _answer(client, session_id)

    assert follow_up["interviewer"]["action"] == "follow_up"
    assert repeated["interviewer"]["action"] == "repeat"
    assert repeated["turn_index"] == 2
    assert next_question["interviewer"]["action"] == "next_question"
    assert finished["interviewer"]["action"] == "finish"
    assert finished["finished"] is True
    entry = client.app.state.registry.get(session_id)
    assert [turn.index for turn in entry.session.turns] == [1, 2, 3]
    assert entry.session.turns[0].analysis["fluency"]["word_count"] == 4
    assert "audio_delivery" in entry.session.turns[0].analysis
    assert created["interviewer"].answers == [
        "A useful candidate answer.",
        "A useful candidate answer.",
        "A useful candidate answer.",
        "A useful candidate answer.",
    ]
    assert stt.calls == 4
    assert tts.calls == 5
    assert client.post(
        f"/api/sessions/{session_id}/answer",
        content=_wav_bytes(),
        headers={"content-type": "audio/wav"},
    ).status_code == 409


def test_answer_reports_reply_audio_synthesis_failure(tmp_path):
    client, _stt, tts, _created = _client(tmp_path)
    session_id = _create_session(client)["session_id"]
    tts.fail = True

    response = client.post(
        f"/api/sessions/{session_id}/answer",
        content=_wav_bytes(),
        headers={"content-type": "audio/wav"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": "reply_audio_failed",
        "message": "The interviewer response audio could not be generated.",
    }


def test_audio_upload_rejects_invalid_content_type_wav_format_size_and_session(
    tmp_path,
):
    client, _stt, _tts, _created = _client(tmp_path)
    session_id = _create_session(client)["session_id"]
    path = f"/api/sessions/{session_id}/answer"

    non_wav = client.post(path, content=b"not wav", headers={"content-type": "audio/wav"})
    wrong_mime = client.post(
        path, content=_wav_bytes(), headers={"content-type": "application/octet-stream"}
    )
    stereo = client.post(
        path,
        content=_wav_bytes(channels=2),
        headers={"content-type": "audio/wav"},
    )
    unsupported_rate = client.post(
        path,
        content=_wav_bytes(sample_rate=7999),
        headers={"content-type": "audio/wav"},
    )
    eight_bit = client.post(
        path,
        content=_wav_bytes(sample_width=1),
        headers={"content-type": "audio/wav"},
    )
    oversized = client.post(
        path,
        content=b"",
        headers={
            "content-type": "audio/wav",
            "content-length": str(20 * 1024 * 1024 + 1),
        },
    )
    unknown = client.post(
        f"/api/sessions/{'a' * 32}/answer",
        content=_wav_bytes(),
        headers={"content-type": "audio/wav"},
    )
    non_wav_mime = client.post(
        path, content=b"nope", headers={"content-type": "text/plain"}
    )

    assert non_wav.status_code == 400
    assert wrong_mime.status_code == 415
    assert stereo.status_code == 400
    assert unsupported_rate.status_code == 400
    assert eight_bit.status_code == 400
    assert oversized.status_code == 413
    assert unknown.status_code == 404
    assert non_wav_mime.status_code == 415
    for response in (
        non_wav,
        wrong_mime,
        stereo,
        unsupported_rate,
        eight_bit,
        oversized,
        unknown,
        non_wav_mime,
    ):
        assert set(response.json()) == {"error", "message"}


def test_transcription_error_retries_without_advancing_interviewer(tmp_path):
    client, stt, _tts, created = _client(
        tmp_path,
        stt_responses=[RuntimeError("do not expose this secret"), _transcript()],
        actions=[("follow_up", "Could you explain further?")],
    )
    session_id = _create_session(client)["session_id"]
    failed = _answer(client, session_id)
    interviewer = created["interviewer"]

    assert failed["flags"]["transcription_error"] is True
    assert failed["interviewer"]["action"] == "repeat"
    assert "secret" not in failed["interviewer"]["text"]
    assert interviewer.answers == []
    retried = _answer(client, session_id)
    assert retried["interviewer"]["action"] == "follow_up"
    assert retried["interviewer"]["audio_url"] != failed["interviewer"]["audio_url"]
    assert interviewer.answers == ["A useful candidate answer."]
    assert stt.calls == 2


def test_finish_analysis_status_and_report_lifecycle(tmp_path):
    client, _stt, _tts, _created = _client(tmp_path)
    session_id = _create_session(client)["session_id"]
    before = client.get(f"/report/{session_id}")
    assert before.status_code == 404
    assert set(before.json()) == {"error", "message"}

    finished = client.post(
        f"/api/sessions/{session_id}/finish", json={"content": True}
    )
    assert finished.status_code == 200
    assert finished.json() == {"status": "analyzing"}
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{session_id}/status")
        if status.json().get("state") in {"ready", "error"}:
            break
        time.sleep(0.01)
    assert status.status_code == 200
    assert status.json()["state"] == "ready"
    assert status.json()["report_url"] == f"/report/{session_id}"
    report = client.get(f"/report/{session_id}")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/html")
    assert "Fake report" in report.text
    assert client.get(f"/report/{'../outside'}").status_code == 404
    analysis_kwargs = _created["analysis_kwargs"]
    assert isinstance(analysis_kwargs, dict)
    assert analysis_kwargs["llm"] is None


def test_finish_uses_content_analysis_only_when_configured(tmp_path):
    client, _stt, _tts, created = _client(tmp_path, llm_configured=True)
    session_id = _create_session(client)["session_id"]

    response = client.post(
        f"/api/sessions/{session_id}/finish", json={"content": True}
    )
    assert response.status_code == 200
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{session_id}/status")
        if status.json().get("state") in {"ready", "error"}:
            break
        time.sleep(0.01)
    assert status.json()["state"] == "ready"
    assert created["analysis_kwargs"]["llm"] is created["analysis_llm"]


def test_list_sessions_reports_existing_session_without_opening_audio(tmp_path):
    client, _stt, _tts, _created = _client(tmp_path)
    session_id = _create_session(client)["session_id"]

    response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json()["sessions"] == [
        {
            "id": session_id,
            "started": client.app.state.registry.get(session_id).session.created_at,
            "number_of_turns": 0,
            "report_available": False,
        }
    ]


def test_run_web_refuses_remote_binding_without_explicit_permission(capsys):
    result = run_web_main(["--host", "0.0.0.0"])

    captured = capsys.readouterr()
    assert result == 2
    assert "Refusing non-local host binding" in captured.err


def test_audio_session_and_report_path_traversal_are_not_served(tmp_path):
    client, _stt, _tts, _created = _client(tmp_path)
    session_id = _create_session(client)["session_id"]

    audio = client.get(f"/api/sessions/{session_id}/audio/%2e%2e%2fsecret.wav")
    report = client.get("/report/%2e%2e%2fsecret")

    assert audio.status_code == 404
    assert report.status_code == 404


def test_static_files_served_with_correct_mime_and_no_cache(tmp_path):
    client, _stt, _tts, _created = _client(tmp_path)

    # 1. GET / serves index.html with no-cache
    root = client.get("/")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert root.headers.get("cache-control") == "no-cache"
    assert "SpeakSure" in root.text

    # 2. GET /static/app.css serves text/css with no-cache
    css = client.get("/static/app.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
    assert css.headers.get("cache-control") == "no-cache"

    # 3. GET /static/js/main.js serves text/javascript with no-cache
    js = client.get("/static/js/main.js")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith("text/javascript")
    assert js.headers.get("cache-control") == "no-cache"

    audio_js = client.get("/static/js/interviewer-audio.js")
    assert audio_js.status_code == 200
    assert audio_js.headers["content-type"].startswith("text/javascript")

    # 4. GET /static/selftest.html serves text/html with no-cache
    selftest = client.get("/static/selftest.html")
    assert selftest.status_code == 200
    assert selftest.headers["content-type"].startswith("text/html")
    assert selftest.headers.get("cache-control") == "no-cache"

    # 5. Traversal and nonexistent files return 404
    assert client.get("/static/../app.py").status_code == 404
    assert client.get("/static/nonexistent.xyz").status_code == 404


def test_static_interface_has_no_review_or_learner_transcript_flow() -> None:
    """Keep the interview room voice-first and make captions interviewer-only."""
    static_dir = Path(__file__).resolve().parents[1] / "src" / "speaksure" / "web" / "static"
    index = (static_dir / "index.html").read_text(encoding="utf-8")
    main = (static_dir / "js" / "main.js").read_text(encoding="utf-8")
    forbidden = (
        "review-panel",
        "review-before-sending",
        "send-answer-btn",
        "transcript-card",
        "transcript-text",
    )
    assert all(term not in index and term not in main for term in forbidden)
    assert 'id="captions-toggle"' in index
    assert 'aria-pressed="false"' in index
    assert 'sandbox="allow-same-origin"' in index
    assert "app.currentLine" in main
    assert "result.transcript" not in main


def test_no_static_file_contains_external_urls():
    static_dir = Path(__file__).resolve().parents[1] / "src" / "speaksure" / "web" / "static"
    assert static_dir.is_dir()
    for file_path in static_dir.rglob("*"):
        if not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            # Disallow http:// and https:// except for localhost in comments
            if "http://" in line or "https://" in line:
                cleaned = line.strip()
                if "localhost" in cleaned and (cleaned.startswith("//") or cleaned.startswith("/*") or cleaned.startswith("#") or cleaned.startswith("<!--")):
                    continue
                assert False, f"Static file {file_path.name} contains external URL: {line}"


def test_vocabulary_hints_flow_and_validation(tmp_path):
    # Test global config + session hints combination
    client, stt, _tts, _created = _client(
        tmp_path,
        stt_config=LocalTranscriptionConfig(model_size="base", device="cpu", hotwords="GlobalWord"),
    )

    # 1. Session A with vocabulary_hints list
    res_a = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "candidate_name": "Student",
            "vocabulary_hints": ["SpeakSure", "FastAPI"],
            "client": {"input_device_label": "Mic"},
        },
    )
    assert res_a.status_code == 200
    session_a_id = res_a.json()["session_id"]
    entry_a = client.app.state.registry.get(session_a_id)
    assert entry_a.session.metadata["vocabulary_hints"] == ["SpeakSure", "FastAPI"]

    # 2. Session B without hints (absent hints change nothing)
    res_b = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "candidate_name": "Student",
            "client": {"input_device_label": "Mic"},
        },
    )
    assert res_b.status_code == 200
    session_b_id = res_b.json()["session_id"]
    entry_b = client.app.state.registry.get(session_b_id)
    assert "vocabulary_hints" not in entry_b.session.metadata

    # 3. Answer for Session A -> hotwords contains combined global and session hints
    _answer(client, session_a_id)
    hotwords_a = stt.last_kwargs.get("hotwords")
    assert hotwords_a == "GlobalWord, SpeakSure, FastAPI"

    # 4. Answer for Session B -> hotwords contains ONLY global hotwords (session A hints do not leak)
    _answer(client, session_b_id)
    hotwords_b = stt.last_kwargs.get("hotwords")
    assert hotwords_b == "GlobalWord"

    # 5. Comma-separated string format accepted
    res_c = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "vocabulary_hints": "Alpha, Beta, Gamma",
            "client": {"input_device_label": "Mic"},
        },
    )
    assert res_c.status_code == 200
    session_c_id = res_c.json()["session_id"]
    entry_c = client.app.state.registry.get(session_c_id)
    assert entry_c.session.metadata["vocabulary_hints"] == ["Alpha", "Beta", "Gamma"]

    _answer(client, session_c_id)
    assert stt.last_kwargs.get("hotwords") == "GlobalWord, Alpha, Beta, Gamma"

    # 6. Absent hints without global config -> hotwords is None
    client_no_global, stt_no_global, _, _ = _client(tmp_path / "no_global")
    res_d = client_no_global.post(
        "/api/sessions",
        json={
            "questions": 3,
            "client": {"input_device_label": "Mic"},
        },
    )
    assert res_d.status_code == 200
    _answer(client_no_global, res_d.json()["session_id"])
    assert stt_no_global.last_kwargs.get("hotwords") is None

    # 7. Validation failures: rejected with 422
    # Too many terms (> 40)
    invalid_too_many = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "vocabulary_hints": [f"term{i}" for i in range(41)],
            "client": {"input_device_label": "Mic"},
        },
    )
    assert invalid_too_many.status_code == 422
    assert set(invalid_too_many.json()) == {"error", "message"}

    # Term too long (> 40 chars)
    invalid_too_long = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "vocabulary_hints": ["a" * 41],
            "client": {"input_device_label": "Mic"},
        },
    )
    assert invalid_too_long.status_code == 422

    # Empty term in list
    invalid_empty = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "vocabulary_hints": ["valid", "   "],
            "client": {"input_device_label": "Mic"},
        },
    )
    assert invalid_empty.status_code == 422

    # Non-string element
    invalid_type = client.post(
        "/api/sessions",
        json={
            "questions": 3,
            "vocabulary_hints": [123],
            "client": {"input_device_label": "Mic"},
        },
    )
    assert invalid_type.status_code == 422
