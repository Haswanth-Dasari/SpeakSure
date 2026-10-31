"""FastAPI application factory for browser-captured local interviews."""

from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

import numpy as np
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, StrictInt, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..audio_capture import CapturedAnswer, wav_bytes
from ..interviewer import InterviewPlan, Interviewer
from ..interviewer_prompts import REPEAT_REQUEST
from ..llm_client import (
    GroqAnalysisLLMConfig,
    GroqLLMConfig,
    build_analysis_llm_client,
    build_llm_client,
)
from ..question_bank import bank_for_session, question_to_dict
from ..resume import (
    MAX_FILE_BYTES,
    ResumeError,
    empty_profile,
    extract_profile,
    extract_resume_text,
    merge_hints,
    resume_hotwords,
    validate_profile,
)
from ..interview_planner import built_in_ids, plan_realistic
from ..resume_questions import profile_background
from ..session_log import InterviewSession
from ..speech_to_text import (
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    SpeechToTextProvider,
    TranscriptionResult,
    describe_stt_configuration,
    load_report_model,
)
from ..text_to_speech import (
    SpeechSynthesisConfig,
    SynthesizedAudio,
    build_speech_synthesizer,
)
from ..analysis.session_analysis import analyze_session
from ..analysis.grammar import _load_pipeline
from ..configuration import get_report_stt_model
from ..report import render_report
from ..report.narrative import generate_narrative
from ..retranscribe import retranscribe_session
from .sessions import (
    WebSession,
    WebSessionRegistry,
    create_session_record,
    is_session_id,
)
from ..live_interview import (
    LiveSettings,
    TurnProcessingResult,
    process_interview_turn,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_AUDIO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STATIC_DIR = Path(__file__).parent / "static"
STATIC_MIME_TYPES: dict[str, str] = {
    ".js": "text/javascript",
    ".css": "text/css",
    ".html": "text/html",
    ".svg": "image/svg+xml",
}
LOGGER = logging.getLogger(__name__)
MODEL_COMPONENTS = ("stt", "tts", "spacy")
QUESTION_COUNTS = {"quick": 3, "standard": 5, "full": 8}


class TTSProvider(Protocol):
    """Minimal synthesis API used by the web audio route."""

    voice_name: str

    def synthesize(self, text: str) -> SynthesizedAudio:
        """Synthesize one spoken line."""
        ...


@dataclass(frozen=True)
class WebSettings:
    """Validated local model settings and directories used by the web app."""

    stt_config: LocalTranscriptionConfig
    tts_config: SpeechSynthesisConfig
    sessions_dir: Path
    analysis_dir: Path
    reports_dir: Path
    cache_dir: Path
    role: str = "software engineering fresher"

    @classmethod
    def from_env(cls) -> WebSettings:
        """Build settings from the project's existing environment configuration."""
        return cls(
            stt_config=LocalTranscriptionConfig.from_env(),
            tts_config=SpeechSynthesisConfig.from_env(),
            sessions_dir=Path("data/sessions"),
            analysis_dir=Path("data/analysis"),
            reports_dir=Path("data/reports"),
            cache_dir=Path("data/web_cache"),
        )


@dataclass
class WebServices:
    """Injectable collaborators and persistence locations for one web app."""

    stt_provider: SpeechToTextProvider
    tts_provider: TTSProvider
    interviewer_factory: Callable[[InterviewPlan], Interviewer]
    analysis_function: Callable[..., InterviewSession]
    report_function: Callable[[InterviewSession], str]
    clock: Callable[[], float]
    sessions_dir: Path
    analysis_dir: Path
    reports_dir: Path
    cache_dir: Path
    llm_configured: bool
    analysis_llm_factory: Callable[[], object] | None = None
    warmup_functions: dict[str, Callable[[], object]] | None = None
    report_stt_provider_factory: Callable[[], SpeechToTextProvider] | None = None
    report_stt_model: str = ""
    live_stt_model: str = ""


class ClientInfo(BaseModel):
    """Browser capture metadata supplied when a session is created."""

    input_device_label: str = Field(default="Unknown microphone", max_length=256)
    constraints: dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    """Validated input for a new interview session."""

    questions: StrictInt | None = Field(default=None, ge=3, le=10)
    role: str | None = Field(default=None, min_length=1, max_length=60)
    length: Literal["quick", "standard", "full"] = "standard"
    candidate_name: str | None = Field(default=None, max_length=120)
    vocabulary_hints: list[str] | str | None = Field(default=None)
    resume_profile: dict[str, Any] | None = Field(default=None)
    client: ClientInfo

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str | None) -> str | None:
        """Trim a role and reject a value that is blank after trimming."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("role must contain between 1 and 60 non-space characters.")
        return normalized

    @field_validator("resume_profile", mode="before")
    @classmethod
    def validate_resume_profile(cls, value: object) -> dict[str, Any] | None:
        """Normalize a user-reviewed resume profile; blank profiles become None."""
        if value is None:
            return None
        try:
            profile = validate_profile(value)
        except ResumeError as error:
            raise ValueError(str(error)) from None
        return profile if any(profile.values()) else None

    @field_validator("vocabulary_hints", mode="before")
    @classmethod
    def validate_vocabulary_hints(cls, value: object) -> list[str] | None:
        """Validate and normalize vocabulary hints (max 40 terms, 1-40 chars each)."""
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            raw_terms = [t.strip() for t in stripped.split(",")]
        elif isinstance(value, list):
            if not all(isinstance(x, str) for x in value):
                raise ValueError("All vocabulary hints must be strings.")
            raw_terms = [x.strip() for x in value]
        else:
            raise ValueError("vocabulary_hints must be a list of strings or comma-separated string.")

        if len(raw_terms) > 40:
            raise ValueError("vocabulary_hints cannot exceed 40 terms.")
        for term in raw_terms:
            if not (1 <= len(term) <= 40):
                raise ValueError("Each vocabulary hint must be between 1 and 40 characters after trimming.")
        return raw_terms


def create_app(
    settings: WebSettings | None = None,
    services: WebServices | None = None,
) -> FastAPI:
    """Create a local interview API with replaceable services for offline tests."""
    active_settings = settings or WebSettings.from_env()
    active_services = services or _default_services(active_settings)
    warmups = active_services.warmup_functions
    initial_model_components = {
        name: ("loading" if warmups is not None and name in warmups else "ready")
        for name in MODEL_COMPONENTS
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Start background model warmup and close the analysis worker pool."""
        def warm_models() -> None:
            """Load each model independently and publish component readiness."""
            for name in MODEL_COMPONENTS:
                warmup = (warmups or {}).get(name)
                if warmup is None:
                    _app.state.model_components[name] = "ready"
                    continue
                try:
                    warmup()
                except Exception:
                    LOGGER.exception("Could not preload the %s model.", name)
                    _app.state.model_components[name] = "error"
                else:
                    _app.state.model_components[name] = "ready"

        if warmups:
            _app.state.model_warmup_thread = threading.Thread(
                target=warm_models,
                name="speaksure-model-warmup",
                daemon=True,
            )
            _app.state.model_warmup_thread.start()
        yield
        _app.state.analysis_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="SpeakSure local interview API", lifespan=lifespan)
    registry = WebSessionRegistry()
    app.state.registry = registry
    app.state.settings = active_settings
    app.state.services = active_services
    app.state.analysis_pool = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="speaksure-analysis"
    )
    app.state.model_components = initial_model_components

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        """Return validation failures in the API's stable error envelope."""
        return _error_response(
            422, "invalid_request", "The request body does not match the API contract."
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        request: Request, error: HTTPException
    ) -> JSONResponse:
        """Format framework route errors with the same stable JSON envelope."""
        del request
        return _error_response(
            error.status_code, "not_found" if error.status_code == 404 else "http_error",
            "The requested resource was not found."
            if error.status_code == 404
            else "The request could not be completed.",
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_error_handler(
        request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        """Format unmatched route and method errors in the stable envelope."""
        del request
        return _error_response(
            error.status_code,
            "not_found" if error.status_code == 404 else "http_error",
            "The requested resource was not found."
            if error.status_code == 404
            else "The request could not be completed.",
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        _request: Request, _error: Exception
    ) -> JSONResponse:
        """Hide implementation details while returning an explicit server error."""
        return _error_response(
            500, "internal_error", "The request could not be completed."
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        """Serve the frontend entry point with no-cache header."""
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return _error_response(404, "not_found", "The requested resource was not found.")
        return HTMLResponse(
            index_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/static/{file_path:path}")
    def serve_static(file_path: str) -> Response:
        """Serve static assets with exact MIME types and no-cache header."""
        target_path = (STATIC_DIR / file_path).resolve()
        static_root = STATIC_DIR.resolve()
        if static_root not in target_path.parents or not target_path.is_file():
            return _error_response(404, "not_found", "The requested resource was not found.")
        mime_type = STATIC_MIME_TYPES.get(target_path.suffix.lower(), "application/octet-stream")
        return Response(
            content=target_path.read_bytes(),
            media_type=mime_type,
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Return model and voice configuration without disclosing credentials."""
        description = describe_stt_configuration(active_settings.stt_config)
        component_states = dict(app.state.model_components)
        if (
            warmups is None
            and getattr(active_services.stt_provider, "actual_device", "not-initialized")
            == "not-initialized"
        ):
            component_states["stt"] = "loading"
        components = [
            {"name": name, "status": component_states[name]}
            for name in MODEL_COMPONENTS
        ]
        model_loaded = all(
            component["status"] == "ready" for component in components
        )
        return {
            "status": "ok" if model_loaded else "loading",
            "components": components,
            "stt_model": description["model"],
            "stt_device_setting": description["device_setting"],
            "tts_voice": active_settings.tts_config.voice_name,
            "llm_configured": active_services.llm_configured,
        }

    @app.post("/api/resume")
    async def read_resume(
        request: Request, x_filename: str = Header(default="resume.txt")
    ) -> Response:
        """Read an uploaded resume locally and return an editable candidate profile."""
        data = await request.body()
        if len(data) > MAX_FILE_BYTES:
            return _error_response(413, "resume_too_large", "The file is larger than 2 MB.")
        return await run_in_threadpool(_profile_from_upload, active_services, data, x_filename)

    @app.post("/api/sessions")
    def create_session(payload: CreateSessionRequest) -> dict[str, object]:
        """Create and persist an interview session and expose its opening line."""
        question_count = (
            payload.questions
            if payload.questions is not None
            else QUESTION_COUNTS[payload.length]
        )
        role = payload.role or active_settings.role
        resume_profile = payload.resume_profile
        planning_llm = None
        if resume_profile and active_services.analysis_llm_factory is not None:
            try:
                planning_llm = active_services.analysis_llm_factory()
            except Exception:
                LOGGER.warning("Resume question planning will use templates only.")
        questions = plan_realistic(question_count, resume_profile, planning_llm)
        plan = InterviewPlan(
            questions=questions,
            role=role,
            candidate_name=(payload.candidate_name or "").strip(),
            background=profile_background(resume_profile) if resume_profile else "",
            followup_on_last_question=bool(resume_profile),
            natural_style=True,
            deep_dive_followups=2 if resume_profile else 1,
        )
        interviewer = active_services.interviewer_factory(plan)
        opening = interviewer.opening()
        metadata: dict[str, object] = {
            "stt_model": active_settings.stt_config.model_size,
            "stt_device": "not-initialized",
            "tts_voice": active_settings.tts_config.voice_name,
            "role": role,
            "length": payload.length,
            "capture_source": "browser",
            "input_device": payload.client.input_device_label,
            "capture_constraints": payload.client.constraints,
        }
        session_hints = list(payload.vocabulary_hints or [])
        if resume_profile:
            # Distinctive resume terms (projects, tools, institutions) help the
            # speech model spell them correctly, both live and in the report pass.
            session_hints = merge_hints(session_hints, resume_hotwords(resume_profile))
        if session_hints:
            metadata["vocabulary_hints"] = session_hints
        custom_questions = [
            question_to_dict(question)
            for question in questions
            if question.id not in built_in_ids()
        ]
        if custom_questions:
            metadata["custom_questions"] = custom_questions
        if resume_profile and any(q["id"].startswith("resume_") for q in custom_questions):
            metadata["resume_used"] = True
            metadata["resume_profile"] = resume_profile
        session = create_session_record(question_count, metadata)
        LOGGER.info("Creating session %s: synthesizing opening line.", session.session_id)
        try:
            _cache_spoken_line(
                active_services,
                session.session_id,
                "opening",
                opening.utterance,
            )
        except Exception:
            LOGGER.exception("Could not synthesize the interview opening.")
            return _error_response(
                502, "tts_failed", "The opening interviewer audio could not be generated."
            )
        LOGGER.info("Session %s: opening audio ready.", session.session_id)
        session_path = active_services.sessions_dir / f"session_{session.session_id}.json"
        session.save_to(session_path)
        entry = WebSession(
            session=session,
            interviewer=interviewer,
            current_question=questions[0].text,
            current_question_id=questions[0].id,
            question_index=1,
            total_main_questions=len(questions),
            session_path=session_path,
            spoken_lines={"opening": opening.utterance},
        )
        registry.add(entry)
        return {
            "session_id": session.session_id,
            "first_question": {
                "id": questions[0].id,
                "text": opening.utterance,
                "audio_url": _audio_url(session.session_id, "opening"),
            },
            "question_index": 1,
            "total_main_questions": len(questions),
        }

    @app.get("/api/sessions/{session_id}/audio/{name}.wav")
    def spoken_audio(session_id: str, name: str) -> Response:
        """Return a previously synthesized interviewer line as a WAV file."""
        entry = _get_session(registry, session_id)
        if entry is None:
            return _error_response(404, "session_not_found", "Session was not found.")
        if _AUDIO_NAME_PATTERN.fullmatch(name) is None:
            return _error_response(404, "audio_not_found", "Audio file was not found.")
        if name not in entry.spoken_lines:
            return _error_response(404, "audio_not_found", "Audio file was not found.")
        cache_path = active_services.cache_dir / session_id / f"{name}.wav"
        if not cache_path.is_file():
            return _error_response(
                404, "audio_not_ready", "The interviewer audio is not available."
            )
        return Response(cache_path.read_bytes(), media_type="audio/wav")

    @app.post("/api/sessions/{session_id}/answer")
    async def answer(session_id: str, request: Request) -> JSONResponse:
        """Accept and validate one raw WAV upload, then process it off-loop."""
        entry = _get_session(registry, session_id)
        if entry is None:
            return _error_response(404, "session_not_found", "Session was not found.")
        if entry.state != "active":
            return _error_response(409, "session_not_active", "Session is not active.")
        if getattr(entry.interviewer, "is_finished", False):
            return _error_response(409, "session_not_active", "Interview has already finished.")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() not in {
            "audio/wav",
            "audio/x-wav",
        }:
            return _error_response(
                415, "unsupported_media_type", "Upload the recording as audio/wav."
            )
        try:
            audio_bytes = await _read_limited_body(request)
            parsed = await run_in_threadpool(_parse_pcm16_mono_wav, audio_bytes)
        except UploadError as error:
            return _error_response(error.status_code, error.code, error.message)

        try:
            result = await run_in_threadpool(
                _process_web_answer,
                entry,
                parsed,
                audio_bytes,
                active_settings,
                active_services,
            )
        except ValueError:
            return _error_response(
                409, "session_not_active", "Session is no longer accepting answers."
            )
        except AudioSynthesisError as error:
            return _error_response(502, "reply_audio_failed", str(error))
        except Exception:
            return _error_response(
                500, "answer_processing_failed", "The answer could not be processed."
            )
        return JSONResponse(result)

    @app.post("/api/sessions/{session_id}/finish")
    async def finish(session_id: str, request: Request) -> JSONResponse:
        """Mark a session complete and start its report analysis in a worker."""
        entry = _get_session(registry, session_id)
        if entry is None:
            return _error_response(404, "session_not_found", "Session was not found.")
        try:
            body = await request.body()
            if len(body) > 4096:
                return _error_response(413, "request_too_large", "Finish request is too large.")
            content_requested = _content_requested(body)
        except ValueError:
            return _error_response(400, "invalid_request", "Finish request must be JSON.")
        with entry.lock:
            if entry.state in {"analyzing", "ready"}:
                return JSONResponse({"status": entry.state})
            entry.content_requested = content_requested
            entry.session.metadata["completed"] = True
            entry.session.metadata["completed_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            entry.session.save_to(entry.session_path)
            entry.state = "analyzing"
            entry.message = "Analysis is running."
            entry.stage = None
            entry.progress = None
        app.state.analysis_pool.submit(
            _run_analysis, entry, active_settings, active_services
        )
        return JSONResponse({"status": "analyzing"})

    @app.get("/api/sessions/{session_id}/status")
    def session_status(session_id: str) -> JSONResponse:
        """Return the current lifecycle state and a report URL when ready."""
        if not is_session_id(session_id):
            return _error_response(404, "session_not_found", "Session was not found.")
        entry = registry.get(session_id)
        if entry is None:
            report_path = active_services.reports_dir / f"{session_id}.html"
            if report_path.is_file():
                return JSONResponse(
                    {
                        "state": "ready",
                        "message": "Report is ready.",
                        "report_url": f"/report/{session_id}",
                    }
                )
            return _error_response(404, "session_not_found", "Session was not found.")
        response: dict[str, object] = {
            "state": entry.state,
            "message": entry.message,
        }
        if entry.stage is not None:
            response["stage"] = entry.stage
        if entry.progress is not None:
            response["progress"] = entry.progress
        if entry.state == "ready":
            response["report_url"] = f"/report/{session_id}"
        return JSONResponse(response)

    @app.patch("/api/sessions/{session_id}/turns/{turn_index}/timing")
    def update_turn_timing(
        session_id: str,
        turn_index: int,
        request: Request,
    ) -> JSONResponse:
        """Persist browser-measured upload and answer-start timing for one turn."""
        entry = _get_session(registry, session_id)
        if entry is None:
            return _error_response(404, "session_not_found", "Session was not found.")
        upload_ms = _timing_header(request.headers.get("x-client-upload-ms"))
        end_wait_ms = _timing_header(request.headers.get("x-client-end-wait-ms"))
        if upload_ms is None or end_wait_ms is None:
            return _error_response(
                400, "invalid_timing", "Client timing headers must be finite non-negative milliseconds."
            )
        with entry.lock:
            turn = next(
                (item for item in entry.session.turns if item.index == turn_index),
                None,
            )
            if turn is None:
                return _error_response(404, "turn_not_found", "Turn was not found.")
            timing = turn.analysis.setdefault("timing", {})
            timing["upload_s"] = upload_ms / 1000
            timing["end_wait_s"] = end_wait_ms / 1000
            # The browser measures the time from the end of the question to the
            # first spoken word. That is the thinking time; 0 means no speech.
            if end_wait_ms > 0:
                turn.thinking_time_seconds = end_wait_ms / 1000
            entry.session.save_to(entry.session_path)
        return JSONResponse({"status": "ok"})

    @app.get("/report/{session_id}", response_class=HTMLResponse)
    def report(session_id: str) -> Response:
        """Serve a generated standalone HTML report after validation."""
        if not is_session_id(session_id):
            return _error_response(404, "report_not_found", "Report was not found.")
        entry = registry.get(session_id)
        report_path = active_services.reports_dir / f"{session_id}.html"
        if entry is not None and entry.state != "ready":
            return _error_response(404, "report_not_found", "Report is not ready.")
        if not report_path.is_file():
            return _error_response(404, "report_not_found", "Report was not found.")
        try:
            return HTMLResponse(report_path.read_text(encoding="utf-8"))
        except OSError:
            return _error_response(500, "report_unavailable", "Report could not be read.")

    @app.get("/api/sessions")
    def list_sessions() -> JSONResponse:
        """List stored session metadata without reading referenced audio files."""
        try:
            sessions = _list_saved_sessions(active_services)
        except (OSError, ValueError, json.JSONDecodeError):
            return _error_response(
                500, "sessions_unavailable", "Saved sessions could not be listed."
            )
        return JSONResponse({"sessions": sessions})

    return app


def _default_services(settings: WebSettings) -> WebServices:
    """Construct production STT, TTS, interviewer, analysis, and report services."""
    stt_provider = LocalSpeechToTextProvider(settings.stt_config)
    tts_provider = build_speech_synthesizer(settings.tts_config)

    def interviewer_factory(plan: InterviewPlan) -> Interviewer:
        """Build the existing interviewer with configured Groq or safe fallback."""
        return Interviewer(build_llm_client(GroqLLMConfig.from_env()), plan)

    def report_function(analyzed: InterviewSession) -> str:
        """Render an analyzed session using the existing standalone report."""
        return render_report(
            analyzed.to_dict(),
            bank_for_session(analyzed.metadata),
            datetime.now(timezone.utc).isoformat(),
        )

    def warmup_tts() -> None:
        """Load the selected synthesis voice when the provider supports warmup."""
        warmup = getattr(tts_provider, "warmup", None)
        if callable(warmup):
            warmup()

    live_model = settings.stt_config.model_size
    report_model = get_report_stt_model()
    report_factory: Callable[[], SpeechToTextProvider] | None = None
    if report_model in {"base", "small", "medium", "large-v3"} and report_model != live_model:
        report_config = replace(settings.stt_config, model_size=report_model)

        def report_factory() -> SpeechToTextProvider:
            """Build the provider for the slower, more accurate report pass."""
            return LocalSpeechToTextProvider(
                report_config, model_loader=load_report_model
            )

    return WebServices(
        report_stt_provider_factory=report_factory,
        report_stt_model=report_model,
        live_stt_model=live_model,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        interviewer_factory=interviewer_factory,
        analysis_function=analyze_session,
        report_function=report_function,
        clock=time.perf_counter,
        sessions_dir=settings.sessions_dir,
        analysis_dir=settings.analysis_dir,
        reports_dir=settings.reports_dir,
        cache_dir=settings.cache_dir,
        llm_configured=bool(GroqLLMConfig.from_env().api_key),
        analysis_llm_factory=lambda: build_analysis_llm_client(
            GroqAnalysisLLMConfig.from_env()
        ),
        warmup_functions={
            "stt": stt_provider.warmup,
            "tts": warmup_tts,
            "spacy": _load_pipeline,
        },
    )


def _get_session(
    registry: WebSessionRegistry, session_id: str
) -> WebSession | None:
    """Look up only syntactically valid session identifiers."""
    if not is_session_id(session_id):
        return None
    return registry.get(session_id)


def _audio_url(session_id: str, name: str) -> str:
    """Build the public route for one generated interviewer line."""
    return f"/api/sessions/{session_id}/audio/{name}.wav"


def _cache_spoken_line(
    services: WebServices,
    session_id: str,
    name: str,
    text: str,
) -> float:
    """Synthesize and cache a spoken line, returning its synthesis duration."""
    cache_path = services.cache_dir / session_id / f"{name}.wav"
    if cache_path.is_file():
        return 0.0
    audio = services.tts_provider.synthesize(text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(wav_bytes(audio.samples, audio.sample_rate))
    return audio.synthesis_seconds


def _timing_header(value: str | None) -> float | None:
    """Parse a finite, non-negative browser timing value in milliseconds."""
    if value is None:
        return None
    try:
        milliseconds = float(value)
    except ValueError:
        return None
    if not np.isfinite(milliseconds) or not 0 <= milliseconds <= 600_000:
        return None
    return milliseconds


def _profile_from_upload(services: WebServices, data: bytes, filename: str) -> JSONResponse:
    """Extract text, redact contact details, and ask the model for a verified profile."""
    try:
        text = extract_resume_text(data, filename)
    except ResumeError as error:
        return _error_response(422, "resume_unreadable", str(error))
    if services.analysis_llm_factory is None or not services.llm_configured:
        return JSONResponse(
            {
                "profile": empty_profile(),
                "info": {},
                "warning": "No language-model key is configured, so the resume could not "
                "be analysed. You can type your details in by hand.",
            }
        )
    try:
        profile, info = extract_profile(text, services.analysis_llm_factory())  # type: ignore[arg-type]
    except ResumeError as error:
        return JSONResponse(
            {
                "profile": empty_profile(),
                "info": {},
                "warning": f"{error} You can type your details in by hand.",
            }
        )
    except Exception:
        LOGGER.exception("Resume analysis failed.")
        return _error_response(502, "resume_failed", "The resume could not be analysed right now.")
    return JSONResponse({"profile": profile, "info": info, "warning": ""})


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    """Create the stable JSON error envelope used by every endpoint."""
    return JSONResponse(
        status_code=status,
        content={"error": code, "message": message},
    )


class UploadError(ValueError):
    """Represent a client audio-upload failure with its HTTP error response."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        """Store a safe client-facing status and message."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class AudioSynthesisError(RuntimeError):
    """Represent a reply that could not be synthesized and cached."""


async def _read_limited_body(request: Request) -> bytes:
    """Read a raw request incrementally, enforcing its maximum size."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            raise UploadError(400, "invalid_request", "Content-Length is invalid.") from None
        if declared_length < 0:
            raise UploadError(400, "invalid_request", "Content-Length is invalid.")
        if declared_length > MAX_UPLOAD_BYTES:
            raise UploadError(413, "audio_too_large", "Audio upload exceeds 20 MB.")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise UploadError(413, "audio_too_large", "Audio upload exceeds 20 MB.")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_pcm16_mono_wav(audio_bytes: bytes) -> tuple[np.ndarray, int, float]:
    """Verify the WAV container and return normalized mono samples and duration."""
    if len(audio_bytes) < 12 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
        raise UploadError(400, "invalid_wav", "Upload a valid RIFF/WAVE file.")
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            if (
                wav_file.getcomptype() != "NONE"
                or wav_file.getsampwidth() != 2
                or wav_file.getnchannels() != 1
                or not 8000 <= sample_rate <= 48000
            ):
                raise UploadError(
                    400,
                    "unsupported_wav_format",
                    "Audio must be PCM 16-bit mono at 8000–48000 Hz.",
                )
            raw_samples = wav_file.readframes(wav_file.getnframes())
    except wave.Error:
        raise UploadError(400, "invalid_wav", "Upload a valid RIFF/WAVE file.") from None
    if not raw_samples or len(raw_samples) % 2:
        raise UploadError(400, "invalid_wav", "The WAV file contains no complete audio samples.")
    samples = np.frombuffer(raw_samples, dtype="<i2").astype(np.float32) / 32768.0
    duration_seconds = len(samples) / sample_rate
    return samples, sample_rate, duration_seconds


def _combine_hotwords(global_hotwords: str, session_hints: list[str] | None) -> str | None:
    """Combine global transcription hotwords with session-specific vocabulary hints."""
    parts: list[str] = []
    if global_hotwords and global_hotwords.strip():
        for item in global_hotwords.split(","):
            cleaned = item.strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
    if session_hints:
        for hint in session_hints:
            cleaned = hint.strip()
            if cleaned and cleaned not in parts:
                parts.append(cleaned)
    return ", ".join(parts) if parts else None


def _process_web_answer(
    entry: WebSession,
    parsed_wav: tuple[np.ndarray, int, float],
    audio_bytes: bytes,
    settings: WebSettings,
    services: WebServices,
) -> dict[str, object]:
    """Process an uploaded answer with the same shared turn function as CLI."""
    with entry.lock:
        if entry.state != "active":
            raise ValueError("Session is not active.")
        if getattr(entry.interviewer, "is_finished", False):
            raise ValueError("Interview has already finished.")
        samples, sample_rate, duration_seconds = parsed_wav
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
        captured = CapturedAnswer(
            samples=samples,
            sample_rate=sample_rate,
            thinking_time_seconds=0.0,
            answer_duration_seconds=duration_seconds,
            stop_reason="no_speech" if rms < 0.008 else "source_ended",
            noise_rms=0.0,
            threshold=0.008,
        )
        turn_index = len(entry.session.turns) + 1
        audio_path = services.sessions_dir / entry.session.session_id / f"turn_{turn_index:02d}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(audio_bytes)

        session_hints = entry.session.metadata.get("vocabulary_hints")
        global_hotwords = getattr(settings.stt_config, "hotwords", "")
        combined_hotwords = _combine_hotwords(global_hotwords, session_hints)

        def transcribe(_captured: CapturedAnswer) -> TranscriptionResult:
            """Transcribe the validated, saved browser WAV using the configured provider."""
            kwargs: dict[str, Any] = {
                "filename": audio_path.name,
                "content_type": "audio/wav",
            }
            if combined_hotwords is not None:
                kwargs["hotwords"] = combined_hotwords
            entry.stage = "transcribing"
            entry.progress = None
            try:
                return services.stt_provider.transcribe_detailed(
                    audio_path.read_bytes(),
                    **kwargs,
                )
            finally:
                entry.stage = None
                entry.progress = None

        processed: TurnProcessingResult = process_interview_turn(
            interviewer=entry.interviewer,
            session=entry.session,
            settings=LiveSettings(sessions_dir=services.sessions_dir),
            captured=captured,
            current_question=entry.current_question,
            current_question_id=entry.current_question_id,
            current_is_follow_up=entry.current_is_follow_up,
            transcribe=transcribe,
            speak=None,
            clock=services.clock,
            output=lambda _message: None,
            empty_attempts=entry.empty_attempts,
            listen_attempts=entry.listen_attempts,
            transcription_errors=tuple(entry.transcription_errors),
            audio_path_override=audio_path,
            safe_transcription_errors=True,
        )
        if processed.turn is not None:
            timing = processed.turn.analysis["timing"]
            timing.update(
                {
                    "end_wait_s": None,
                    "upload_s": None,
                    "stt_s": timing.get("stt_seconds", 0.0),
                    "llm_s": timing.get("llm_seconds"),
                    "tts_s": None,
                }
            )
        entry.empty_attempts = processed.empty_attempts
        entry.listen_attempts = processed.listen_attempts
        entry.transcription_errors = list(processed.transcription_errors)
        reply = processed.reply
        if processed.turn is not None:
            entry.session.metadata["stt_device"] = getattr(
                services.stt_provider, "actual_device", "unknown"
            )
            entry.session.save_to(entry.session_path)
            if reply is not None and reply.kind in {"follow_up", "clarify"}:
                entry.current_question = reply.utterance
                entry.current_is_follow_up = True
            elif reply is not None and reply.kind == "next_question":
                entry.current_question = reply.utterance
                entry.current_question_id = reply.question_id or entry.current_question_id
                entry.question_index = min(
                    entry.question_index + 1, entry.total_main_questions
                )
                entry.current_is_follow_up = False
            entry.empty_attempts = 0
            entry.listen_attempts = 0
            entry.transcription_errors = []

        reply_text = (
            REPEAT_REQUEST_TEXT
            if processed.transcription_error
            else reply.utterance if reply is not None else REPEAT_REQUEST_TEXT
        )
        action = (
            "repeat"
            if processed.transcription_error
            else _web_action(reply.kind) if reply is not None else "repeat"
        )
        entry.spoken_line_count += 1
        line_name = f"turn_{turn_index:02d}_{entry.spoken_line_count:04d}"
        entry.spoken_lines[line_name] = reply_text
        try:
            synthesis_seconds = _cache_spoken_line(
                services,
                entry.session.session_id,
                line_name,
                reply_text,
            )
        except Exception as error:
            LOGGER.exception("Could not synthesize interviewer response.")
            raise AudioSynthesisError(
                "The interviewer response audio could not be generated."
            ) from error
        if processed.turn is not None:
            timing = processed.turn.analysis["timing"]
            timing["tts_s"] = synthesis_seconds
            timing["tts_synthesis_seconds"] = synthesis_seconds
            entry.session.save_to(entry.session_path)
        transcript = (
            processed.transcription.text if processed.transcription is not None else ""
        )
        flags = _answer_flags(processed, rms)
        return {
            "turn_index": turn_index,
            "transcript": transcript,
            "words_count": (
                len(processed.transcription.words)
                if processed.transcription is not None
                else 0
            ),
            "flags": flags,
            "interviewer": {
                "action": action,
                "text": reply_text,
                "audio_url": _audio_url(entry.session.session_id, line_name),
            },
            "finished": bool(getattr(entry.interviewer, "is_finished", False)),
            "turn_recorded": processed.turn is not None,
        }


REPEAT_REQUEST_TEXT = REPEAT_REQUEST


def _web_action(kind: str) -> str:
    """Map existing interviewer kinds to the public web action vocabulary."""
    if kind == "closing":
        return "finish"
    if kind == "clarify":
        return "clarify"
    if kind in {"next_question", "follow_up", "repeat"}:
        return kind
    return "next_question"


def _answer_flags(processed: TurnProcessingResult, rms: float) -> dict[str, object]:
    """Expose safe transcript and audio-quality flags for one upload."""
    delivery = (
        processed.turn.analysis.get("audio_delivery", {})
        if processed.turn is not None
        else {}
    )
    quality_flags = delivery.get("quality_flags", [])
    if not isinstance(quality_flags, list):
        quality_flags = []
    transcript = processed.transcription
    return {
        "too_quiet": "too_quiet" in quality_flags or rms < 0.008,
        "very_short": "very_short" in quality_flags
        or (transcript is not None and len(transcript.text.split()) < 2),
        "clipped": "clipped" in quality_flags,
        "too_few_words": (
            transcript is not None and len(transcript.text.split()) < 2
        ),
        "transcription_error": processed.transcription_error,
        "low_confidence_words": (
            [
                word.text
                for word in transcript.words
                if word.probability < 0.5
            ]
            if transcript is not None
            else []
        ),
    }


def _content_requested(body: bytes) -> bool:
    """Parse the optional finish request and validate its content flag."""
    if not body.strip():
        return False
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise ValueError("Finish request must be valid JSON.") from None
    if not isinstance(data, dict) or (
        "content" in data and not isinstance(data["content"], bool)
    ):
        raise ValueError("Finish request content must be a boolean.")
    return bool(data.get("content", False))


def _run_analysis(
    entry: WebSession, settings: WebSettings, services: WebServices
) -> None:
    """Run whole-session analysis and report rendering outside request threads."""
    try:
        analysis_llm = None
        if entry.content_requested and services.llm_configured:
            if services.analysis_llm_factory is not None:
                analysis_llm = services.analysis_llm_factory()
            else:
                analysis_llm = build_analysis_llm_client(
                    GroqAnalysisLLMConfig.from_env()
                )
        role = str(
            entry.session.metadata.get("role", settings.role)
        )

        def update_progress(stage: str, progress: float) -> None:
            """Publish one known analysis stage and its normalized progress."""
            if stage not in {"transcribing", "delivery", "grammar", "content", "report"}:
                return
            with entry.lock:
                entry.stage = stage
                entry.progress = min(1.0, max(0.0, float(progress)))
                entry.message = f"Preparing feedback: {stage}."

        session_for_analysis = entry.session
        if services.report_stt_provider_factory is not None:
            try:
                update_progress("transcribing", 0.0)
                session_for_analysis = retranscribe_session(
                    entry.session,
                    services.report_stt_provider_factory(),
                    report_model=services.report_stt_model,
                    live_model=services.live_stt_model,
                    hotwords=_combine_hotwords(
                        getattr(settings.stt_config, "hotwords", ""),
                        entry.session.metadata.get("vocabulary_hints"),
                    ),
                    audio_root=Path("."),
                    progress_callback=update_progress,
                )
            except Exception:
                LOGGER.exception(
                    "Report transcription pass failed; using live transcripts."
                )
                session_for_analysis = entry.session

        analyzed = services.analysis_function(
            session_for_analysis,
            audio_root=Path("."),
            llm=analysis_llm,
            role=role,
            progress_callback=update_progress,
        )
        if analysis_llm is not None:
            update_progress("report", 0.0)
            try:
                narrative = generate_narrative(
                    analyzed.to_dict(), bank_for_session(analyzed.metadata), analysis_llm
                )
            except Exception:
                LOGGER.exception("The written review failed; the report will omit it.")
                narrative = None
            if narrative:
                analyzed.analysis["narrative"] = narrative
        analysis_path = services.analysis_dir / f"{entry.session.session_id}.analyzed.json"
        analyzed.save_to(analysis_path)
        update_progress("report", 0.0)
        report_text = services.report_function(analyzed)
        report_path = services.reports_dir / f"{entry.session.session_id}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        with entry.lock:
            entry.report_path = report_path
            entry.state = "ready"
            entry.message = "Report is ready."
            entry.stage = "report"
            entry.progress = 1.0
    except Exception:
        with entry.lock:
            entry.state = "error"
            entry.message = "Analysis failed. Review the session and try again."


def _list_saved_sessions(services: WebServices) -> list[dict[str, object]]:
    """Read persisted session JSON metadata without opening audio paths."""
    sessions: list[dict[str, object]] = []
    for session_path in sorted(services.sessions_dir.glob("*.json")):
        data = json.loads(session_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("turns"), list):
            continue
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not is_session_id(session_id):
            continue
        sessions.append(
            {
                "id": session_id,
                "started": data.get("created_at"),
                "number_of_turns": len(data["turns"]),
                "report_available": (
                    services.reports_dir / f"{session_id}.html"
                ).is_file(),
            }
        )
    return sessions
