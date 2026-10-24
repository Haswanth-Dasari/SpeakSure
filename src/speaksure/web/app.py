"""FastAPI application factory for browser-captured local interviews."""

from __future__ import annotations

import io
import json
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, StrictInt
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
from ..question_bank import HR_QUESTIONS, select_questions
from ..session_log import InterviewSession
from ..speech_to_text import (
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    SpeechToTextProvider,
    TranscriptionResult,
    describe_stt_configuration,
)
from ..text_to_speech import (
    SpeechSynthesisConfig,
    SynthesizedAudio,
    build_speech_synthesizer,
)
from ..analysis.session_analysis import analyze_session
from ..report import render_report
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


class ClientInfo(BaseModel):
    """Browser capture metadata supplied when a session is created."""

    input_device_label: str = Field(default="Unknown microphone", max_length=256)
    constraints: dict[str, Any] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
    """Validated input for a new interview session."""

    questions: StrictInt = Field(ge=3, le=10)
    candidate_name: str | None = Field(default=None, max_length=120)
    client: ClientInfo


def create_app(
    settings: WebSettings | None = None,
    services: WebServices | None = None,
) -> FastAPI:
    """Create a local interview API with replaceable services for offline tests."""
    active_settings = settings or WebSettings.from_env()
    active_services = services or _default_services(active_settings)
    app = FastAPI(title="SpeakSure local interview API")
    registry = WebSessionRegistry()
    app.state.registry = registry
    app.state.settings = active_settings
    app.state.services = active_services
    app.state.analysis_pool = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="speaksure-analysis"
    )

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
    def pending_frontend() -> HTMLResponse:
        """Serve the placeholder until the browser interface is implemented."""
        index_path = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Return model and voice configuration without disclosing credentials."""
        description = describe_stt_configuration(active_settings.stt_config)
        model_loaded = (
            getattr(active_services.stt_provider, "actual_device", "not-initialized")
            != "not-initialized"
        )
        return {
            "status": "ok" if model_loaded else "loading",
            "stt_model": description["model"],
            "stt_device_setting": description["device_setting"],
            "tts_voice": active_settings.tts_config.voice_name,
            "llm_configured": active_services.llm_configured,
        }

    @app.post("/api/sessions")
    def create_session(payload: CreateSessionRequest) -> dict[str, object]:
        """Create and persist an interview session and expose its opening line."""
        questions = select_questions(payload.questions)
        plan = InterviewPlan(
            questions=questions,
            role=active_settings.role,
            candidate_name=(payload.candidate_name or "").strip(),
        )
        interviewer = active_services.interviewer_factory(plan)
        opening = interviewer.opening()
        metadata: dict[str, object] = {
            "stt_model": active_settings.stt_config.model_size,
            "stt_device": "not-initialized",
            "tts_voice": active_settings.tts_config.voice_name,
            "role": active_settings.role,
            "capture_source": "browser",
            "input_device": payload.client.input_device_label,
            "capture_constraints": payload.client.constraints,
        }
        session = create_session_record(payload.questions, metadata)
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
        """Return a cached or newly synthesized spoken line as a WAV file."""
        entry = _get_session(registry, session_id)
        if entry is None:
            return _error_response(404, "session_not_found", "Session was not found.")
        if _AUDIO_NAME_PATTERN.fullmatch(name) is None:
            return _error_response(404, "audio_not_found", "Audio file was not found.")
        text = entry.spoken_lines.get(name)
        if text is None:
            return _error_response(404, "audio_not_found", "Audio file was not found.")
        cache_path = active_services.cache_dir / session_id / f"{name}.wav"
        try:
            if not cache_path.is_file():
                audio = active_services.tts_provider.synthesize(text)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(wav_bytes(audio.samples, audio.sample_rate))
            return Response(cache_path.read_bytes(), media_type="audio/wav")
        except Exception:
            return _error_response(
                502, "tts_failed", "The interviewer audio could not be generated."
            )

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
            if entry.state == "error":
                return _error_response(409, "analysis_failed", entry.message)
            entry.content_requested = content_requested
            entry.session.metadata["completed"] = True
            entry.session.metadata["completed_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            entry.session.save_to(entry.session_path)
            entry.state = "analyzing"
            entry.message = "Analysis is running."
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
        if entry.state == "ready":
            response["report_url"] = f"/report/{session_id}"
        return JSONResponse(response)

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

    bank_by_id = {question.id: question for question in HR_QUESTIONS}

    def report_function(analyzed: InterviewSession) -> str:
        """Render an analyzed session using the existing standalone report."""
        return render_report(
            analyzed.to_dict(),
            bank_by_id,
            datetime.now(timezone.utc).isoformat(),
        )

    return WebServices(
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

        def transcribe(_captured: CapturedAnswer) -> TranscriptionResult:
            """Transcribe the validated, saved browser WAV using the configured provider."""
            return services.stt_provider.transcribe_detailed(
                audio_path.read_bytes(),
                filename=audio_path.name,
                content_type="audio/wav",
            )

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
        analyzed = services.analysis_function(
            entry.session,
            audio_root=Path("."),
            llm=analysis_llm,
            role=settings.role,
        )
        analysis_path = services.analysis_dir / f"{entry.session.session_id}.analyzed.json"
        analyzed.save_to(analysis_path)
        report_text = services.report_function(analyzed)
        report_path = services.reports_dir / f"{entry.session.session_id}.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        with entry.lock:
            entry.report_path = report_path
            entry.state = "ready"
            entry.message = "Report is ready."
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
