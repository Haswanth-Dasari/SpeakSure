"""Half-duplex orchestration for live, spoken mock interviews."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_capture import CapturedAnswer, save_wav
from .analysis.audio_delivery import analyze_audio_delivery
from .analysis.fluency import analyze_fluency
from .interviewer import Interviewer, InterviewerTurn
from .interviewer_prompts import REPEAT_REQUEST
from .session_log import InterviewSession, Turn
from .speech_to_text import TranscriptionResult


@dataclass(frozen=True)
class LiveSettings:
    """Runtime timing, transcript, and persistence settings for live interviews."""

    settle_seconds: float = 0.4
    min_answer_words: int = 2
    save_audio: bool = True
    sessions_dir: Path = Path("data/sessions")

    def __post_init__(self) -> None:
        """Validate microphone settling, minimum words, and session directory."""
        if (
            not math.isfinite(self.settle_seconds)
            or not 0 <= self.settle_seconds <= 3
        ):
            raise ValueError("settle_seconds must be from 0 to 3.")
        if (
            isinstance(self.min_answer_words, bool)
            or not isinstance(self.min_answer_words, int)
            or self.min_answer_words < 1
        ):
            raise ValueError("min_answer_words must be an integer of at least 1.")
        if not isinstance(self.save_audio, bool):
            raise ValueError("save_audio must be a boolean.")
        if "\x00" in str(self.sessions_dir):
            raise ValueError("sessions_dir must be a valid directory path.")


@dataclass(frozen=True)
class SpokenTiming:
    """Synthesis and playback durations for one spoken utterance."""

    synthesis_seconds: float
    playback_seconds: float


@dataclass(frozen=True)
class TurnProcessingResult:
    """State and output produced by processing one captured interview answer."""

    reply: InterviewerTurn | None
    turn: Turn | None
    transcription: TranscriptionResult | None
    capture_diagnostics: dict[str, object]
    empty_attempts: int
    listen_attempts: int
    transcription_errors: tuple[dict[str, object], ...]
    pending_speech_failure: bool
    transcription_error: bool = False


def format_transcript_flags(
    result: TranscriptionResult, threshold: float = 0.5
) -> str:
    """Mark recognized words with confidence below the requested threshold."""
    if not result.words:
        return result.text
    transcript_words = result.text.split()
    if len(transcript_words) == len(result.words):
        return " ".join(
            f"[{text}?]" if word.probability < threshold else text
            for text, word in zip(transcript_words, result.words)
        )
    return " ".join(
        f"[{word.text.strip()}?]"
        if word.probability < threshold
        else word.text.strip()
        for word in result.words
    )


def process_interview_turn(
    *,
    interviewer: Interviewer,
    session: InterviewSession,
    settings: LiveSettings,
    captured: CapturedAnswer,
    current_question: str,
    current_question_id: str | None,
    current_is_follow_up: bool,
    transcribe: Callable[[CapturedAnswer], TranscriptionResult],
    speak: Callable[[str], SpokenTiming | None] | None,
    clock: Callable[[], float],
    output: Callable[[str], None],
    empty_attempts: int = 0,
    listen_attempts: int = 0,
    transcription_errors: tuple[dict[str, object], ...] = (),
    pending_speech_failure: bool = False,
    audio_path_override: Path | None = None,
    safe_transcription_errors: bool = False,
) -> TurnProcessingResult:
    """Transcribe, analyze, decide, and record one turn for terminal or web use."""
    attempts = listen_attempts + 1
    capture_diagnostics = _capture_diagnostics(captured, attempts=attempts)
    if captured.samples.size:
        output(
            f"(heard {capture_diagnostics['seconds']:.1f} s, "
            f"peak {capture_diagnostics['peak']:.2f}, "
            f"level {capture_diagnostics['rms_dbfs']:.1f} dB)"
        )

    raw_result: TranscriptionResult | None = None
    stt_seconds = 0.0
    answer_text = ""
    errors = list(transcription_errors)
    if captured.stop_reason != "no_speech" and captured.samples.size:
        started_at = clock()
        try:
            raw_result = transcribe(captured)
        except Exception as error:
            stt_seconds = clock() - started_at
            empty_attempts += 1
            message = (
                "Speech transcription failed. Please try again."
                if safe_transcription_errors
                else str(error)[:200]
            )
            errors.append(
                {
                    "type": type(error).__name__,
                    "message": message,
                    "captured_seconds": capture_diagnostics["seconds"],
                }
            )
            if not safe_transcription_errors:
                output(f"(transcription error: {type(error).__name__}: {message})")
            output("Sorry, I could not understand that. Please try again.")
            if speak is not None:
                try:
                    speak(REPEAT_REQUEST)
                except Exception:
                    pending_speech_failure = True
            return TurnProcessingResult(
                reply=None,
                turn=None,
                transcription=None,
                capture_diagnostics=capture_diagnostics,
                empty_attempts=empty_attempts,
                listen_attempts=attempts,
                transcription_errors=tuple(errors),
                pending_speech_failure=pending_speech_failure,
                transcription_error=True,
            )
        stt_seconds = clock() - started_at
        answer_text = raw_result.text.strip()
        if len(answer_text.split()) < settings.min_answer_words:
            answer_text = ""
        output("You: " + format_transcript_flags(raw_result))
    else:
        output("You: ")

    if not answer_text:
        empty_attempts += 1
    llm_started_at = clock()
    reply = interviewer.next_turn(answer_text)
    llm_elapsed = clock() - llm_started_at
    llm_seconds = (
        llm_elapsed if reply.llm_latency_seconds is not None else None
    )
    if reply.kind == "repeat":
        output(f"Interviewer: {reply.utterance}")
        if speak is not None:
            try:
                speak(reply.utterance)
            except Exception:
                pending_speech_failure = True
        return TurnProcessingResult(
            reply=reply,
            turn=None,
            transcription=raw_result,
            capture_diagnostics=capture_diagnostics,
            empty_attempts=empty_attempts,
            listen_attempts=attempts,
            transcription_errors=tuple(errors),
            pending_speech_failure=pending_speech_failure,
        )

    final_answer = raw_result or _empty_transcription()
    turn_index = len(session.turns) + 1
    audio_path: str | None = None
    if settings.save_audio and captured.samples.size:
        if audio_path_override is not None:
            audio_path = str(audio_path_override)
        else:
            wav_path = (
                settings.sessions_dir
                / session.session_id
                / f"turn_{turn_index:02d}.wav"
            )
            audio_path = str(
                save_wav(wav_path, captured.samples, captured.sample_rate)
            )

    capture_settings = getattr(session, "capture_settings", None)
    end_silence = getattr(capture_settings, "end_silence_seconds", None)
    if end_silence is None:
        end_silence = session.metadata.get("end_silence_seconds")
    low_confidence_words = [
        {"word": word.text, "probability": word.probability}
        for word in final_answer.words
        if word.probability < 0.5
    ]
    fluency = analyze_fluency(
        final_answer, captured.thinking_time_seconds
    )
    audio_delivery = analyze_audio_delivery(audio_path, int(fluency["word_count"]))
    analysis: dict[str, object] = {
        "timing": {
            "capture_stop_reason": captured.stop_reason,
            "thinking_time_seconds": captured.thinking_time_seconds,
            "answer_duration_seconds": captured.answer_duration_seconds,
            "stt_seconds": stt_seconds,
            "llm_seconds": llm_seconds,
            "tts_synthesis_seconds": None,
            "response_gap_estimate_seconds": None,
            "end_silence_seconds": end_silence,
        },
        "interviewer": {
            "kind": reply.kind,
            "question_id": reply.question_id,
            "rationale": reply.rationale,
            "used_fallback": reply.used_fallback,
            "fallback_reason": reply.fallback_reason,
            "llm_latency_seconds": reply.llm_latency_seconds,
        },
        "asked": {
            "main_question_id": current_question_id,
            "is_follow_up": current_is_follow_up,
        },
        "empty_attempts": empty_attempts,
        "low_confidence_words": low_confidence_words,
        "transcription_errors": errors,
        "capture": capture_diagnostics,
        "fluency": fluency,
        "audio_delivery": audio_delivery,
    }
    if pending_speech_failure:
        analysis["speech_output_failed"] = True
        pending_speech_failure = False
    turn = session.add_turn(
        current_question,
        answer=final_answer,
        audio_path=audio_path,
        thinking_time_seconds=captured.thinking_time_seconds,
        answer_duration_seconds=captured.answer_duration_seconds,
        analysis=analysis,
    )

    output(f"Interviewer: {reply.utterance}")
    spoken_timing = None
    if speak is not None:
        try:
            spoken_timing = speak(reply.utterance)
        except Exception:
            pending_speech_failure = True
            turn.analysis["speech_output_failed"] = True
    if speak is not None:
        if spoken_timing is None:
            turn.analysis["speech_output_failed"] = True
        else:
            turn.analysis["timing"]["tts_synthesis_seconds"] = (
        spoken_timing.synthesis_seconds
            )
    tts_seconds = (
        spoken_timing.synthesis_seconds if spoken_timing is not None else 0.0
    )
    timing = turn.analysis["timing"]
    timing["response_gap_estimate_seconds"] = (
        stt_seconds + (llm_seconds or 0.0) + tts_seconds
    )
    return TurnProcessingResult(
        reply=reply,
        turn=turn,
        transcription=raw_result,
        capture_diagnostics=capture_diagnostics,
        empty_attempts=0,
        listen_attempts=0,
        transcription_errors=(),
        pending_speech_failure=pending_speech_failure,
    )


class LiveInterviewRunner:
    """Coordinate playback, microphone capture, transcription, and interview turns."""

    def __init__(
        self,
        *,
        interviewer: Interviewer,
        session: InterviewSession,
        settings: LiveSettings,
        listen: Callable[[], CapturedAnswer],
        transcribe: Callable[[CapturedAnswer], TranscriptionResult],
        speak: Callable[[str], SpokenTiming],
        output: Callable[[str], None] = print,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Store injected collaborators for deterministic and half-duplex execution."""
        self._interviewer = interviewer
        self._session = session
        self._settings = settings
        self._listen = listen
        self._transcribe = transcribe
        self._speak = speak
        self._output = output
        self._sleep = sleep
        self._clock = clock
        self._pending_speech_failure = False
        self._session_path: Path | None = None

    @property
    def session_path(self) -> Path | None:
        """Return the most recently persisted session path, if saved."""
        return self._session_path

    def run(self) -> InterviewSession:
        """Run the half-duplex interview and persist completed turns."""
        try:
            opening = self._interviewer.opening()
            self._output(f"Interviewer: {opening.utterance}")
            self._speak_or_report(opening.utterance)
            current_question = self._interviewer.plan.questions[0].text
            current_question_id: str | None = self._interviewer.plan.questions[0].id
            current_is_follow_up = False
            empty_attempts = 0
            listen_attempts = 0
            transcription_errors: list[dict[str, object]] = []

            while not self._interviewer.is_finished:
                self._sleep(self._settings.settle_seconds)
                self._output("Listening...")
                captured = self._listen()
                processed = process_interview_turn(
                    interviewer=self._interviewer,
                    session=self._session,
                    settings=self._settings,
                    captured=captured,
                    current_question=current_question,
                    current_question_id=current_question_id,
                    current_is_follow_up=current_is_follow_up,
                    transcribe=self._transcribe,
                    speak=self._speak_or_report,
                    clock=self._clock,
                    output=self._output,
                    empty_attempts=empty_attempts,
                    listen_attempts=listen_attempts,
                    transcription_errors=tuple(transcription_errors),
                    pending_speech_failure=self._pending_speech_failure,
                )
                empty_attempts = processed.empty_attempts
                listen_attempts = processed.listen_attempts
                transcription_errors = list(processed.transcription_errors)
                self._pending_speech_failure = processed.pending_speech_failure
                if processed.transcription_error or (
                    processed.reply is not None
                    and processed.reply.kind == "repeat"
                ):
                    continue
                reply = processed.reply
                if reply is None:
                    continue
                self._save_session()
                if reply.kind in {"follow_up", "clarify"}:
                    current_question = reply.utterance
                    current_is_follow_up = True
                elif reply.kind == "next_question":
                    current_question = reply.utterance
                    current_question_id = reply.question_id
                    current_is_follow_up = False
                empty_attempts = 0
                listen_attempts = 0
                transcription_errors = []
        except KeyboardInterrupt:
            self._save_session()
            self._output("Interview stopped early.")
        except Exception:
            self._save_session()
            raise
        return self._session

    def _speak_or_report(self, text: str) -> SpokenTiming | None:
        """Speak one utterance and fall back to text when playback fails."""
        try:
            return self._speak(text)
        except Exception:
            self._pending_speech_failure = True
            return None

    def _save_session(self) -> None:
        """Persist the current interview and remember its most recent path."""
        self._session_path = self._session.save(self._settings.sessions_dir)


def _empty_transcription() -> TranscriptionResult:
    """Create the explicit empty transcription used when no result exists."""
    return TranscriptionResult(
        text="",
        words=(),
        duration_seconds=0.0,
        language="en",
        language_probability=0.0,
        device="none",
    )


def _capture_diagnostics(
    captured: CapturedAnswer, *, attempts: int
) -> dict[str, object]:
    """Compute sample-level diagnostics for one capture and include attempt count."""
    samples = np.asarray(captured.samples, dtype=np.float64)
    seconds = len(samples) / captured.sample_rate if captured.sample_rate > 0 else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-6))
    return {
        "seconds": seconds,
        "peak": peak,
        "rms_dbfs": rms_dbfs,
        "stop_reason": captured.stop_reason,
        "attempts": attempts,
    }
