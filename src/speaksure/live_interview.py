"""Half-duplex orchestration for live, spoken mock interviews."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio_capture import CapturedAnswer, save_wav
from .interviewer import Interviewer
from .interviewer_prompts import REPEAT_REQUEST
from .session_log import InterviewSession
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
            current_question = self._interviewer._plan.questions[0].text
            empty_attempts = 0

            while not self._interviewer.is_finished:
                self._sleep(self._settings.settle_seconds)
                self._output("Listening...")
                captured = self._listen()
                raw_result: TranscriptionResult | None = None
                stt_seconds = 0.0
                answer_text = ""

                if captured.stop_reason != "no_speech" and captured.samples.size:
                    started_at = self._clock()
                    try:
                        raw_result = self._transcribe(captured)
                    except Exception:
                        stt_seconds = self._clock() - started_at
                        empty_attempts += 1
                        self._output("Sorry, I could not understand that. Please try again.")
                        self._speak_or_report(REPEAT_REQUEST)
                        continue
                    stt_seconds = self._clock() - started_at
                    answer_text = raw_result.text.strip()
                    if len(answer_text.split()) < self._settings.min_answer_words:
                        answer_text = ""
                    self._output("You: " + format_transcript_flags(raw_result))
                else:
                    self._output("You: ")

                if not answer_text:
                    empty_attempts += 1
                llm_started_at = self._clock()
                reply = self._interviewer.next_turn(answer_text)
                llm_elapsed = self._clock() - llm_started_at
                llm_seconds = (
                    llm_elapsed if reply.llm_latency_seconds is not None else None
                )
                if reply.kind == "repeat":
                    self._output(f"Interviewer: {reply.utterance}")
                    self._speak_or_report(reply.utterance)
                    continue

                final_answer = raw_result or _empty_transcription()
                turn_index = len(self._session.turns) + 1
                audio_path: str | None = None
                if self._settings.save_audio and captured.samples.size:
                    wav_path = (
                        self._settings.sessions_dir
                        / self._session.session_id
                        / f"turn_{turn_index:02d}.wav"
                    )
                    audio_path = str(
                        save_wav(wav_path, captured.samples, captured.sample_rate)
                    )

                capture_settings = getattr(
                    self._session, "capture_settings", None
                )
                end_silence = getattr(
                    capture_settings, "end_silence_seconds", None
                )
                if end_silence is None:
                    end_silence = self._session.metadata.get(
                        "end_silence_seconds"
                    )
                low_confidence_words = [
                    {"word": word.text, "probability": word.probability}
                    for word in final_answer.words
                    if word.probability < 0.5
                ]
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
                    "empty_attempts": empty_attempts,
                    "low_confidence_words": low_confidence_words,
                }
                if self._pending_speech_failure:
                    analysis["speech_output_failed"] = True
                    self._pending_speech_failure = False
                turn = self._session.add_turn(
                    current_question,
                    answer=final_answer,
                    audio_path=audio_path,
                    thinking_time_seconds=captured.thinking_time_seconds,
                    answer_duration_seconds=captured.answer_duration_seconds,
                    analysis=analysis,
                )

                self._output(f"Interviewer: {reply.utterance}")
                spoken_timing = self._speak_or_report(reply.utterance)
                if spoken_timing is None:
                    turn.analysis["speech_output_failed"] = True
                else:
                    timing = turn.analysis["timing"]
                    timing["tts_synthesis_seconds"] = spoken_timing.synthesis_seconds
                tts_seconds = (
                    spoken_timing.synthesis_seconds if spoken_timing is not None else 0.0
                )
                timing = turn.analysis["timing"]
                timing["response_gap_estimate_seconds"] = (
                    stt_seconds + (llm_seconds or 0.0) + tts_seconds
                )
                self._save_session()
                if reply.kind != "closing":
                    current_question = reply.utterance
                empty_attempts = 0
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
