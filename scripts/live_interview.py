"""Run a live, half-duplex spoken HR mock interview."""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.audio_capture import (  # noqa: E402
    SAMPLE_RATE,
    CaptureSettings,
    CapturedAnswer,
    calibrate_microphone,
    record_answer,
    wav_bytes,
)
from speaksure.interviewer import InterviewPlan, Interviewer  # noqa: E402
from speaksure.live_interview import (  # noqa: E402
    LiveInterviewRunner,
    LiveSettings,
    SpokenTiming,
)
from speaksure.llm_client import (  # noqa: E402
    GroqLLMConfig,
    LLMError,
    build_llm_client,
)
from speaksure.question_bank import select_questions  # noqa: E402
from speaksure.session_log import InterviewSession  # noqa: E402
from speaksure.speech_to_text import (  # noqa: E402
    EmptyAudioError,
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    TranscriptionResult,
)
from speaksure.text_to_speech import (  # noqa: E402
    SpeechSynthesisConfig,
    build_speech_synthesizer,
    speak as speak_text,
)


def main() -> int:
    """Configure and run a live voice interview with local speech recognition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--follow-ups", type=int, default=1)
    parser.add_argument("--role", default="software engineering fresher")
    parser.add_argument("--name", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--end-silence", type=float, default=1.5)
    parser.add_argument("--settle-seconds", type=float, default=0.4)
    parser.add_argument("--no-save-audio", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    try:
        questions = select_questions(args.questions, seed=args.seed)
        plan = InterviewPlan(
            questions=questions,
            role=args.role,
            candidate_name=args.name,
            max_followups_per_question=args.follow_ups,
        )
        stt_config = LocalTranscriptionConfig.from_env()
        stt_provider = LocalSpeechToTextProvider(stt_config)
        llm_config = GroqLLMConfig.from_env()
        if not llm_config.api_key:
            raise LLMError(
                "Groq is not configured. Set GROQ_API_KEY in the project-root .env file."
            )
        llm = build_llm_client(llm_config)
        tts_config = SpeechSynthesisConfig.from_env()
        synthesizer = build_speech_synthesizer(tts_config)
        capture_settings = CaptureSettings(
            calibration_seconds=2.0,
            end_silence_seconds=args.end_silence,
        )
        live_settings = LiveSettings(
            settle_seconds=args.settle_seconds,
            save_audio=not args.no_save_audio,
        )

        if not args.no_warmup:
            print("Warming up models...")
            warmup_audio = wav_bytes(np.zeros(SAMPLE_RATE, dtype=np.float32))
            try:
                stt_provider.transcribe_detailed(
                    warmup_audio,
                    filename="warmup.wav",
                    content_type="audio/wav",
                )
            except EmptyAudioError:
                pass
            synthesizer.synthesize("This is a speech warmup.")

        print("Calibrating the microphone. Please stay quiet for 2 seconds.")
        noise_profile = calibrate_microphone(
            settings=capture_settings,
            device=args.input_device,
        )
        print(
            f"Noise level: {noise_profile.noise_rms:.5f}; "
            f"threshold: {noise_profile.threshold:.5f}."
        )

        session = InterviewSession.new(
            mode="hr",
            difficulty="standard",
            planned_questions=len(questions),
            metadata={
                "stt_model": stt_config.model_size,
                "stt_device": stt_provider.actual_device
                if stt_provider.actual_device != "not-initialized"
                else stt_config.device,
                "llm_model": llm_config.model,
                "tts_voice": tts_config.voice_name,
                "end_silence_seconds": args.end_silence,
                "settle_seconds": args.settle_seconds,
                "seed": args.seed,
                "role": args.role,
            },
        )

        def listen() -> CapturedAnswer:
            """Capture one answer using the session's calibrated noise profile."""
            return record_answer(
                settings=capture_settings,
                device=args.input_device,
                noise_profile=noise_profile,
            )

        def transcribe(captured: CapturedAnswer) -> TranscriptionResult:
            """Transcribe captured WAV bytes with the configured local provider."""
            return stt_provider.transcribe_detailed(
                wav_bytes(captured.samples, captured.sample_rate),
                filename="answer.wav",
                content_type="audio/wav",
            )

        def speak(text: str) -> SpokenTiming:
            """Speak text on the selected output device and return its timings."""
            synthesis_seconds, playback_seconds = speak_text(
                text,
                synthesizer,
                device=args.output_device,
            )
            return SpokenTiming(synthesis_seconds, playback_seconds)

        interviewer = Interviewer(llm, plan)
        runner = LiveInterviewRunner(
            interviewer=interviewer,
            session=session,
            settings=live_settings,
            listen=listen,
            transcribe=transcribe,
            speak=speak,
        )
        completed = runner.run()
        _print_summary(completed, runner.session_path)
        return 0
    except Exception as error:
        error_name = type(error).__name__
        print(
            "Could not start or continue the live interview "
            f"({error_name}). Check the configured model, API key, and audio devices.",
            file=sys.stderr,
        )
        return 1


def _print_summary(
    session: InterviewSession, session_path: Path | None
) -> None:
    """Print turn totals, fallback count, average timings, and saved location."""
    timings = [
        turn.analysis.get("timing", {})
        for turn in session.turns
    ]
    empty_attempts = sum(
        int(turn.analysis.get("empty_attempts", 0)) for turn in session.turns
    )
    fallbacks = sum(
        bool(turn.analysis.get("interviewer", {}).get("used_fallback", False))
        for turn in session.turns
    )
    stt_values = _numeric_values(timing.get("stt_seconds") for timing in timings)
    llm_values = _numeric_values(timing.get("llm_seconds") for timing in timings)
    tts_values = _numeric_values(
        timing.get("tts_synthesis_seconds") for timing in timings
    )
    gap_values = _numeric_values(
        timing.get("response_gap_estimate_seconds") for timing in timings
    )
    print(
        f"Summary: {len(session.turns)} turns, {empty_attempts} empty attempts, "
        f"{fallbacks} fallbacks."
    )
    print(
        f"Average STT: {_average(stt_values):.3f}s; "
        f"LLM: {_average(llm_values):.3f}s; "
        f"TTS synthesis: {_average(tts_values):.3f}s; "
        f"response gap: {_average(gap_values):.3f}s."
    )
    print(f"Session: {session_path if session_path is not None else 'not saved'}")


def _numeric_values(values: Iterable[object]) -> list[float]:
    """Keep finite numeric values and omit missing timing measurements."""
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _average(values: list[float]) -> float:
    """Return the arithmetic mean or zero when no measurements are available."""
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
