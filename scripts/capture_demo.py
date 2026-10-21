"""Capture one microphone answer, optionally transcribe it, and save a WAV."""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.audio_capture import (  # noqa: E402
    CaptureSettings,
    list_input_devices,
    record_answer,
    save_wav,
    wav_bytes,
)
from speaksure.speech_to_text import (  # noqa: E402
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
)


def main() -> int:
    """Run microphone capture, save the answer, and optionally transcribe it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--end-silence", type=float, default=1.5)
    parser.add_argument("--no-speech-timeout", type=float, default=20.0)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--no-transcribe", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("data/captures"))
    args = parser.parse_args()

    if args.list_devices:
        devices = list_input_devices()
        if not devices:
            print("No input devices found.")
        for index, name in devices:
            print(f"{index}: {name}")
        return 0

    settings = CaptureSettings(
        end_silence_seconds=args.end_silence,
        no_speech_timeout_seconds=args.no_speech_timeout,
        max_answer_seconds=args.max_seconds,
    )

    def report_event(event: str, details: dict[str, float]) -> None:
        if event == "calibrated":
            print(
                f"Noise level: {details['noise_rms']:.5f}; "
                f"threshold: {details['threshold']:.5f}. Speak now."
            )
        elif event == "speech_started":
            print("(speech detected)")

    print("Stay quiet for 1 second...")
    captured = record_answer(
        settings=settings,
        device=args.device,
        on_event=report_event,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    saved_path = None
    if captured.stop_reason != "no_speech":
        saved_path = save_wav(
            args.out_dir / f"capture_{timestamp}.wav",
            captured.samples,
            captured.sample_rate,
        )
    total_seconds = len(captured.samples) / captured.sample_rate
    print(f"Stop reason: {captured.stop_reason}")
    print(f"Thinking time: {captured.thinking_time_seconds:.2f} seconds")
    print(f"Answer duration: {captured.answer_duration_seconds:.2f} seconds")
    print(f"Total captured: {total_seconds:.2f} seconds")
    print(f"Noise RMS: {captured.noise_rms:.5f}")
    print(f"Threshold: {captured.threshold:.5f}")
    if saved_path is not None:
        print(f"Saved: {saved_path}")

    if captured.stop_reason == "no_speech":
        print("No speech was detected before the timeout.")
        return 0
    if args.no_transcribe:
        return 0

    provider = LocalSpeechToTextProvider(LocalTranscriptionConfig.from_env())
    result = provider.transcribe_detailed(
        wav_bytes(captured.samples, captured.sample_rate),
        filename=saved_path.name,
        content_type="audio/wav",
    )
    probabilities = [word.probability for word in result.words]
    lowest_probability = min(probabilities) if probabilities else None
    print(f"Device: {result.device}")
    print(f"Text: {result.text}")
    if lowest_probability is None:
        print("Lowest word probability: unavailable")
    else:
        print(f"Lowest word probability: {lowest_probability:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
