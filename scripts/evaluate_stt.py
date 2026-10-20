import time
import wave
from pathlib import Path
from statistics import median

from jiwer import wer
from speaksure.speech_to_text import (
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
)

AUDIO_PATH = Path("harvard.wav")

REFERENCE = (
    "The stale smell of old beer lingers. "
    "It takes heat to bring out the odor. "
    "A cold dip restores health and zest. "
    "A salt pickle tastes fine with ham. "
    "Tacos al pastor are my favorite. "
    "A zestful food is the hot cross bun."
)


def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    audio_bytes = AUDIO_PATH.read_bytes()

    with wave.open(str(AUDIO_PATH), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()

    config = LocalTranscriptionConfig.from_env()
    provider = LocalSpeechToTextProvider(config)

    print("SpeakSure WER Baseline")
    print("=" * 45)
    print(f"Audio: {AUDIO_PATH}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Model: {config.model_size}")
    print(f"Configured device: {config.device}")

    latencies = []
    transcript = ""

    for run in range(1, 4):
        start = time.perf_counter()

        transcript = provider.transcribe(
            audio_bytes,
            filename=AUDIO_PATH.name,
            content_type="audio/wav",
        )

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        print(f"\nRun {run}: {elapsed:.3f} seconds")
        print(f"WER: {wer(REFERENCE, transcript):.4f}")

    print("\n" + "=" * 45)
    print(f"Actual initialized device: {provider.actual_device}")
    print(f"Median of warm runs: {median(latencies[1:]):.3f} seconds")
    print(f"Warm real-time factor: {median(latencies[1:]) / duration:.4f}")
    print(f"Final WER: {wer(REFERENCE, transcript):.4f}")
    print("\nReference:")
    print(REFERENCE)
    print("\nModel transcript:")
    print(transcript)


if __name__ == "__main__":
    main()
