"""Print detailed local transcription output for a WAV recording."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.speech_to_text import (  # noqa: E402
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    describe_stt_configuration,
)


def main() -> None:
    """Transcribe a WAV file and print its text, timing, and confidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path, help="Path to a WAV recording")
    wav_path = parser.parse_args().wav_path
    config = LocalTranscriptionConfig.from_env()
    description = describe_stt_configuration(config)
    print(
        f"STT: model={description['model']} "
        f"device={description['device_setting']} "
        f"language={description['language']}"
    )
    audio = wav_path.read_bytes()
    provider = LocalSpeechToTextProvider(config)
    result = provider.transcribe_detailed(
        audio,
        filename=wav_path.name,
        content_type="audio/wav",
    )

    print(f"Device: {result.device}")
    print(
        f"Language: {result.language or 'unknown'} "
        f"(probability: {result.language_probability:.3f})"
    )
    print("\nTranscript:")
    print(result.text)
    print("\nWords:")
    print(f"{'Word':<24} {'Start':>10} {'End':>10} {'Probability':>12}")
    for word in result.words:
        print(
            f"{word.text:<24} {word.start:>10.3f} {word.end:>10.3f} "
            f"{word.probability:>12.3f}"
        )


if __name__ == "__main__":
    main()
