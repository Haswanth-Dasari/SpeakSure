"""Transcribe a WAV file and demonstrate session log persistence."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.session_log import InterviewSession  # noqa: E402
from speaksure.speech_to_text import (  # noqa: E402
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
)


def main() -> None:
    """Transcribe one WAV file, persist a session, and verify its round trip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path, help="Path to a WAV recording")
    wav_path = parser.parse_args().wav_path
    provider = LocalSpeechToTextProvider(LocalTranscriptionConfig.from_env())
    result = provider.transcribe_detailed(
        wav_path.read_bytes(),
        filename=wav_path.name,
        content_type="audio/wav",
    )
    session = InterviewSession.new(
        mode="HR interview",
        difficulty="Intermediate",
        planned_questions=1,
        metadata={"model_size": provider.config.model_size},
    )
    session.add_turn(
        "Tell me about yourself.",
        answer=result,
        answer_duration_seconds=result.duration_seconds,
    )
    saved_path = session.save(PROJECT_ROOT / "data" / "sessions")
    reloaded = InterviewSession.load(saved_path)
    if reloaded.to_dict() != session.to_dict():
        raise RuntimeError("Saved session did not match the original.")
    probabilities = [word.probability for word in result.words]
    lowest_probability = min(probabilities) if probabilities else 0.0

    print(saved_path)
    print("round-trip OK")
    print(f"Turns: {len(reloaded.turns)}")
    print(f"Words: {len(result.words)}")
    print(f"Lowest word probability: {lowest_probability:.3f}")


if __name__ == "__main__":
    main()
