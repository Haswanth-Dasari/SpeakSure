"""Compare local transcription with and without a disfluency-style prompt."""

from __future__ import annotations

import argparse
import difflib
import io
import sys
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.analysis.lexicons import HESITATION_FILLERS  # noqa: E402
from speaksure.analysis.text_utils import normalize_token  # noqa: E402
from speaksure.speech_to_text import (  # noqa: E402
    EmptyAudioError,
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    SpeechToTextError,
    TranscriptionResult,
)

FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
SAMPLE_RATE = 16000


def count_fillers(text: str) -> int:
    """Count normalized hesitation-filler tokens in a transcript."""
    return sum(
        normalize_token(token) in HESITATION_FILLERS
        for token in text.split()
    )


def mean_word_probability(result: TranscriptionResult) -> float | None:
    """Return the mean recognized-word probability, or None without words."""
    if not result.words:
        return None
    return sum(word.probability for word in result.words) / len(result.words)


def normalized_tokens(text: str) -> list[str]:
    """Split text into non-empty normalized tokens."""
    return [
        token
        for part in text.split()
        if (token := normalize_token(part))
    ]


def tokens_only_in_b(text_a: str, text_b: str) -> list[str]:
    """Return normalized B tokens in inserted/replacement diff regions."""
    tokens_a = normalized_tokens(text_a)
    tokens_b = normalized_tokens(text_b)
    matcher = difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False)
    return [
        token
        for tag, _, _, b_start, b_end in matcher.get_opcodes()
        if tag in {"insert", "replace"}
        for token in tokens_b[b_start:b_end]
    ]


def compare_transcripts(
    result_a: TranscriptionResult, result_b: TranscriptionResult
) -> dict[str, object]:
    """Summarize filler counts, confidence means, and B-only normalized tokens."""
    return {
        "fillers_a": count_fillers(result_a.text),
        "fillers_b": count_fillers(result_b.text),
        "mean_probability_a": mean_word_probability(result_a),
        "mean_probability_b": mean_word_probability(result_b),
        "tokens_only_in_b": tokens_only_in_b(result_a.text, result_b.text),
    }


def summarize_filler_counts(counts: list[tuple[int, int]]) -> tuple[int, int]:
    """Sum paired A and B filler counts across files."""
    return (
        sum(count_a for count_a, _ in counts),
        sum(count_b for _, count_b in counts),
    )


def hallucination_risk(text: str, prompt: str) -> bool:
    """Flag any non-empty output or an output containing a prompt word."""
    output_tokens = set(normalized_tokens(text))
    prompt_tokens = set(normalized_tokens(prompt))
    return bool(text.strip()) or bool(output_tokens & prompt_tokens)


def read_wav(path: Path) -> bytes:
    """Read a WAV file after checking the required sample format."""
    try:
        audio = path.read_bytes()
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            if (
                wav_file.getframerate() != SAMPLE_RATE
                or wav_file.getnchannels() != 1
                or wav_file.getsampwidth() != 2
                or wav_file.getcomptype() != "NONE"
                or wav_file.getnframes() <= 0
            ):
                raise ValueError("expected non-empty 16 kHz mono 16-bit PCM WAV")
    except (OSError, EOFError, wave.Error, ValueError) as error:
        raise ValueError(f"{path}: {error}") from None
    return audio


def silence_and_noise_wav() -> tuple[bytes, bytes]:
    """Build three seconds of digital silence and seeded low-level noise in memory."""
    sample_count = SAMPLE_RATE * 3
    silence = np.zeros(sample_count, dtype="<i2")
    noise = np.random.default_rng(0).uniform(
        -0.002, 0.002, sample_count
    )
    noise_samples = np.rint(noise * 32767).astype("<i2")
    return _encode_wav(silence), _encode_wav(noise_samples)


def _encode_wav(samples: np.ndarray) -> bytes:
    """Encode mono PCM16 samples as an in-memory WAV."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(samples.tobytes())
    return buffer.getvalue()


def _transcribe(
    audio: bytes, config: LocalTranscriptionConfig, filename: str
) -> TranscriptionResult:
    """Transcribe WAV bytes through the local provider, treating no speech as empty."""
    provider = LocalSpeechToTextProvider(config)
    try:
        return provider.transcribe_detailed(
            audio, filename=filename, content_type="audio/wav"
        )
    except EmptyAudioError:
        return TranscriptionResult("", (), 0.0, "en", 0.0, "none")


def _format_probability(probability: float | None) -> str:
    """Render an optional mean confidence value."""
    return "-" if probability is None else f"{probability:.3f}"


def main(argv: list[str] | None = None) -> int:
    """Run paired local transcriptions for WAV files and optional silence checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_files", nargs="+", type=Path)
    parser.add_argument("--prompt", default=FILLER_PROMPT)
    parser.add_argument("--silence-check", action="store_true")
    args = parser.parse_args(argv)

    try:
        file_audio = [(path, read_wav(path)) for path in args.wav_files]
        base_config = LocalTranscriptionConfig.from_env()
        config_a = replace(base_config, initial_prompt="")
        config_b = replace(base_config, initial_prompt=args.prompt)
        totals: list[tuple[int, int]] = []
        for path, audio in file_audio:
            result_a = _transcribe(audio, config_a, path.name)
            result_b = _transcribe(audio, config_b, path.name)
            comparison = compare_transcripts(result_a, result_b)
            totals.append(
                (int(comparison["fillers_a"]), int(comparison["fillers_b"]))
            )
            print(f"\nFile: {path}")
            print(f"Transcript A (no prompt): {result_a.text}")
            print(f"Transcript B (prompt): {result_b.text}")
            print(
                "Fillers A/B: "
                f"{comparison['fillers_a']} / {comparison['fillers_b']}"
            )
            print(
                "Mean word probability A/B: "
                f"{_format_probability(comparison['mean_probability_a'])} / "
                f"{_format_probability(comparison['mean_probability_b'])}"
            )
            print(
                "Words only in B: "
                + (", ".join(comparison["tokens_only_in_b"]) or "-")
            )

        total_a, total_b = summarize_filler_counts(totals)
        print(f"\nTotal fillers in A: {total_a}")
        print(f"Total fillers in B: {total_b}")

        if args.silence_check:
            silence_audio, noise_audio = silence_and_noise_wav()
            silence_result = _transcribe(
                silence_audio, config_b, "silence.wav"
            )
            noise_result = _transcribe(noise_audio, config_b, "noise.wav")
            print(f"\nSilence transcript: {silence_result.text}")
            print(f"Noise transcript: {noise_result.text}")
            risk = hallucination_risk(
                silence_result.text, args.prompt
            ) or hallucination_risk(noise_result.text, args.prompt)
            print("HALLUCINATION RISK" if risk else "no hallucination on silence/noise")
    except (SpeechToTextError, ValueError) as error:
        print(f"Filler probe failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
