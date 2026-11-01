"""Tests for local initial-prompt handling and filler-probe logic."""

import io
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from scripts.tools.filler_probe import (
    compare_transcripts,
    count_fillers,
    hallucination_risk,
    summarize_filler_counts,
    tokens_only_in_b,
)
from src.speaksure.configuration import get_speech_to_text_initial_prompt
from src.speaksure.speech_to_text import (
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    TranscriptWord,
    TranscriptionResult,
)


def make_wav_audio() -> bytes:
    """Create a short valid 16 kHz mono WAV for the fake local provider."""
    samples = np.zeros(1600, dtype="<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


@pytest.mark.parametrize(("prompt", "expected"), [("keep ums", "keep ums"), ("", None)])
def test_initial_prompt_reaches_local_whisper_call(prompt, expected):
    """Local transcription supplies the configured prompt or None to Whisper."""
    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            word = SimpleNamespace(
                word=" Hello",
                start=0.0,
                end=0.4,
                probability=0.9,
            )
            segment = SimpleNamespace(text="Hello.", words=[word])
            info = SimpleNamespace(
                duration=0.1,
                language="en",
                language_probability=0.9,
            )
            return iter([segment]), info

    config = LocalTranscriptionConfig(
        model_size="base",
        device="cpu",
        initial_prompt=prompt,
    )
    provider = LocalSpeechToTextProvider(
        config,
        model_loader=lambda *args: (FakeModel(), "cpu", "int8"),
    )

    provider.transcribe_detailed(
        make_wav_audio(), filename="fake.wav", content_type="audio/wav"
    )

    assert captured["initial_prompt"] is expected


def test_initial_prompt_setting_defaults_empty_and_rejects_over_300(monkeypatch):
    """Prompt environment setting defaults empty and enforces its length limit."""
    monkeypatch.delenv("SPEECH_TO_TEXT_INITIAL_PROMPT", raising=False)
    assert get_speech_to_text_initial_prompt() == ""
    assert LocalTranscriptionConfig.from_env().initial_prompt == ""

    monkeypatch.setenv("SPEECH_TO_TEXT_INITIAL_PROMPT", "x" * 301)
    with pytest.raises(ValueError, match="at most 300 characters"):
        LocalTranscriptionConfig.from_env()


def test_probe_helpers_compare_fillers_tokens_and_confidence():
    """Pure probe helpers count fillers and identify normalized B-only tokens."""
    result_a = TranscriptionResult(
        "Um, we can try.",
        (
            TranscriptWord("Um", 0, 0.1, 0.7),
            TranscriptWord("we", 0.2, 0.3, 0.9),
            TranscriptWord("can", 0.4, 0.5, 0.8),
            TranscriptWord("try", 0.6, 0.7, 0.8),
        ),
        1.0,
        "en",
        0.9,
        "cpu",
    )
    result_b = TranscriptionResult(
        "Umm, we can, uh, try.",
        (TranscriptWord("Umm", 0, 0.1, 0.8), TranscriptWord("try", 0.5, 0.8, 1.0)),
        1.0,
        "en",
        0.9,
        "cpu",
    )

    assert count_fillers("UM, then uh... hmm!") == 3
    assert tokens_only_in_b("We can try", "We can uh try") == ["uh"]
    assert compare_transcripts(result_a, result_b) == {
        "fillers_a": 1,
        "fillers_b": 2,
        "mean_probability_a": pytest.approx(0.8),
        "mean_probability_b": pytest.approx(0.9),
        "tokens_only_in_b": ["umm", "uh"],
    }
    assert summarize_filler_counts([(1, 2), (0, 4)]) == (1, 6)


def test_hallucination_risk_checks_any_nonempty_text_and_empty_output():
    """Risk detection flags output text and leaves a genuinely empty output clear."""
    assert hallucination_risk("Let me think.", "Let me think")
    assert not hallucination_risk("", "Let me think")
