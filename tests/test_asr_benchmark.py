"""Offline tests for the injectable ASR benchmark loop."""

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.asr_benchmark import DEFAULT_MODELS, run_benchmark, write_outputs
from src.speaksure.speech_to_text import TranscriptWord


def _wav_bytes() -> bytes:
    """Make a tiny WAV solely for duration parsing by the benchmark."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16000)
        audio_file.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    return output.getvalue()


def test_default_benchmark_models_match_supported_local_sizes():
    assert DEFAULT_MODELS == ("base", "small", "medium", "large-v3")


def test_benchmark_skips_load_failure_and_excludes_cpu_reported_for_cuda(
    tmp_path, capsys
):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(_wav_bytes())
    rows = [
        {
            "clip": clip,
            "reference": "Hello world",
            "names": ["Maya Patel"],
            "condition": "plain",
            "input_device": "USB microphone",
            "input_device_id": "2",
            "category": "names",
            "notes": "quiet room",
        }
    ]
    configurations = []

    class FakeProvider:
        def __init__(self, config):
            configurations.append(config)
            if config.model_size == "small":
                raise RuntimeError("fake load rejection")
            self.actual_device = "cuda" if config.model_size == "base" else "cpu"

        def transcribe_detailed(self, *_args, **_kwargs):
            device = self.actual_device
            return SimpleNamespace(
                text="Hello world Maya Patel",
                words=(TranscriptWord("Hello", 0, 0.2, 0.9),),
                device=device,
            )

    results, failures = run_benchmark(
        rows,
        ["base", "small", "medium"],
        device="cuda",
        language="en",
        hotwords="Maya",
        provider_factory=FakeProvider,
        torch_module=False,
    )

    assert [row["model"] for row in results] == ["base", "medium"]
    assert results[0]["actual_device"] == "cuda"
    assert results[0]["warmup"] is True
    assert results[0]["hotwords"] == "Maya"
    assert results[0]["input_device"] == "USB microphone"
    assert results[0]["category"] == "names"
    assert results[0]["word_level_errors"] == "[ins maya] [ins patel]"
    assert results[1]["device_mismatch"] is True
    assert failures == [("small", "RuntimeError: fake load rejection")]
    assert [config.model_size for config in configurations] == [
        "base",
        "small",
        "medium",
    ]
    assert "Skipping small:" in capsys.readouterr().err


def test_benchmark_outputs_group_counts_and_per_clip_word_errors(tmp_path):
    rows = [
        {
            "model": "base",
            "clip": "clip.wav",
            "reference": "hello there",
            "hypothesis": "hello world",
            "word_level_errors": "[there->world]",
            "wer": 0.5,
            "input_device": "USB microphone",
            "category": "plain",
            "actual_device": "cpu",
        }
    ]

    markdown_path = write_outputs(
        tmp_path / "results.csv",
        rows,
        ("model", "input_device", "category"),
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| model=base, input_device=USB microphone, category=plain | 2 |" in markdown
    assert "Reference: hello there" in markdown
    assert "Hypothesis: hello world" in markdown
    assert "Word errors: [there->world]" in markdown
    header = (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.endswith(
        "input_device,input_device_id,category,notes,word_level_errors"
    )
