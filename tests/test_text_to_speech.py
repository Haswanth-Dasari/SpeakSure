import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.speaksure.text_to_speech import (
    PiperSpeechSynthesizer,
    SpeechSynthesisConfig,
    SpeechSynthesisError,
    SynthesizedAudio,
    build_speech_synthesizer,
    play_audio,
)


class FakeVoice:
    def __init__(self, chunks=None, error=None):
        self.config = SimpleNamespace(sample_rate=22050)
        self.chunks = chunks or []
        self.error = error
        self.calls = 0

    def synthesize(self, text, syn_config):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return iter(self.chunks)


def _write_voice_files(models_dir: Path, voice_name: str = "test-voice") -> None:
    (models_dir / f"{voice_name}.onnx").touch()
    (models_dir / f"{voice_name}.onnx.json").touch()


def test_chunked_int16_audio_is_converted_and_uses_voice_sample_rate(tmp_path):
    _write_voice_files(tmp_path)
    voice = FakeVoice(
        chunks=[
            SimpleNamespace(audio_int16_array=np.array([-32768, 0], dtype=np.int16)),
            SimpleNamespace(audio_int16_array=np.array([16384, 32767], dtype=np.int16)),
        ]
    )
    synthesizer = PiperSpeechSynthesizer(
        "test-voice", tmp_path, voice_loader=lambda path: voice
    )

    result = synthesizer.synthesize("Hello.")

    assert result.samples.dtype == np.float32
    assert result.samples == pytest.approx([-1.0, 0.0, 0.5, 32767 / 32768])
    assert result.sample_rate == 22050


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_empty_text_is_rejected(tmp_path, text):
    synthesizer = PiperSpeechSynthesizer("test-voice", tmp_path)

    with pytest.raises(ValueError, match="empty"):
        synthesizer.synthesize(text)


def test_text_longer_than_2000_characters_is_rejected(tmp_path):
    synthesizer = PiperSpeechSynthesizer("test-voice", tmp_path)

    with pytest.raises(ValueError, match="2000"):
        synthesizer.synthesize("a" * 2001)


def test_voice_is_loaded_only_once_across_synthesis_calls(tmp_path):
    _write_voice_files(tmp_path)
    voice = FakeVoice(
        chunks=[SimpleNamespace(audio_int16_array=np.array([0, 1000], dtype=np.int16))]
    )
    loads = []
    synthesizer = PiperSpeechSynthesizer(
        "test-voice",
        tmp_path,
        voice_loader=lambda path: loads.append(path) or voice,
    )

    synthesizer.synthesize("First.")
    synthesizer.synthesize("Second.")

    assert loads == [tmp_path / "test-voice.onnx"]
    assert voice.calls == 2


def test_missing_voice_file_error_includes_download_command(tmp_path):
    synthesizer = PiperSpeechSynthesizer(
        "test-voice", tmp_path, voice_loader=lambda path: pytest.fail("must not load")
    )

    with pytest.raises(SpeechSynthesisError) as error:
        synthesizer.synthesize("Hello.")

    assert (
        "python -m piper.download_voices --download-dir"
        in str(error.value)
    )
    assert "test-voice" in str(error.value)


def test_failing_voice_is_wrapped_and_chained(tmp_path):
    _write_voice_files(tmp_path)
    original_error = RuntimeError("ONNX failed")
    voice = FakeVoice(error=original_error)
    synthesizer = PiperSpeechSynthesizer(
        "test-voice", tmp_path, voice_loader=lambda path: voice
    )

    with pytest.raises(SpeechSynthesisError) as error:
        synthesizer.synthesize("Hello.")

    assert error.value.__cause__ is original_error


def test_build_synthesizer_uses_configured_values(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_TO_SPEECH_VOICE", "custom-voice")
    monkeypatch.setenv("TEXT_TO_SPEECH_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("TEXT_TO_SPEECH_LENGTH_SCALE", "1.4")

    synthesizer = build_speech_synthesizer(SpeechSynthesisConfig.from_env())

    assert isinstance(synthesizer, PiperSpeechSynthesizer)
    assert synthesizer.voice_name == "custom-voice"
    assert synthesizer.models_dir == tmp_path
    assert synthesizer.length_scale == 1.4


def test_speech_settings_defaults_and_valid_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("TEXT_TO_SPEECH_VOICE", raising=False)
    monkeypatch.delenv("TEXT_TO_SPEECH_MODELS_DIR", raising=False)
    monkeypatch.delenv("TEXT_TO_SPEECH_LENGTH_SCALE", raising=False)

    defaults = SpeechSynthesisConfig.from_env()
    assert defaults.voice_name == "en_US-lessac-medium"
    assert defaults.models_dir == Path("models/piper")
    assert defaults.length_scale == 1.0

    monkeypatch.setenv("TEXT_TO_SPEECH_VOICE", "en_GB-alan-medium")
    monkeypatch.setenv("TEXT_TO_SPEECH_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("TEXT_TO_SPEECH_LENGTH_SCALE", "0.5")
    configured = SpeechSynthesisConfig.from_env()
    assert configured.voice_name == "en_GB-alan-medium"
    assert configured.models_dir == tmp_path
    assert configured.length_scale == 0.5


@pytest.mark.parametrize("value", ["not-a-number", "0.49", "2.01", "nan", "inf"])
def test_invalid_speech_length_scale_is_rejected(monkeypatch, value):
    monkeypatch.setenv("TEXT_TO_SPEECH_LENGTH_SCALE", value)

    with pytest.raises(ValueError, match="TEXT_TO_SPEECH_LENGTH_SCALE"):
        SpeechSynthesisConfig.from_env()


def test_invalid_voice_and_model_directory_settings_are_rejected(monkeypatch):
    monkeypatch.setenv("TEXT_TO_SPEECH_VOICE", "../voice")
    with pytest.raises(ValueError, match="TEXT_TO_SPEECH_VOICE"):
        SpeechSynthesisConfig.from_env()

    with pytest.raises(ValueError, match="TEXT_TO_SPEECH_MODELS_DIR"):
        SpeechSynthesisConfig("valid-voice", Path("invalid\x00path"), 1.0)


def test_play_audio_uses_sounddevice_with_audio_sample_rate(monkeypatch):
    calls = {}
    sounddevice = SimpleNamespace(
        play=lambda samples, **kwargs: calls.update(samples=samples, **kwargs),
        wait=lambda: calls.update(waited=True),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    audio = SynthesizedAudio(
        samples=np.array([0.1, -0.1], dtype=np.float32),
        sample_rate=24000,
        synthesis_seconds=0.2,
    )

    playback_seconds = play_audio(audio, device=3)

    assert calls["samples"] is audio.samples
    assert calls["samplerate"] == 24000
    assert calls["device"] == 3
    assert calls["waited"] is True
    assert playback_seconds >= 0


def test_real_time_factor_uses_synthesis_and_audio_durations():
    audio = SynthesizedAudio(
        samples=np.zeros(4000, dtype=np.float32),
        sample_rate=2000,
        synthesis_seconds=1.5,
    )

    assert audio.duration_seconds == 2.0
    assert audio.real_time_factor == 0.75


def test_empty_audio_real_time_factor_is_zero():
    audio = SynthesizedAudio(
        samples=np.array([], dtype=np.float32),
        sample_rate=22050,
        synthesis_seconds=0.1,
    )

    assert audio.real_time_factor == 0.0
