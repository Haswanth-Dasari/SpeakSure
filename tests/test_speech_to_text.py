import io
import sys
import wave
from types import ModuleType
from types import SimpleNamespace

import pytest

from src.speaksure import configuration
from src.speaksure import speech_to_text
from src.speaksure.speech_to_text import (
    EmptyAudioError,
    LocalModelError,
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    MissingTranscriptionConfiguration,
    OpenAISpeechToTextProvider,
    SpeechToTextError,
    TranscriptionConfig,
    UnsupportedAudioFormat,
    _cached_local_model,
    create_speech_to_text_provider,
    validate_audio,
)


@pytest.fixture(autouse=True)
def clear_local_model_cache():
    _cached_local_model.cache_clear()
    speech_to_text._LOADED_LOCAL_DEVICES.clear()
    yield
    _cached_local_model.cache_clear()
    speech_to_text._LOADED_LOCAL_DEVICES.clear()


def make_wav_audio() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0" * 1600)
    return buffer.getvalue()


def test_audio_validation_accepts_streamlit_wav_recording():
    validate_audio(b"wav data", "audio/wav")
    validate_audio(b"webm data", "audio/webm;codecs=opus")


@pytest.mark.parametrize("audio", [b"", None])
def test_audio_validation_rejects_empty_recordings(audio):
    with pytest.raises(EmptyAudioError, match="No audio"):
        validate_audio(audio, "audio/wav")


def test_audio_validation_rejects_unsupported_format_and_large_recording():
    for content_type in ("video/mp4", "audio/ogg", "audio/flac"):
        with pytest.raises(UnsupportedAudioFormat, match="format"):
            validate_audio(b"data", content_type)
    with pytest.raises(UnsupportedAudioFormat, match="too large"):
        validate_audio(b"x" * 25_000_001, "audio/wav")


def test_openai_provider_transcribes_audio_with_mocked_client():
    captured = {}

    class Client:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return SimpleNamespace(text="  Recognized words.  ")

    provider = OpenAISpeechToTextProvider(
        TranscriptionConfig("test-key", "test-transcription-model"),
        client_factory=lambda **kwargs: Client(),
    )

    transcript = provider.transcribe(
        b"wav data", filename="recording.wav", content_type="audio/wav"
    )

    assert transcript == "Recognized words."
    assert captured == {
        "model": "test-transcription-model",
        "file": ("recording.wav", b"wav data", "audio/wav"),
    }


def test_provider_initialization_uses_environment_configuration(monkeypatch):
    client_arguments = {}
    request_arguments = {}

    class Client:
        def __init__(self, **kwargs):
            client_arguments.update(kwargs)

        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    request_arguments.update(kwargs)
                    return SimpleNamespace(text="Recognized.")

    monkeypatch.setenv("OPENAI_API_KEY", "environment-test-key")
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "environment-model")
    provider = OpenAISpeechToTextProvider(client_factory=Client)

    provider.transcribe(
        b"webm data",
        filename="recording.webm",
        content_type="audio/webm;codecs=opus",
    )

    assert client_arguments == {"api_key": "environment-test-key"}
    assert request_arguments == {
        "model": "environment-model",
        "file": ("recording.webm", b"webm data", "audio/webm"),
    }


def test_openai_provider_rejects_empty_transcription():
    class Client:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    return SimpleNamespace(text="  ")

    provider = OpenAISpeechToTextProvider(
        TranscriptionConfig("test-key", "whisper-1"),
        client_factory=lambda **kwargs: Client(),
    )

    with pytest.raises(EmptyAudioError, match="No speech was recognized"):
        provider.transcribe(b"wav data", filename="recording.wav", content_type="audio/wav")


def test_openai_provider_sanitizes_provider_errors(caplog):
    class Client:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("request failed with secret")

    provider = OpenAISpeechToTextProvider(
        TranscriptionConfig("test-secret", "whisper-1"),
        client_factory=lambda **kwargs: Client(),
    )

    with pytest.raises(SpeechToTextError, match="transcription failed") as error:
        provider.transcribe(b"wav data", filename="recording.wav", content_type="audio/wav")
    assert "secret" not in str(error.value)
    assert "secret" not in caplog.text


@pytest.mark.parametrize(
    ("status_code", "error_code", "message"),
    [
        (401, None, "authentication failed"),
        (429, "insufficient_quota", "quota or rate limit"),
        (None, None, "network and try again"),
    ],
)
def test_openai_provider_reports_safe_actionable_errors(
    status_code, error_code, message
):
    class ProviderFailure(RuntimeError):
        def __init__(self):
            self.status_code = status_code
            self.code = error_code
            super().__init__("secret-bearing provider details")

    class Client:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kwargs):
                    if status_code is None:
                        raise ConnectionError("private network detail")
                    raise ProviderFailure()

    provider = OpenAISpeechToTextProvider(
        TranscriptionConfig("test-secret", "whisper-1"),
        client_factory=lambda **kwargs: Client(),
    )
    with pytest.raises(SpeechToTextError, match=message) as error:
        provider.transcribe(b"wav data", filename="recording.wav", content_type="audio/wav")
    assert "secret" not in str(error.value)


def test_missing_transcription_configuration_is_clear(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingTranscriptionConfiguration, match="OPENAI_API_KEY"):
        OpenAISpeechToTextProvider()


def test_local_provider_selection_does_not_require_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SPEECH_TO_TEXT_PROVIDER", "local")
    monkeypatch.setenv("SPEECH_TO_TEXT_MODEL_SIZE", "base")
    monkeypatch.setenv("SPEECH_TO_TEXT_DEVICE", "cpu")

    provider = create_speech_to_text_provider()

    assert isinstance(provider, LocalSpeechToTextProvider)


def test_local_provider_and_model_defaults(monkeypatch):
    monkeypatch.delenv("SPEECH_TO_TEXT_PROVIDER", raising=False)
    monkeypatch.delenv("SPEECH_TO_TEXT_MODEL_SIZE", raising=False)
    monkeypatch.delenv("SPEECH_TO_TEXT_DEVICE", raising=False)

    assert configuration.get_speech_to_text_provider() == "local"
    assert LocalTranscriptionConfig.from_env() == LocalTranscriptionConfig(
        model_size="base", device="auto"
    )


def test_openai_provider_remains_explicit_and_requires_configuration(monkeypatch):
    monkeypatch.setenv("SPEECH_TO_TEXT_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingTranscriptionConfiguration):
        create_speech_to_text_provider()


def test_invalid_provider_does_not_fall_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SpeechToTextError, match="local.*openai"):
        create_speech_to_text_provider("invalid")


def test_local_transcription_uses_cached_mock_model_without_key(monkeypatch):
    initialized = []
    inputs = []

    class FakeModel:
        def transcribe(self, audio, *, beam_size):
            inputs.append((audio.dtype.name, audio.size, beam_size))
            return iter([SimpleNamespace(text=" Hello "), SimpleNamespace(text="world.")]), object()

    def fake_model_factory(model_size, *, device, compute_type):
        initialized.append((model_size, device, compute_type))
        return FakeModel()

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = fake_model_factory
    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 0
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = LocalSpeechToTextProvider(
        LocalTranscriptionConfig(model_size="base", device="auto")
    )

    assert provider.transcribe(
        make_wav_audio(), filename="recording.wav", content_type="audio/wav"
    ) == "Hello world."
    assert provider.transcribe(
        make_wav_audio(), filename="recording.wav", content_type="audio/wav"
    ) == "Hello world."
    assert initialized == [("base", "cpu", "int8")]
    assert inputs == [("float32", 1600, 5), ("float32", 1600, 5)]


def test_cuda_uses_float16_when_available(monkeypatch):
    initialized = []

    class FakeModel:
        def transcribe(self, audio, *, beam_size):
            return iter([SimpleNamespace(text="GPU mock.")]), object()

    def fake_model_factory(model_size, *, device, compute_type):
        initialized.append((model_size, device, compute_type))
        return FakeModel()

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = fake_model_factory
    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 1
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    transcript = LocalSpeechToTextProvider(
        LocalTranscriptionConfig(model_size="base", device="auto")
    ).transcribe(make_wav_audio(), filename="recording.wav", content_type="audio/wav")

    assert transcript == "GPU mock."
    assert initialized == [("base", "cuda", "float16")]


def test_auto_mode_falls_back_to_cpu_if_cuda_initialization_fails(monkeypatch):
    initialized = []

    class FakeModel:
        def transcribe(self, audio, *, beam_size):
            return iter([SimpleNamespace(text="CPU fallback.")]), object()

    def fake_model_factory(model_size, *, device, compute_type):
        initialized.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("CUDA unavailable")
        return FakeModel()

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = fake_model_factory
    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 1
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    transcript = LocalSpeechToTextProvider(
        LocalTranscriptionConfig(model_size="base", device="auto")
    ).transcribe(make_wav_audio(), filename="recording.wav", content_type="audio/wav")

    assert transcript == "CPU fallback."
    assert initialized == [("cuda", "float16"), ("cpu", "int8")]
    assert speech_to_text.get_loaded_local_device("base", "auto") == "cpu"


def test_auto_mode_falls_back_if_cuda_inference_fails(monkeypatch):
    initialized = []

    class FakeModel:
        def __init__(self, device):
            self.device = device

        def transcribe(self, audio, *, beam_size):
            if self.device == "cuda":
                raise RuntimeError("missing CUDA runtime")
            return iter([SimpleNamespace(text="CPU inference fallback.")]), object()

    def fake_model_factory(model_size, *, device, compute_type):
        initialized.append((device, compute_type))
        return FakeModel(device)

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = fake_model_factory
    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 1
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    provider = LocalSpeechToTextProvider(
        LocalTranscriptionConfig(model_size="base", device="auto")
    )
    transcript = provider.transcribe(
        make_wav_audio(), filename="recording.wav", content_type="audio/wav"
    )

    assert transcript == "CPU inference fallback."
    assert initialized == [("cuda", "float16"), ("cpu", "int8")]
    assert speech_to_text.get_loaded_local_device("base", "auto") == "cpu"
    assert provider.actual_device == "cpu"


def test_explicit_cuda_errors_instead_of_falling_back(monkeypatch):
    initialized = []

    def fake_model_factory(model_size, *, device, compute_type):
        initialized.append(device)
        raise RuntimeError("CUDA runtime details")

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = fake_model_factory
    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 1
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)

    with pytest.raises(LocalModelError, match="CUDA"):
        LocalSpeechToTextProvider(
            LocalTranscriptionConfig(model_size="base", device="cuda")
        ).transcribe(make_wav_audio(), filename="recording.wav", content_type="audio/wav")
    assert initialized == ["cuda"]


def test_local_model_initialization_failure_is_safe(monkeypatch):
    faster_whisper = ModuleType("faster_whisper")

    def fail_initialization(*args, **kwargs):
        raise RuntimeError("sensitive runtime detail")

    faster_whisper.WhisperModel = fail_initialization
    ctranslate2 = ModuleType("ctranslate2")
    ctranslate2.get_cuda_device_count = lambda: 0
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
    provider = LocalSpeechToTextProvider(
        LocalTranscriptionConfig(model_size="base", device="cpu")
    )

    with pytest.raises(LocalModelError, match="CPU") as error:
        provider.transcribe(
            make_wav_audio(), filename="recording.wav", content_type="audio/wav"
        )
    assert "sensitive runtime detail" not in str(error.value)


def test_local_transcription_keeps_audio_validation():
    provider = LocalSpeechToTextProvider(
        LocalTranscriptionConfig(model_size="base", device="cpu"),
        model_loader=lambda *args: pytest.fail("invalid audio reached the model"),
    )
    with pytest.raises(EmptyAudioError):
        provider.transcribe(b"", filename="recording.wav", content_type="audio/wav")
    with pytest.raises(UnsupportedAudioFormat):
        provider.transcribe(b"audio", filename="recording.ogg", content_type="audio/ogg")


def test_local_diagnostic_reports_selected_and_actual_device(monkeypatch):
    monkeypatch.setenv("SPEECH_TO_TEXT_PROVIDER", "local")
    monkeypatch.setenv("SPEECH_TO_TEXT_MODEL_SIZE", "base")
    monkeypatch.setenv("SPEECH_TO_TEXT_DEVICE", "cpu")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    diagnostic = speech_to_text.diagnose_transcription_configuration()

    assert diagnostic["provider"] == "local"
    assert diagnostic["model"] == "base"
    assert diagnostic["device"] == "cpu"
    assert diagnostic["actual_device"] == "not-initialized"
    assert diagnostic["api_key_configured"] is False
