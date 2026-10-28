from pathlib import Path

import pytest

from src.speaksure import configuration
from src.speaksure.llm_client import GroqAnalysisLLMConfig, GroqLLMConfig
from src.speaksure.speech_to_text import (
    LocalTranscriptionConfig,
    MissingTranscriptionConfiguration,
    TranscriptionConfig,
    describe_stt_configuration,
    diagnose_transcription_configuration,
)
from src.speaksure.text_to_speech import SpeechSynthesisConfig


def test_root_dotenv_loads_when_working_directory_is_elsewhere(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    env_file = project_root / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=root-test-key\nOPENAI_TRANSCRIPTION_MODEL=root-model\n",
        encoding="utf-8",
    )
    outside_root = tmp_path / "elsewhere"
    outside_root.mkdir()
    monkeypatch.chdir(outside_root)
    monkeypatch.setattr(configuration, "ENV_FILE", env_file)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_MODEL", raising=False)

    configuration.load_environment()

    assert configuration.get_api_key() == "root-test-key"
    assert TranscriptionConfig.from_env().model == "root-model"


def test_operating_system_environment_takes_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=file-key\nOPENAI_TRANSCRIPTION_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "process-model")

    configuration.load_environment(env_file)

    assert configuration.get_api_key() == "process-key"
    assert TranscriptionConfig.from_env().model == "process-model"


@pytest.mark.parametrize(
    ("key", "default", "dotenv_value", "process_value", "get_value"),
    [
        (
            "SPEECH_TO_TEXT_MODEL_SIZE",
            "base",
            "small",
            "tiny",
            lambda: LocalTranscriptionConfig.from_env().model_size,
        ),
        (
            "TEXT_TO_SPEECH_VOICE",
            "en_US-lessac-medium",
            "dotenv-voice",
            "process-voice",
            lambda: SpeechSynthesisConfig.from_env().voice_name,
        ),
        (
            "LLM_MODEL",
            "openai/gpt-oss-20b",
            "dotenv-model",
            "process-model",
            lambda: GroqLLMConfig.from_env().model,
        ),
        (
            "LLM_ANALYSIS_MODEL",
            "openai/gpt-oss-120b",
            "dotenv-analysis-model",
            "process-analysis-model",
            lambda: GroqAnalysisLLMConfig.from_env().model,
        ),
    ],
)
def test_configuration_precedence_for_each_settings_family(
    key, default, dotenv_value, process_value, get_value, tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{key}={dotenv_value}\n", encoding="utf-8")
    monkeypatch.delenv(key, raising=False)

    assert get_value() == default

    configuration.load_environment(env_file)
    assert get_value() == dotenv_value

    monkeypatch.setenv(key, process_value)
    configuration.load_environment(env_file)
    assert get_value() == process_value


def test_environment_loading_is_idempotent_and_does_not_override_process_values(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    configuration.load_environment(env_file)
    assert configuration.get_llm_model() == "dotenv-model"

    monkeypatch.setenv("LLM_MODEL", "process-model")
    configuration.load_environment(env_file)
    configuration.load_environment(env_file)
    assert configuration.get_llm_model() == "process-model"


def test_stt_configuration_description_is_safe_and_contains_only_flags(monkeypatch):
    config = LocalTranscriptionConfig(
        model_size="small",
        device="cpu",
        hotwords="secret-hotwords",
        initial_prompt="secret-prompt",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-key")
    monkeypatch.setenv("GROQ_API_KEY", "secret-groq-key")

    description = describe_stt_configuration(config)

    assert description == {
        "provider": "local",
        "model": "small",
        "device_setting": "cpu",
        "language": "en",
        "vad_filter": True,
        "hotwords_set": True,
        "initial_prompt_set": True,
    }
    assert not any(
        secret in repr(description)
        for secret in ("secret-hotwords", "secret-prompt", "secret-api-key", "secret-groq-key")
    )


def test_transcribe_demo_prints_configured_model_and_actual_device(
    tmp_path, monkeypatch, capsys
):
    from scripts import transcribe_demo

    config = LocalTranscriptionConfig(model_size="tiny", device="cpu", language="en")
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"offline test audio")
    monkeypatch.setattr(
        transcribe_demo.LocalTranscriptionConfig,
        "from_env",
        classmethod(lambda cls: config),
    )

    class FakeProvider:
        def __init__(self, received_config):
            assert received_config is config

        def transcribe_detailed(self, *_args, **_kwargs):
            return type(
                "Result",
                (),
                {
                    "device": "cpu",
                    "language": "en",
                    "language_probability": 1.0,
                    "text": "offline transcript",
                    "words": (),
                },
            )()

    monkeypatch.setattr(transcribe_demo, "LocalSpeechToTextProvider", FakeProvider)
    monkeypatch.setattr(
        transcribe_demo.sys, "argv", ["transcribe_demo.py", str(wav_path)]
    )

    transcribe_demo.main()

    output = capsys.readouterr().out
    assert "STT: model=tiny device=cpu language=en" in output
    assert "Device: cpu" in output


@pytest.mark.parametrize("key", ["", "   "])
def test_missing_or_blank_key_is_rejected(key, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", key)

    with pytest.raises(MissingTranscriptionConfiguration, match="OPENAI_API_KEY"):
        TranscriptionConfig.from_env()


def test_transcription_model_defaults_and_supports_environment_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_MODEL", raising=False)
    assert TranscriptionConfig.from_env().model == "whisper-1"

    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "custom-model")
    assert TranscriptionConfig.from_env().model == "custom-model"


def test_chat_provider_uses_shared_key_and_model_configuration(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "shared-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "chat-test-model")

    assert configuration.get_api_key() == "shared-test-key"
    assert configuration.get_chat_model() == "chat-test-model"


def test_safe_diagnostic_does_not_return_key_and_reports_provider_configuration(
    monkeypatch,
):
    monkeypatch.setenv("SPEECH_TO_TEXT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "never-return-this-test-key")
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "safe-diagnostic-model")

    diagnostic = diagnose_transcription_configuration()

    assert diagnostic["root_env_exists"] == Path(configuration.ENV_FILE).is_file()
    assert diagnostic["api_key_configured"] is True
    assert diagnostic["model"] == "safe-diagnostic-model"
    assert isinstance(diagnostic["provider_sdk_installed"], bool)
    assert diagnostic["provider_configuration_matches"] is True
    assert "never-return-this-test-key" not in repr(diagnostic)


def test_safe_diagnostic_reports_missing_key_without_leaking(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    diagnostic = diagnose_transcription_configuration()

    assert diagnostic["api_key_configured"] is False
    assert diagnostic["provider_configuration_matches"] is True
