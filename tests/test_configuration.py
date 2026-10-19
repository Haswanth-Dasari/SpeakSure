from pathlib import Path

import pytest

from src.speaksure import configuration
from src.speaksure.ai_provider import ProviderConfig
from src.speaksure.speech_to_text import (
    MissingTranscriptionConfiguration,
    TranscriptionConfig,
    diagnose_transcription_configuration,
)


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
    config = ProviderConfig.from_env()

    assert config.api_key == "shared-test-key"
    assert config.model == "chat-test-model"


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
