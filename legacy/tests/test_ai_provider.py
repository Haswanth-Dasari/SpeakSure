from types import SimpleNamespace

import pytest

from legacy.ai_provider import (
    MissingConfiguration, OpenAIConversationProvider, ProviderConfig, ProviderError,
    format_conversation_history, validate_response,
)


def test_history_pairs_previous_questions_and_answers_in_order():
    assert format_conversation_history(["Q1", "Q2", "Q3"], ["A1", "A2"]) == [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]


@pytest.mark.parametrize("value", ["", "  ", "Tell me more.", "Q1? Q2?", "?" * 401, None, 42])
def test_response_validation_rejects_invalid_responses(value):
    with pytest.raises(ProviderError):
        validate_response(value)


def test_response_validation_normalizes_a_single_question():
    assert validate_response("  What   did you learn?  ") == "What did you learn?"


def test_missing_api_key_is_reported_without_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingConfiguration, match="OPENAI_API_KEY"):
        ProviderConfig.from_env()


def test_api_failure_is_sanitized(monkeypatch):
    class BrokenClient:
        def __init__(self, **kwargs):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("failed with sk-secret-value")

    provider = OpenAIConversationProvider(
        ProviderConfig("sk-secret-value", "test-model"), client_factory=BrokenClient
    )
    with pytest.raises(ProviderError) as error:
        provider.next_question(
            practice_mode="HR interview", difficulty="Intermediate",
            questions=["Tell me about yourself?"], answers=["I lead teams."],
        )
    assert "sk-secret-value" not in str(error.value)
    assert "Retry" in str(error.value)


def test_provider_request_includes_latest_answer():
    captured = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    message = SimpleNamespace(content="What happened next?")
                    choice = SimpleNamespace(message=message)
                    return SimpleNamespace(choices=[choice])

    provider = OpenAIConversationProvider(
        ProviderConfig("test-key", "test-model"), client_factory=RecordingClient
    )
    provider.next_question(
        practice_mode="HR interview",
        difficulty="Intermediate",
        questions=["Opening question", "Follow-up question"],
        answers=["Earlier answer", "The latest submitted answer."],
    )

    user_message = captured["messages"][1]["content"]
    assert "The latest submitted answer." in user_message
    assert "Earlier answer" in user_message
