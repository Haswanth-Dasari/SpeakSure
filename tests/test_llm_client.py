from types import SimpleNamespace

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)
from openai._exceptions import httpx2 as httpx

from src.speaksure import configuration
from src.speaksure.llm_client import (
    GroqChatClient,
    GroqLLMConfig,
    LLMError,
    LLMResult,
    build_llm_client,
)


API_KEY = "test-secret-groq-key"
MODEL = "test-chat-model"
MESSAGES = [{"role": "user", "content": "Say hello."}]


def response(content="Hello!", model=MODEL, prompt_tokens=12, completion_tokens=4, finish_reason="stop"):
    """Build a completion response object for client tests."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


class FakeClient:
    """Record chat and model-list requests without making network calls."""

    def __init__(self, result=None, error=None, chunks=None, models=None):
        """Set fake response behavior and initialize request records."""
        self.result = result if result is not None else response()
        self.error = error
        self.chunks = chunks or []
        self.available_models = models or []
        self.requests = []
        self.model_list_calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )
        self.models = SimpleNamespace(list=self.list_models)

    def create_completion(self, **kwargs):
        """Record a completion request and return or raise its configured result."""
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return iter(self.chunks)
        return self.result

    def list_models(self):
        """Record and return the configured model list."""
        self.model_list_calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            data=[SimpleNamespace(id=item) for item in self.available_models]
        )


def make_client(fake, *, key=API_KEY, reasoning_effort="low"):
    """Build a Groq client around a network-free fake."""
    return GroqChatClient(
        api_key=key,
        model=MODEL,
        reasoning_effort=reasoning_effort,
        timeout_seconds=20,
        client=fake,
    )


def status_error(error_type, status, message, *, headers=None):
    """Create an OpenAI status exception with a real httpx response and request."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        status,
        headers=headers or {},
        json={"error": {"message": message}},
        request=request,
    )
    return error_type(message, response=response, body={"error": {"message": message}})


def test_complete_maps_text_model_usage_and_finish_reason():
    fake = FakeClient(result=response(model="reported-model"))
    result = make_client(fake).complete(MESSAGES)

    assert result == LLMResult(
        text="Hello!",
        model="reported-model",
        latency_seconds=result.latency_seconds,
        prompt_tokens=12,
        completion_tokens=4,
        finish_reason="stop",
    )
    assert result.latency_seconds >= 0
    assert fake.requests[0]["max_completion_tokens"] == 400
    assert fake.requests[0]["temperature"] == 0.7


@pytest.mark.parametrize("content", ["", None])
def test_empty_content_raises_actionable_llm_error(content):
    client = make_client(FakeClient(result=response(content=content)))

    with pytest.raises(LLMError, match="whole token budget for reasoning") as error:
        client.complete(MESSAGES)

    assert "raise max_tokens" in str(error.value)


@pytest.mark.parametrize(
    ("reasoning_effort", "included"),
    [("", False), ("medium", True)],
)
def test_reasoning_effort_is_sent_only_when_non_empty(reasoning_effort, included):
    fake = FakeClient()

    make_client(fake, reasoning_effort=reasoning_effort).complete(MESSAGES)

    assert ("reasoning_effort" in fake.requests[0]) is included
    if included:
        assert fake.requests[0]["reasoning_effort"] == "medium"


def test_json_mode_adds_json_object_response_format():
    fake = FakeClient()

    make_client(fake).complete(MESSAGES, json_mode=True)

    assert fake.requests[0]["response_format"] == {"type": "json_object"}


def test_stream_yields_only_non_empty_text_deltas():
    fake = FakeClient(
        chunks=[
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=""))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" there"))]),
        ]
    )

    assert list(make_client(fake).stream(MESSAGES)) == ["Hi", " there"]
    assert fake.requests[0]["stream"] is True


def test_authentication_failure_is_mapped_without_leaking_key():
    fake = FakeClient(error=status_error(AuthenticationError, 401, "bad credential"))

    with pytest.raises(LLMError, match="authentication failed") as error:
        make_client(fake).complete(MESSAGES)

    assert "valid" in str(error.value)
    assert API_KEY not in str(error.value)


@pytest.mark.parametrize(
    ("headers", "retry_message"),
    [({}, False), ({"retry-after": "17"}, True)],
)
def test_rate_limit_message_includes_retry_after_when_present(headers, retry_message):
    fake = FakeClient(
        error=status_error(RateLimitError, 429, "limited", headers=headers)
    )

    with pytest.raises(LLMError, match="free tier is rate limited") as error:
        make_client(fake).complete(MESSAGES)

    assert ("17 seconds" in str(error.value)) is retry_message
    assert API_KEY not in str(error.value)


def test_timeout_error_is_mapped_without_leaking_key():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    fake = FakeClient(error=APITimeoutError(request=request))

    with pytest.raises(LLMError, match="timed out") as error:
        make_client(fake).complete(MESSAGES)

    assert API_KEY not in str(error.value)


def test_connection_error_is_mapped_without_leaking_key():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    fake = FakeClient(error=APIConnectionError(request=request))

    with pytest.raises(LLMError, match="Could not connect") as error:
        make_client(fake).complete(MESSAGES)

    assert API_KEY not in str(error.value)


def test_unknown_model_error_suggests_model_listing_without_leaking_key():
    fake = FakeClient(error=status_error(NotFoundError, 404, "not found"))

    with pytest.raises(LLMError, match="test-chat-model") as error:
        make_client(fake).complete(MESSAGES)

    assert "scripts/llm_demo.py --list-models" in str(error.value)
    assert API_KEY not in str(error.value)


def test_other_status_error_includes_status_and_exact_provider_detail():
    fake = FakeClient(
        error=status_error(APIStatusError, 400, "Unsupported response format")
    )

    with pytest.raises(LLMError, match="status 400") as error:
        make_client(fake).complete(MESSAGES)

    assert "Unsupported response format" in str(error.value)
    assert API_KEY not in str(error.value)


def test_other_status_error_redacts_key_in_provider_detail():
    fake = FakeClient(
        error=status_error(APIStatusError, 400, f"Rejected {API_KEY}")
    )

    with pytest.raises(LLMError) as error:
        make_client(fake).complete(MESSAGES)

    assert API_KEY not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_missing_key_fails_before_fake_client_is_called():
    fake = FakeClient()
    client = make_client(fake, key="")

    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        client.complete(MESSAGES)

    assert fake.requests == []
    assert fake.model_list_calls == 0


def test_list_models_returns_sorted_ids():
    fake = FakeClient(models=["z-model", "a-model", "m-model"])

    assert make_client(fake).list_models() == ["a-model", "m-model", "z-model"]
    assert fake.model_list_calls == 1


def test_llm_settings_use_defaults_and_validate_environment(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)

    assert configuration.get_groq_api_key() == ""
    assert configuration.get_llm_model() == "openai/gpt-oss-20b"
    assert configuration.get_llm_reasoning_effort() == "low"
    assert configuration.get_llm_timeout_seconds() == 20

    monkeypatch.setenv("LLM_REASONING_EFFORT", "")
    assert configuration.get_llm_reasoning_effort() == ""


@pytest.mark.parametrize("timeout", ["0", "121", "invalid"])
def test_invalid_llm_timeout_is_rejected(timeout, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", timeout)

    with pytest.raises(ValueError, match="LLM_TIMEOUT_SECONDS"):
        configuration.get_llm_timeout_seconds()


def test_invalid_reasoning_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "extreme")

    with pytest.raises(ValueError, match="LLM_REASONING_EFFORT"):
        configuration.get_llm_reasoning_effort()


def test_build_llm_client_uses_configured_values(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", API_KEY)
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "high")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "45")

    config = GroqLLMConfig.from_env()
    client = build_llm_client(config)

    assert isinstance(client, GroqChatClient)
    assert client.api_key == API_KEY
    assert client.model == "configured-model"
    assert client.reasoning_effort == "high"
    assert client.timeout_seconds == 45
