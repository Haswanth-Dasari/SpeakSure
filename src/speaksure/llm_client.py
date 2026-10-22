"""Provider-neutral language-model interface and Groq chat-completion client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Protocol

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)

from .configuration import (
    get_groq_api_key,
    get_llm_analysis_model,
    get_llm_analysis_reasoning_effort,
    get_llm_model,
    get_llm_reasoning_effort,
    get_llm_timeout_seconds,
)


class LLMError(RuntimeError):
    """A safe, actionable language-model error that never exposes credentials."""


@dataclass(frozen=True)
class LLMResult:
    """Text returned by a language model with timing and usage metadata."""

    text: str
    model: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None


class LLMClient(Protocol):
    """Interface implemented by language-model clients."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
        json_mode: bool = False,
    ) -> LLMResult:
        """Return one completed response for the supplied conversation."""
        ...

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> Iterator[str]:
        """Yield non-empty text deltas from the model response."""
        ...


@dataclass(frozen=True)
class GroqLLMConfig:
    """Validated credentials and request settings for the Groq LLM client."""

    api_key: str
    model: str
    reasoning_effort: str
    timeout_seconds: int

    def __post_init__(self) -> None:
        """Validate LLM settings without exposing the configured API key."""
        if self.reasoning_effort not in {"", "low", "medium", "high"}:
            raise ValueError(
                "LLM_REASONING_EFFORT must be one of: '', 'low', 'medium', or 'high'."
            )
        if not 1 <= self.timeout_seconds <= 120:
            raise ValueError("LLM_TIMEOUT_SECONDS must be from 1 to 120.")

    @classmethod
    def from_env(cls) -> GroqLLMConfig:
        """Read Groq LLM credentials and validated settings from the environment."""
        return cls(
            api_key=get_groq_api_key(),
            model=get_llm_model(),
            reasoning_effort=get_llm_reasoning_effort(),
            timeout_seconds=get_llm_timeout_seconds(),
        )


@dataclass(frozen=True)
class GroqAnalysisLLMConfig:
    """Validated credentials and dedicated settings for content-analysis calls."""

    api_key: str
    model: str
    reasoning_effort: str
    timeout_seconds: int

    def __post_init__(self) -> None:
        """Validate analysis-specific reasoning effort and shared timeout."""
        if self.reasoning_effort not in {"", "low", "medium", "high"}:
            raise ValueError(
                "LLM_ANALYSIS_REASONING_EFFORT must be one of: '', 'low', 'medium', or 'high'."
            )
        if not 1 <= self.timeout_seconds <= 120:
            raise ValueError("LLM_TIMEOUT_SECONDS must be from 1 to 120.")

    @classmethod
    def from_env(cls) -> GroqAnalysisLLMConfig:
        """Read the shared Groq key and timeout with analysis-specific settings."""
        return cls(
            api_key=get_groq_api_key(),
            model=get_llm_analysis_model(),
            reasoning_effort=get_llm_analysis_reasoning_effort(),
            timeout_seconds=get_llm_timeout_seconds(),
        )


class GroqChatClient:
    """Call Groq's OpenAI-compatible chat-completions API safely."""

    def __init__(
        self,
        api_key: str,
        model: str,
        reasoning_effort: str = "",
        timeout_seconds: int = 20,
        client: object | None = None,
    ) -> None:
        """Set client options, optionally accepting an injected test client."""
        self.api_key = api_key.strip()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _get_client(self) -> object:
        """Create and cache the OpenAI-compatible client only when needed."""
        if self._client is None:
            if not self.api_key:
                raise self._safe_error(
                    "Groq is not configured. Set GROQ_API_KEY in the project-root .env "
                    "file (not .env.example), then retry."
                )
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=self.timeout_seconds,
                max_retries=1,
            )
        return self._client

    def _check_key(self) -> None:
        """Reject missing credentials before creating a client or making a request."""
        if not self.api_key:
            raise self._safe_error(
                "Groq is not configured. Set GROQ_API_KEY in the project-root .env "
                "file (not .env.example), then retry."
            )

    def _safe_error(self, message: str) -> LLMError:
        """Redact the configured API key from any user-displayable error message."""
        if self.api_key:
            message = message.replace(self.api_key, "[redacted]")
        return LLMError(message)

    def _map_api_error(self, error: APIError) -> LLMError:
        """Map SDK failures to actionable messages without exposing headers or secrets."""
        if isinstance(error, AuthenticationError):
            return self._safe_error(
                "Groq authentication failed. Check that GROQ_API_KEY is valid."
            )
        if isinstance(error, RateLimitError):
            retry_after = error.response.headers.get("retry-after")
            message = "Groq's free tier is rate limited. Wait and retry."
            if retry_after:
                message += f" Retry after: {retry_after} seconds."
            return self._safe_error(message)
        if isinstance(error, APITimeoutError):
            return self._safe_error(
                "The Groq request timed out. Retry or increase LLM_TIMEOUT_SECONDS."
            )
        if isinstance(error, APIConnectionError):
            return self._safe_error(
                "Could not connect to Groq. Check your internet connection and retry."
            )
        if isinstance(error, NotFoundError) or (
            isinstance(error, APIStatusError) and error.status_code == 404
        ):
            return self._safe_error(
                f"Groq model '{self.model}' was not found. Check the model id with "
                "scripts/llm_demo.py --list-models."
            )
        if isinstance(error, APIStatusError):
            detail = self._error_detail(error)
            return self._safe_error(
                f"Groq API returned status {error.status_code}: {detail}"
            )
        return self._safe_error(
            "The Groq API request failed. Retry or check the provider status."
        )

    def _error_detail(self, error: APIStatusError) -> str:
        """Extract a provider error detail while redacting the configured API key."""
        body = error.body
        detail: object = None
        if isinstance(body, dict):
            error_body = body.get("error", body)
            if isinstance(error_body, dict):
                detail = error_body.get("message")
            elif isinstance(error_body, str):
                detail = error_body
        elif isinstance(body, str):
            detail = body
        if not isinstance(detail, str) or not detail.strip():
            return "The provider rejected the request."
        message = detail.strip()
        if self.api_key:
            message = message.replace(self.api_key, "[redacted]")
        return message

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
        json_mode: bool = False,
    ) -> LLMResult:
        """Return a completed response and usage metadata from Groq."""
        self._check_key()
        request: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            request["response_format"] = {"type": "json_object"}

        started_at = time.perf_counter()
        try:
            response = self._get_client().chat.completions.create(**request)
        except APIError as error:
            raise self._map_api_error(error) from None
        latency_seconds = time.perf_counter() - started_at
        choice = response.choices[0]
        content = choice.message.content
        if content is None or content == "":
            raise self._safe_error(
                "Groq returned an empty response. The model may have used the whole token "
                "budget for reasoning; raise max_tokens and retry."
            )
        usage = response.usage
        return LLMResult(
            text=content,
            model=response.model,
            latency_seconds=latency_seconds,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=(
                usage.completion_tokens if usage is not None else None
            ),
            finish_reason=choice.finish_reason,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> Iterator[str]:
        """Yield non-empty response deltas from a streaming Groq request."""
        self._check_key()
        request: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        try:
            chunks = self._get_client().chat.completions.create(**request)
            for chunk in chunks:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except APIError as error:
            raise self._map_api_error(error) from None

    def list_models(self) -> list[str]:
        """Return sorted model identifiers available to the configured Groq key."""
        self._check_key()
        try:
            response = self._get_client().models.list()
        except APIError as error:
            raise self._map_api_error(error) from None
        return sorted(model.id for model in response.data)


def build_llm_client(config: GroqLLMConfig) -> LLMClient:
    """Build the Groq LLM client from validated configuration values."""
    return GroqChatClient(
        api_key=config.api_key,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        timeout_seconds=config.timeout_seconds,
    )


def build_analysis_llm_client(config: GroqAnalysisLLMConfig) -> LLMClient:
    """Build a Groq client using dedicated offline analysis settings."""
    return GroqChatClient(
        api_key=config.api_key,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        timeout_seconds=config.timeout_seconds,
    )
