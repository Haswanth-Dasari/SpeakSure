"""Replaceable AI conversation provider for concise next-question generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from src.speaksure.configuration import get_api_key, get_chat_model


class ConversationProvider(Protocol):
    def next_question(
        self, *, practice_mode: str, difficulty: str, questions: Sequence[str], answers: Sequence[str]
    ) -> str: ...


class ProviderError(RuntimeError):
    """A safe, user-displayable provider error that never contains credentials."""


class MissingConfiguration(ProviderError):
    """Raised when the provider cannot be configured from the environment."""


@dataclass(frozen=True)
class ProviderConfig:
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        key = get_api_key()
        if not key:
            raise MissingConfiguration(
                "AI is not configured. Set OPENAI_API_KEY in the process environment or the "
                "project-root .env file (not .env.example), then restart SpeakSure."
            )
        return cls(key, get_chat_model())


def format_conversation_history(questions: Sequence[str], answers: Sequence[str]) -> list[dict[str, str]]:
    """Pair the available question and answer turns into a provider-neutral history."""
    return [
        {"question": question, "answer": answers[index]}
        for index, question in enumerate(questions)
        if index < len(answers)
    ]


def validate_response(response: object) -> str:
    """Accept a single short question; reject empty, oversized, or multi-question output."""
    if not isinstance(response, str):
        raise ProviderError("The AI returned an invalid response. Retry or continue with scripted prompts.")
    cleaned = " ".join(response.strip().split())
    if not cleaned or len(cleaned) > 400 or "?" not in cleaned or cleaned.count("?") != 1:
        raise ProviderError("The AI response was not one concise follow-up question. Retry or continue with scripted prompts.")
    return cleaned


class OpenAIConversationProvider:
    """OpenAI provider implementation; imports the SDK only when an API call is made."""

    def __init__(self, config: ProviderConfig | None = None, client_factory: Callable[..., object] | None = None) -> None:
        self.config = config or ProviderConfig.from_env()
        self._client_factory = client_factory

    def next_question(
        self, *, practice_mode: str, difficulty: str, questions: Sequence[str], answers: Sequence[str]
    ) -> str:
        history = format_conversation_history(questions, answers)
        latest = answers[-1] if answers else ""
        previous = "\n".join(
            f"Q: {turn['question']}\nA: {turn['answer']}" for turn in history[:-1]
        ) or "No earlier turns."
        role = "a professional interviewer" if practice_mode == "HR interview" else "a friendly conversation partner"
        system = (
            f"You are {role} conducting English practice in {practice_mode} mode at {difficulty} difficulty. "
            "Ask exactly one relevant, concise follow-up question based on the latest answer. "
            "Use language appropriate to the difficulty and setting. Do not answer, explain, list questions, "
            "or continue the interview beyond this single next question. Output only that question."
        )
        user = f"Earlier conversation:\n{previous}\n\nLatest question: {questions[-1] if questions else 'Opening prompt'}\nLatest answer: {latest}"
        try:
            if self._client_factory is None:
                from openai import OpenAI

                client_factory = OpenAI
            else:
                client_factory = self._client_factory
            client = client_factory(api_key=self.config.api_key)
            result = client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=100,
                temperature=0.6,
            )
            content = result.choices[0].message.content
            return validate_response(content)
        except ProviderError:
            raise
        except Exception as exc:
            # Never include SDK exception text: it may contain request headers or credentials.
            raise ProviderError("The AI request failed. Retry, or continue with scripted prompts.") from None
