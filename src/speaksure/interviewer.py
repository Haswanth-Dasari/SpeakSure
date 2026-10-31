"""Decision logic for a safe, spoken HR mock interviewer."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Literal

from .interviewer_prompts import (
    DENIAL_NOTE,
    STYLE_NOTES,
    CLOSING_TEXT,
    FALLBACK_ACK,
    OPENING_TEMPLATE,
    OPENING_TEMPLATE_NAMELESS,
    NO_ANSWER_ACK,
    REPEAT_REQUEST,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
)
from .llm_client import LLMClient, LLMError
from .question_bank import BankQuestion

TurnKind = Literal["opening", "follow_up", "next_question", "repeat", "closing"]
DecisionAction = Literal["follow_up", "next_question"]
RECENT_EXCHANGES_LIMIT = 3
EARLIER_ANSWER_WORD_LIMIT = 60


_DENIAL_PATTERN = re.compile(
    r"\b(?:no|not any|never|none|nothing|haven't|have not|hadn't|didn't|did not|"
    r"don't|do not|can't think|cannot think)\b[^.?!]{0,50}\b(?:disagreements?|"
    r"conflicts?|arguments?|mistakes?|problems?|issues?|challenges?|difficult\w*|"
    r"weakness(?:es)?|failures?|struggles?|experience|time)\b",
    re.IGNORECASE,
)
_SHORT_NO_PATTERN = re.compile(r"^\s*(?:no|nope|not really|never)\b", re.IGNORECASE)
_PRESUPPOSING_PATTERN = re.compile(
    r"\bhow did you (?:resolve|solve|handle|overcome|deal with|fix|get past|work through)\b"
    r"|\bthose (?:disagreements?|conflicts?|problems?|issues?|challenges?|mistakes?)\b"
    r"|\bwhat did you learn from (?:that|those|it)\b",
    re.IGNORECASE,
)


def answer_denies_experience(answer: str) -> bool:
    """Return whether the answer says the asked-about situation did not happen."""
    if _DENIAL_PATTERN.search(answer):
        return True
    return len(answer.split()) <= 8 and bool(_SHORT_NO_PATTERN.match(answer))


def follow_up_presupposes(utterance: str) -> bool:
    """Return whether a follow-up assumes a situation the candidate has just denied."""
    return bool(_PRESUPPOSING_PATTERN.search(utterance))


@dataclass(frozen=True)
class InterviewPlan:
    """A fixed ordered set of interview questions and interviewer settings."""

    questions: tuple[BankQuestion, ...]
    role: str = "software engineering fresher"
    candidate_name: str = ""
    max_followups_per_question: int = 1
    # Short summary of the candidate's resume, used only to make follow-ups specific.
    background: str = ""
    # When true, the final planned question may also receive a follow-up.
    followup_on_last_question: bool = False
    # When true, extra style guidance and the "do not presuppose" guard are applied.
    natural_style: bool = False
    # Follow-up limit for resume project questions (a real interviewer digs deeper there).
    deep_dive_followups: int = 1

    def __post_init__(self) -> None:
        """Validate that the plan has questions and a supported follow-up limit."""
        if not self.questions:
            raise ValueError("An interview plan must include at least one question.")
        if self.max_followups_per_question not in {0, 1, 2}:
            raise ValueError("max_followups_per_question must be 0, 1, or 2.")
        if self.deep_dive_followups not in {0, 1, 2}:
            raise ValueError("deep_dive_followups must be 0, 1, or 2.")


@dataclass(frozen=True)
class InterviewerTurn:
    """One spoken interviewer response and its private logging metadata."""

    utterance: str
    kind: TurnKind
    question_id: str | None
    rationale: str
    llm_latency_seconds: float | None
    used_fallback: bool
    fallback_reason: str | None


class _InvalidModelResponse(ValueError):
    """Indicate that a model response did not match the expected JSON contract."""


class Interviewer:
    """Run an HR interview, using an LLM for follow-up and transition decisions."""

    def __init__(self, llm: LLMClient, plan: InterviewPlan) -> None:
        """Initialize question position, conversation history, and turn counters."""
        self._llm = llm
        self._plan = plan
        self._question_index = 0
        self._questions_asked = 0
        self._followups_used = 0
        self._repeat_used = False
        self._finished = False
        self._opened = False
        self._history: list[tuple[str, str]] = []
        self._current_question_text = plan.questions[0].text
        self._covered_topics: list[str] = []

    @property
    def is_finished(self) -> bool:
        """Return whether the final planned question has been answered."""
        return self._finished

    @property
    def questions_asked(self) -> int:
        """Return the number of distinct planned questions already presented."""
        return self._questions_asked

    @property
    def plan(self) -> InterviewPlan:
        """Return the immutable interview plan."""
        return self._plan

    def opening(self) -> InterviewerTurn:
        """Return the scripted greeting and first question without calling the LLM."""
        if self._opened:
            raise RuntimeError("The interview opening has already been delivered.")
        self._opened = True
        first_question = self._plan.questions[0]
        if self._plan.candidate_name:
            greeting = OPENING_TEMPLATE.format(
                name=self._plan.candidate_name, role=self._plan.role
            )
        else:
            greeting = OPENING_TEMPLATE_NAMELESS.format(role=self._plan.role)
        self._questions_asked = 1
        return InterviewerTurn(
            utterance=f"{greeting} {first_question.text}",
            kind="opening",
            question_id=first_question.id,
            rationale="Scripted interview opening.",
            llm_latency_seconds=None,
            used_fallback=False,
            fallback_reason=None,
        )

    def next_turn(self, answer_text: str) -> InterviewerTurn:
        """Choose a repeat, follow-up, next question, or closing response."""
        if not self._opened:
            raise RuntimeError("Call opening() before submitting an answer.")
        if self._finished:
            raise RuntimeError("The interview has already finished.")

        current_question = self._plan.questions[self._question_index]
        answer = answer_text.strip()
        if not answer and not self._repeat_used:
            self._repeat_used = True
            return InterviewerTurn(
                utterance=REPEAT_REQUEST,
                kind="repeat",
                question_id=current_question.id,
                rationale="The candidate response was empty.",
                llm_latency_seconds=None,
                used_fallback=False,
                fallback_reason=None,
            )

        if not answer and self._repeat_used:
            if self._question_index == len(self._plan.questions) - 1:
                self._record_completed_exchange(answer)
                self._finished = True
                return InterviewerTurn(
                    utterance=CLOSING_TEXT,
                    kind="closing",
                    question_id=current_question.id,
                    rationale="no answer after repeat",
                    llm_latency_seconds=None,
                    used_fallback=False,
                    fallback_reason=None,
                )
            next_question = self._plan.questions[self._question_index + 1]
            self._record_completed_exchange(answer)
            return self._move_to_next_question(
                next_question,
                utterance=f"{NO_ANSWER_ACK} {next_question.text}",
                rationale="no answer after repeat",
                latency=None,
                fallback_reason=None,
            )

        if self._question_index == len(self._plan.questions) - 1:
            if (
                self._plan.followup_on_last_question
                and self._followups_used < self._followup_limit(current_question)
            ):
                return self._decide_after_last_question(current_question, answer)
            self._record_completed_exchange(answer)
            self._finished = True
            return InterviewerTurn(
                utterance=CLOSING_TEXT,
                kind="closing",
                question_id=current_question.id,
                rationale="The final planned question has been answered.",
                llm_latency_seconds=None,
                used_fallback=False,
                fallback_reason=None,
            )

        next_question = self._plan.questions[self._question_index + 1]
        messages = self._make_messages(current_question, answer, next_question)
        self._record_completed_exchange(answer)
        result, decision, fallback_reason = self._get_decision(messages)
        if decision is None:
            return self._move_to_next_question(
                next_question,
                rationale="A safe next-question fallback was used.",
                latency=result,
                fallback_reason=fallback_reason or "invalid_response",
            )

        action, utterance, rationale = decision
        if (
            action == "follow_up"
            and self._plan.natural_style
            and answer_denies_experience(answer)
            and follow_up_presupposes(utterance)
        ):
            return self._move_to_next_question(
                next_question,
                rationale="The follow-up assumed something the candidate said did not happen.",
                latency=result,
                fallback_reason=None,
            )
        if action == "follow_up" and self._followups_used >= self._followup_limit(current_question):
            return self._move_to_next_question(
                next_question,
                rationale="The follow-up limit was reached.",
                latency=result,
                fallback_reason=None,
            )
        sanitized = sanitize_utterance(utterance)
        if not sanitized or len(sanitized.split()) > 60:
            return self._move_to_next_question(
                next_question,
                rationale="A safe next-question fallback was used.",
                latency=result,
                fallback_reason="invalid_utterance",
            )
        if action == "follow_up":
            self._followups_used += 1
            self._current_question_text = sanitized
            return InterviewerTurn(
                utterance=sanitized,
                kind="follow_up",
                question_id=current_question.id,
                rationale=rationale,
                llm_latency_seconds=result,
                used_fallback=False,
                fallback_reason=None,
            )
        return self._move_to_next_question(
            next_question,
            utterance=sanitized,
            rationale=rationale,
            latency=result,
            fallback_reason=None,
        )

    def _followup_limit(self, question: BankQuestion) -> int:
        """Return how many follow-ups this question may receive."""
        if question.id.startswith("resume_project"):
            return max(self._plan.max_followups_per_question, self._plan.deep_dive_followups)
        return self._plan.max_followups_per_question

    def _decide_after_last_question(
        self, current_question: BankQuestion, answer: str
    ) -> InterviewerTurn:
        """Ask one last follow-up if the model finds a worthwhile gap, else close."""
        final_marker = BankQuestion(
            "closing",
            "closing",
            "None. That was the final question. Choose next_question only if no "
            "follow-up is needed, and keep the utterance to a short thank-you.",
            (),
        )
        messages = self._make_messages(current_question, answer, final_marker)
        self._record_completed_exchange(answer)
        latency, decision, _reason = self._get_decision(messages)
        if decision is not None and decision[0] == "follow_up":
            sanitized = sanitize_utterance(decision[1])
            if sanitized and len(sanitized.split()) <= 60:
                self._followups_used += 1
                self._current_question_text = sanitized
                return InterviewerTurn(
                    utterance=sanitized,
                    kind="follow_up",
                    question_id=current_question.id,
                    rationale=decision[2],
                    llm_latency_seconds=latency,
                    used_fallback=False,
                    fallback_reason=None,
                )
        self._finished = True
        return InterviewerTurn(
            utterance=CLOSING_TEXT,
            kind="closing",
            question_id=current_question.id,
            rationale="The final planned question has been answered.",
            llm_latency_seconds=latency,
            used_fallback=False,
            fallback_reason=None,
        )

    def _make_messages(
        self,
        question: BankQuestion,
        answer: str,
        next_question: BankQuestion,
    ) -> list[dict[str, str]]:
        """Create the protected system and turn-specific model messages."""
        earlier_exchanges = self._history[-RECENT_EXCHANGES_LIMIT:]
        earlier_text = "none"
        if earlier_exchanges:
            earlier_text = "\n".join(
                f"Q: {neutralize_markup(previous_question)} "
                f"A: {_truncate_answer(neutralize_markup(previous_answer))}"
                for previous_question, previous_answer in earlier_exchanges
            )
        topics = ", ".join(self._covered_topics) or "none"
        user_message = USER_TEMPLATE.format(
            role=self._plan.role,
            candidate_name=self._plan.candidate_name or "not provided",
            current_question=neutralize_markup(self._current_question_text),
            hints="; ".join(question.rubric_hints),
            followups_remaining=max(
                0, self._followup_limit(question) - self._followups_used
            ),
            next_question=neutralize_markup(next_question.text),
            topics_covered=topics,
            earlier_exchanges=earlier_text,
            candidate_answer=neutralize_markup(answer),
        )
        if self._plan.natural_style:
            user_message += STYLE_NOTES
            if answer_denies_experience(answer):
                user_message += DENIAL_NOTE
        if self._plan.background:
            user_message += (
                "\nBackground from the candidate's resume (data, may be incomplete; "
                "use it only to make a follow-up specific, never read it aloud and "
                "never state facts that are not in it): "
                + neutralize_markup(self._plan.background)
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _get_decision(
        self, messages: list[dict[str, str]]
    ) -> tuple[float, tuple[DecisionAction, str, str] | None, str | None]:
        """Retry malformed or failed model calls once, returning safe failure categories."""
        total_latency = 0.0
        max_tokens = 500
        last_failure = "invalid_response"
        for attempt in range(2):
            started_at = time.perf_counter()
            try:
                result = self._llm.complete(
                    messages,
                    temperature=0.6,
                    max_tokens=max_tokens,
                    json_mode=True,
                )
            except LLMError as error:
                total_latency += time.perf_counter() - started_at
                last_failure = "llm_error"
                if attempt == 0 and "empty" in str(error).lower():
                    max_tokens *= 2
                continue
            total_latency += result.latency_seconds
            try:
                return total_latency, _parse_decision(result.text), None
            except _InvalidModelResponse:
                last_failure = "invalid_response"
        return total_latency, None, last_failure

    def _move_to_next_question(
        self,
        question: BankQuestion,
        *,
        utterance: str | None = None,
        rationale: str,
        latency: float | None,
        fallback_reason: str | None,
    ) -> InterviewerTurn:
        """Advance to the next planned question and reset its per-question counters."""
        self._question_index += 1
        self._questions_asked += 1
        self._followups_used = 0
        self._repeat_used = False
        self._current_question_text = sanitize_utterance(utterance or question.text)
        current_topic = self._plan.questions[self._question_index - 1].topic
        if current_topic not in self._covered_topics:
            self._covered_topics.append(current_topic)
        spoken_text = sanitize_utterance(utterance or question.text)
        used_fallback = fallback_reason is not None
        return InterviewerTurn(
            utterance=(
                f"{FALLBACK_ACK} {question.text}" if used_fallback else spoken_text
            ),
            kind="next_question",
            question_id=question.id,
            rationale=rationale,
            llm_latency_seconds=latency,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
        )


    def _record_completed_exchange(self, answer: str) -> None:
        """Remember the exact question asked and the unmodified candidate answer."""
        self._history.append((self._current_question_text, answer))


def _truncate_answer(answer: str) -> str:
    """Limit an earlier answer to sixty words and mark truncation visibly."""
    words = answer.split()
    if len(words) > EARLIER_ANSWER_WORD_LIMIT:
        return f"{' '.join(words[:EARLIER_ANSWER_WORD_LIMIT])} ..."
    return " ".join(words)


def neutralize_markup(text: str) -> str:
    """Replace angle brackets with spaces and collapse repeated whitespace."""
    return " ".join(text.replace("<", " ").replace(">", " ").split())


def _parse_decision(
    text: str,
) -> tuple[DecisionAction, str, str]:
    """Parse and validate the model's JSON action, utterance, and log rationale."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", cleaned, flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        raise _InvalidModelResponse("The model response was not valid JSON.") from None
    if not isinstance(payload, dict) or set(payload) != {
        "action",
        "utterance",
        "reason",
    }:
        raise _InvalidModelResponse("The model response did not match the JSON schema.")
    action = payload["action"]
    utterance = payload["utterance"]
    reason = payload["reason"]
    if not isinstance(action, str) or action not in {"follow_up", "next_question"}:
        raise _InvalidModelResponse("The model action was invalid.")
    if not isinstance(utterance, str) or not isinstance(reason, str):
        raise _InvalidModelResponse("The model response fields must be strings.")
    return action, utterance, reason


def sanitize_utterance(text: str) -> str:
    """Remove markdown, emojis, control characters, and excess whitespace."""
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    text = re.sub(r"(?m)^\s*(?:[-+*•‣]|\d+[.)])\s+", "", text)
    text = text.replace("*", "").replace("`", "")
    text = re.sub(
        r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF"
        r"\U0001F1E6-\U0001F1FF\uFE0F\u200D\u20E3]",
        "",
        text,
    )
    text = "".join(
        " "
        if unicodedata.category(character) in {"Cc", "Cs"}
        else character
        for character in text
    )
    text = " ".join(text.split()).strip()
    quote_pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
    for opening_quote, closing_quote in quote_pairs:
        if text.startswith(opening_quote) and text.endswith(closing_quote):
            text = text[len(opening_quote) : -len(closing_quote)].strip()
            break
    return text
