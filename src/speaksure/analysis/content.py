"""Offline LLM-based answer scoring against interview question rubrics."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..interviewer import neutralize_markup
from ..llm_client import LLMClient, LLMError
from ..question_bank import BankQuestion
from ..session_log import Turn
from ..speech_to_text import TranscriptionResult
from .content_prompts import CONTENT_SYSTEM_PROMPT, CONTENT_USER_TEMPLATE
from .text_utils import normalize_token

# Question ids whose content ratings include the STAR component.
BEHAVIOURAL_QUESTION_IDS = ("teamwork", "conflict", "challenge", "mistake")
# Minimum normalized transcript token count eligible for LLM scoring.
MIN_WORDS_TO_RATE = 6
# Relative weights for content-score components before unavailable components are removed.
WEIGHTS = {
    "relevance": 0.25,
    "coverage": 0.30,
    "specificity": 0.20,
    "clarity": 0.15,
    "star": 0.10,
}
# Credit given to a covered hint whose model quote cannot be verified verbatim.
UNVERIFIED_EVIDENCE_CREDIT = 0.5
# Version for this content-analysis output shape.
CONTENT_ANALYZER_VERSION = 1
# Deterministic sampling temperature used for rubric evaluation.
CONTENT_TEMPERATURE = 0.0
# Maximum output tokens for the first JSON-mode content-analysis attempt.
CONTENT_MAX_TOKENS = 1500
# Number of additional attempts allowed after invalid model output.
CONTENT_RETRIES = 1
# Number of words allowed in each evidence quotation.
MAX_EVIDENCE_WORDS = 15
_STAR_KEYS = ("situation", "task", "action", "result")
_RESULT_KEYS = {
    "relevance",
    "specificity",
    "clarity",
    "coverage",
    "star",
    "strengths",
    "improvements",
    "reason",
}


def parse_content_result(
    raw_text: str, hints: Sequence[str], behavioural: bool
) -> dict[str, Any]:
    """Parse and strictly validate a model result, accepting optional JSON fences."""
    text = raw_text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence is not None:
        text = fence.group(1)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Content result is not valid JSON.") from error
    if not isinstance(parsed, dict):
        raise ValueError("Content result must be a JSON object.")
    if parsed.keys() != _RESULT_KEYS:
        raise ValueError("Content result contains unexpected fields.")
    missing = _RESULT_KEYS - parsed.keys()
    if missing:
        raise ValueError("Content result is missing required fields.")
    for key in ("relevance", "specificity", "clarity"):
        value = parsed[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{key} must be an integer from 1 to 5.")

    coverage = parsed["coverage"]
    if not isinstance(coverage, list) or len(coverage) != len(hints):
        raise ValueError("coverage must contain exactly one entry per hint.")
    validated_coverage = []
    for index, entry in enumerate(coverage):
        if not isinstance(entry, dict) or set(entry) != {"hint", "covered", "evidence"}:
            raise ValueError("Each coverage entry must contain hint, covered, and evidence.")
        if entry["hint"] != hints[index]:
            raise ValueError("Coverage hints must match the supplied hints in order.")
        if not isinstance(entry["covered"], bool):
            raise ValueError("Coverage covered values must be booleans.")
        if not isinstance(entry["evidence"], str):
            raise ValueError("Coverage evidence values must be strings.")
        if len(entry["evidence"].split()) > MAX_EVIDENCE_WORDS:
            raise ValueError("Coverage evidence must be at most 15 words.")
        if not entry["covered"] and entry["evidence"]:
            raise ValueError("Uncovered hints must have empty evidence.")
        validated_coverage.append(dict(entry))
    parsed["coverage"] = validated_coverage

    star = parsed["star"]
    if behavioural:
        if (
            not isinstance(star, dict)
            or set(star) != set(_STAR_KEYS)
            or any(not isinstance(star[key], bool) for key in _STAR_KEYS)
        ):
            raise ValueError("star must contain boolean situation, task, action, and result.")
    else:
        parsed["star"] = None

    for key in ("strengths", "improvements"):
        value = parsed[key]
        if (
            not isinstance(value, list)
            or len(value) > 2
            or any(not isinstance(item, str) for item in value)
        ):
            raise ValueError(f"{key} must be a list of at most two strings.")
    if not isinstance(parsed["reason"], str):
        raise ValueError("reason must be a string.")
    return parsed


def verify_evidence(evidence: str, answer_text: str) -> bool:
    """Return whether normalized evidence tokens occur contiguously in the answer."""
    evidence_tokens = _normalized_tokens(evidence)
    answer_tokens = _normalized_tokens(answer_text)
    if not evidence_tokens:
        return False
    width = len(evidence_tokens)
    return any(
        answer_tokens[index : index + width] == evidence_tokens
        for index in range(len(answer_tokens) - width + 1)
    )


def score_content(parsed: dict[str, Any], answer_text: str) -> dict[str, Any]:
    """Verify coverage evidence and compute normalized weighted content scores."""
    scored = dict(parsed)
    coverage_entries = []
    covered_credit = 0.0
    for entry in parsed["coverage"]:
        verified = bool(entry["covered"]) and verify_evidence(
            entry["evidence"], answer_text
        )
        coverage_entries.append({**entry, "evidence_verified": verified})
        if entry["covered"]:
            covered_credit += 1.0 if verified else UNVERIFIED_EVIDENCE_CREDIT
    scored["coverage"] = coverage_entries
    coverage_ratio = (
        covered_credit / len(coverage_entries) if coverage_entries else None
    )
    star = parsed["star"]
    star_ratio = (
        sum(bool(star[key]) for key in _STAR_KEYS) / len(_STAR_KEYS)
        if isinstance(star, dict)
        else None
    )
    components: dict[str, float | None] = {
        "relevance": (parsed["relevance"] - 1) / 4,
        "coverage": coverage_ratio,
        "specificity": (parsed["specificity"] - 1) / 4,
        "clarity": (parsed["clarity"] - 1) / 4,
        "star": star_ratio,
    }
    available = [
        (name, value)
        for name, value in components.items()
        if value is not None
    ]
    effective_weight = sum(WEIGHTS[name] for name, _ in available)
    content_score = (
        100.0
        * sum(WEIGHTS[name] * value for name, value in available)
        / effective_weight
        if effective_weight
        else None
    )
    scored.update(
        {
            "coverage_ratio": coverage_ratio,
            "star_ratio": star_ratio,
            "content_components": components,
            "content_score": content_score,
        }
    )
    return scored


def analyze_content(
    turn: Turn,
    bank_by_id: Mapping[str, BankQuestion],
    role: str,
    llm: LLMClient,
) -> dict[str, Any]:
    """Rate a turn against its planned question hints without exposing LLM errors."""
    interviewer = turn.analysis.get("interviewer", {})
    question_id = (
        interviewer.get("question_id") if isinstance(interviewer, dict) else None
    )
    question = bank_by_id.get(question_id) if isinstance(question_id, str) else None
    if question is None:
        return {
            "analyzer_version": CONTENT_ANALYZER_VERSION,
            "rated": False,
            "error": "unknown_question",
        }

    answer_text = _answer_text(turn.answer)
    if len(_normalized_tokens(answer_text)) < MIN_WORDS_TO_RATE:
        return {
            "analyzer_version": CONTENT_ANALYZER_VERSION,
            "rated": False,
            "reason": "too_short_or_empty",
        }

    behavioural = question.id in BEHAVIOURAL_QUESTION_IDS
    prompt_answer = neutralize_markup(answer_text)
    numbered_hints = "\n".join(
        f"{index}. {hint}" for index, hint in enumerate(question.rubric_hints, 1)
    )
    user_message = CONTENT_USER_TEMPLATE.format(
        role=neutralize_markup(role),
        question=neutralize_markup(turn.question),
        hints=numbered_hints,
        behavioural=str(behavioural).lower(),
        answer=prompt_answer,
    )
    messages = [
        {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    max_tokens = CONTENT_MAX_TOKENS
    last_category = "invalid_output"
    for attempt in range(CONTENT_RETRIES + 1):
        try:
            response = llm.complete(
                messages,
                temperature=CONTENT_TEMPERATURE,
                max_tokens=max_tokens,
                json_mode=True,
            )
        except LLMError as error:
            last_category = "llm_error"
            if attempt < CONTENT_RETRIES and "empty" in str(error).lower():
                max_tokens *= 2
                continue
            return {
                "analyzer_version": CONTENT_ANALYZER_VERSION,
                "rated": False,
                "error": last_category,
            }
        try:
            parsed = parse_content_result(
                response.text, question.rubric_hints, behavioural
            )
        except ValueError:
            last_category = "invalid_output"
            if attempt < CONTENT_RETRIES:
                continue
            return {
                "analyzer_version": CONTENT_ANALYZER_VERSION,
                "rated": False,
                "error": last_category,
            }
        scored = score_content(parsed, answer_text)
        return {
            "analyzer_version": CONTENT_ANALYZER_VERSION,
            "rated": True,
            "model": response.model,
            "question_id": question.id,
            **scored,
        }
    return {
        "analyzer_version": CONTENT_ANALYZER_VERSION,
        "rated": False,
        "error": last_category,
    }


def _answer_text(answer: TranscriptionResult | None) -> str:
    """Return the transcript text when a turn has an answer."""
    return answer.text if answer is not None and isinstance(answer.text, str) else ""


def _normalized_tokens(text: str) -> list[str]:
    """Return normalized non-empty whitespace-delimited tokens."""
    return [
        token
        for part in text.split()
        if (token := normalize_token(part))
    ]
