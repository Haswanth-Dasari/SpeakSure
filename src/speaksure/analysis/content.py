"""Offline LLM-based answer scoring against interview question rubrics."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

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
# Version for this per-question content-analysis output shape.
CONTENT_ANALYZER_VERSION = 2
# Deterministic sampling temperature used for rubric evaluation.
CONTENT_TEMPERATURE = 0.0
# Maximum output tokens for the first JSON-mode content-analysis attempt.
CONTENT_MAX_TOKENS = 1500
# Number of additional attempts allowed after invalid model output.
CONTENT_RETRIES = 1
# Maximum additional attempts after a rate limit or timeout.
TRANSIENT_RETRIES = 3
# Base delay for transient retries before jitter, in seconds.
RETRY_BASE_SECONDS = 2.0
# Maximum random jitter added to a retry delay, in seconds.
RETRY_JITTER_SECONDS = 0.25
# Maximum retry delay, in seconds.
RETRY_MAX_SECONDS = 20.0
# Number of words allowed in each evidence quotation.
MAX_EVIDENCE_WORDS = 15
# ASR confidence below this value is considered low.
ASR_LOW = 0.6
# ASR confidence at or above this value is considered high.
ASR_HIGH = 0.8
# Exact STAR fields required for behavioral-question output.
_STAR_KEYS = ("situation", "task", "action", "result")
# Exact top-level fields accepted in a model response.
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


LOGGER = logging.getLogger(__name__)


def repair_content_json(
    raw_text: str, hints: Sequence[str], behavioural: bool
) -> str:
    """Fix harmless formatting slips in a model reply before strict validation.

    Only presentation problems are repaired: extra keys, reworded hint labels,
    over-long evidence quotes, evidence on uncovered hints, integral floats and
    over-long advice lists. Scores, coverage decisions and quotes are never
    invented, and quoted evidence is still verified against the transcript.
    Anything else is returned unchanged so the strict parser rejects it.
    """
    text = raw_text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence is not None:
        text = fence.group(1)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return raw_text
    if not isinstance(parsed, dict):
        return raw_text
    if not behavioural:
        parsed.setdefault("star", None)
    if not _RESULT_KEYS <= parsed.keys():
        return raw_text
    parsed = {key: parsed[key] for key in _RESULT_KEYS}
    for key in ("relevance", "specificity", "clarity"):
        value = parsed[key]
        if isinstance(value, float) and value.is_integer():
            parsed[key] = int(value)
        elif isinstance(value, str) and value.strip().isdigit():
            parsed[key] = int(value.strip())
    coverage = parsed["coverage"]
    if isinstance(coverage, list) and len(coverage) == len(hints):
        fixed = []
        for hint, entry in zip(hints, coverage):
            if not isinstance(entry, dict) or not {"covered", "evidence"} <= entry.keys():
                return raw_text
            evidence = entry["evidence"] if isinstance(entry["evidence"], str) else ""
            covered = entry["covered"]
            if not covered:
                evidence = ""
            words = evidence.split()
            if len(words) > MAX_EVIDENCE_WORDS:
                evidence = " ".join(words[:MAX_EVIDENCE_WORDS])
            fixed.append({"hint": hint, "covered": covered, "evidence": evidence})
        parsed["coverage"] = fixed
    for key in ("strengths", "improvements"):
        if isinstance(parsed[key], list):
            parsed[key] = parsed[key][:2]
    if not behavioural:
        parsed["star"] = None
    return json.dumps(parsed)


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


MAX_RATING_SAMPLES = 5


def get_rating_samples() -> int:
    """Return how many independent ratings to take per question (default 1)."""
    raw = os.environ.get("CONTENT_RATING_SAMPLES", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 1
    return min(MAX_RATING_SAMPLES, max(1, value))


def select_median_result(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the median-scored rated result, annotated with the score spread.

    Language-model ratings differ between runs even at temperature zero, so the
    result whose score is the median is shown (keeping its quotes and advice
    consistent with its own scores) together with every sample score.
    """
    rated = [item for item in results if item.get("rated") is True]
    if not rated:
        return results[-1]
    ordered = sorted(rated, key=lambda item: float(item.get("content_score", 0.0)))
    chosen = dict(ordered[len(ordered) // 2])
    scores = [round(float(item.get("content_score", 0.0)), 1) for item in ordered]
    chosen["content_score_samples"] = scores
    chosen["content_score_spread"] = round(scores[-1] - scores[0], 1)
    if "attempts" in chosen:
        chosen["attempts"] = sum(int(item.get("attempts", 0)) for item in results)
    return chosen


def analyze_content(
    turns: Sequence[Turn],
    main_question_id: str | None,
    bank_by_id: Mapping[str, BankQuestion],
    role: str,
    llm: LLMClient,
    *,
    asr_mean_probability: float | None,
    asr_reliability: str,
    sleep: Callable[[float], None] = time.sleep,
    include_attempts: bool = False,
) -> dict[str, Any]:
    """Rate a question group, taking the median of CONTENT_RATING_SAMPLES ratings."""
    samples = get_rating_samples()
    kwargs: dict[str, Any] = {
        "asr_mean_probability": asr_mean_probability,
        "asr_reliability": asr_reliability,
        "sleep": sleep,
        "include_attempts": include_attempts,
    }
    first = _analyze_content_once(
        turns, main_question_id, bank_by_id, role, llm, **kwargs
    )
    if samples <= 1 or first.get("rated") is not True:
        return first
    results = [first]
    for _ in range(samples - 1):
        results.append(
            _analyze_content_once(
                turns, main_question_id, bank_by_id, role, llm, **kwargs
            )
        )
    return select_median_result(results)


def _analyze_content_once(
    turns: Sequence[Turn],
    main_question_id: str | None,
    bank_by_id: Mapping[str, BankQuestion],
    role: str,
    llm: LLMClient,
    *,
    asr_mean_probability: float | None,
    asr_reliability: str,
    sleep: Callable[[float], None] = time.sleep,
    include_attempts: bool = False,
) -> dict[str, Any]:
    """Rate a question group once with safe retries and optional attempt metadata."""

    def result_with_attempts(result: dict[str, Any], attempts: int) -> dict[str, Any]:
        """Attach attempt counts for session groups when explicitly requested."""
        if include_attempts:
            result["attempts"] = attempts
        return result

    question = bank_by_id.get(main_question_id) if main_question_id else None
    if question is None:
        return result_with_attempts(
            {
                "analyzer_version": CONTENT_ANALYZER_VERSION,
                "rated": False,
                "error": "unknown_question",
            },
            0,
        )

    pairs = [
        (turn.question, _answer_text(turn.answer))
        for turn in turns
    ]
    answer_text = " ".join(answer for _, answer in pairs).strip()
    if len(_normalized_tokens(answer_text)) < MIN_WORDS_TO_RATE:
        return result_with_attempts(
            {
                "analyzer_version": CONTENT_ANALYZER_VERSION,
                "rated": False,
                "reason": "too_short_or_empty",
            },
            0,
        )

    behavioural = question.id in BEHAVIOURAL_QUESTION_IDS
    numbered_hints = "\n".join(
        f"{index}. {hint}" for index, hint in enumerate(question.rubric_hints, 1)
    )
    exchange = "\n\n".join(
        f"Pair {index}:\nQuestion asked: {neutralize_markup(question_text)}\n"
        f"<candidate_answer>{neutralize_markup(answer)}</candidate_answer>"
        for index, (question_text, answer) in enumerate(pairs, 1)
    )
    user_message = CONTENT_USER_TEMPLATE.format(
        role=neutralize_markup(role),
        main_question=neutralize_markup(question.text),
        hints=numbered_hints,
        behavioural=str(behavioural).lower(),
        asr_mean_probability=(
            "unknown"
            if asr_mean_probability is None
            else f"{asr_mean_probability:.3f}"
        ),
        asr_reliability=asr_reliability.upper(),
        exchange=exchange,
    )
    messages = [
        {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    max_tokens = CONTENT_MAX_TOKENS
    last_category = "invalid_output"
    attempt = 0
    transient_retries = 0
    invalid_retries = 0
    while True:
        attempt += 1
        try:
            response = llm.complete(
                messages,
                temperature=CONTENT_TEMPERATURE,
                max_tokens=max_tokens,
                json_mode=True,
            )
        except LLMError as error:
            last_category = "llm_error"
            safe_message = str(error).lower()
            transient = "rate limited" in safe_message or "timed out" in safe_message
            if transient and transient_retries < TRANSIENT_RETRIES:
                delay = min(
                    RETRY_BASE_SECONDS * (2**transient_retries),
                    RETRY_MAX_SECONDS,
                )
                delay = min(
                    delay + random.uniform(0.0, RETRY_JITTER_SECONDS),
                    RETRY_MAX_SECONDS,
                )
                transient_retries += 1
                sleep(delay)
                continue
            if (
                not transient
                and "empty" in safe_message
                and invalid_retries < CONTENT_RETRIES
            ):
                max_tokens *= 2
                invalid_retries += 1
                continue
            return result_with_attempts(
                {
                    "analyzer_version": CONTENT_ANALYZER_VERSION,
                    "rated": False,
                    "error": last_category,
                },
                attempt,
            )
        try:
            parsed = parse_content_result(
                repair_content_json(
                    response.text, question.rubric_hints, behavioural
                ),
                question.rubric_hints,
                behavioural,
            )
        except ValueError as error:
            last_category = "invalid_output"
            LOGGER.warning(
                "Content rating rejected for question %s (attempt %d): %s",
                question.id,
                attempt,
                error,
            )
            if invalid_retries < CONTENT_RETRIES:
                invalid_retries += 1
                continue
            return result_with_attempts(
                {
                    "analyzer_version": CONTENT_ANALYZER_VERSION,
                    "rated": False,
                    "error": last_category,
                },
                attempt,
            )
        scored = score_content(parsed, answer_text)
        return result_with_attempts(
            {
                "analyzer_version": CONTENT_ANALYZER_VERSION,
                "rated": True,
                "model": response.model,
                "question_id": question.id,
                **scored,
            },
            attempt,
        )


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
