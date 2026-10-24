"""Token and sentence operations for transcript heuristics."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence


def normalize_token(text: str) -> str:
    """Lowercase a token and strip edge punctuation, retaining inner apostrophes and hyphens."""
    normalized = text.lower()
    start = 0
    end = len(normalized)
    while start < end and unicodedata.category(normalized[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(normalized[end - 1]).startswith("P"):
        end -= 1
    return normalized[start:end]


def count_phrases(
    tokens: Sequence[str], phrases: Iterable[Sequence[str]]
) -> int:
    """Count phrase matches left-to-right without counting overlapping tokens twice."""
    normalized_phrases = sorted(
        {
            tuple(normalize_token(token) for token in phrase if normalize_token(token))
            for phrase in phrases
        },
        key=lambda phrase: (-len(phrase), phrase),
    )
    normalized_phrases = [phrase for phrase in normalized_phrases if phrase]
    normalized_tokens = [normalize_token(token) for token in tokens]
    matches = 0
    index = 0
    while index < len(normalized_tokens):
        matching_phrase = next(
            (
                phrase
                for phrase in normalized_phrases
                if tuple(normalized_tokens[index : index + len(phrase)]) == phrase
            ),
            None,
        )
        if matching_phrase is None:
            index += 1
        else:
            matches += 1
            index += len(matching_phrase)
    return matches


def split_sentences(text: str) -> list[str]:
    """Split text at periods, question marks, and exclamation marks, omitting empties."""
    return [piece.strip() for piece in re.split(r"[.?!]+", text) if piece.strip()]
