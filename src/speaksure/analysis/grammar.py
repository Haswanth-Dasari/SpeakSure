"""Offline, spoken-language grammar heuristics for one transcript turn."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import re
from typing import Any

from ..speech_to_text import TranscriptionResult
from .lexicons import HESITATION_FILLERS
from .text_utils import normalize_token
from .transcript_quality import LOW_CONFIDENCE

MIN_WORDS_TO_CHECK = 5
GRAMMAR_ANALYZER_VERSION = 1
_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass"})
_MASS_NOUNS = frozenset(
    {
        "advice",
        "equipment",
        "feedback",
        "homework",
        "information",
        "knowledge",
        "music",
        "research",
        "software",
        "traffic",
        "water",
        "work",
        # Abstract nouns that are normally used without an article in
        # interview answers ("I handle pressure", "I have experience").
        "accountability",
        "adaptability",
        "anxiety",
        "attention",
        "code",
        "collaboration",
        "commitment",
        "communication",
        "confidence",
        "consistency",
        "creativity",
        "curiosity",
        "data",
        "debugging",
        "dedication",
        "development",
        "discipline",
        "education",
        "effort",
        "empathy",
        "energy",
        "engineering",
        "enthusiasm",
        "experience",
        "flexibility",
        "focus",
        "growth",
        "guidance",
        "hardware",
        "help",
        "honesty",
        "integrity",
        "intelligence",
        "interest",
        "leadership",
        "learning",
        "management",
        "mentorship",
        "money",
        "motivation",
        "networking",
        "ownership",
        "passion",
        "patience",
        "persistence",
        "planning",
        "practice",
        "pressure",
        "programming",
        "progress",
        "resilience",
        "science",
        "stress",
        "success",
        "support",
        "teamwork",
        "technology",
        "testing",
        "time",
        "training",
    }
)
_PAST_TIME_WORDS = frozenset(
    {"day", "days", "month", "months", "night", "nights", "week", "weeks", "year", "years"}
)
_PRESENT_HABIT_WORDS = frozenset(
    {"always", "generally", "often", "usually", "normally", "routinely"}
)
_IRREGULAR_PAST = {
    "be": "was",
    "begin": "began",
    "bring": "brought",
    "buy": "bought",
    "come": "came",
    "do": "did",
    "eat": "ate",
    "get": "got",
    "give": "gave",
    "go": "went",
    "have": "had",
    "know": "knew",
    "make": "made",
    "meet": "met",
    "run": "ran",
    "say": "said",
    "see": "saw",
    "take": "took",
    "think": "thought",
    "write": "wrote",
}
_IRREGULAR_PRESENT_THIRD = {
    "be": "is",
    "do": "does",
    "have": "has",
}


@dataclass(frozen=True)
class _TranscriptWord:
    text: str
    normalized: str
    start: int
    end: int
    index: int
    probability: float | None


def analyze_grammar(answer: TranscriptionResult | None) -> dict[str, object]:
    """Return a JSON-compatible grammar result for one spoken answer."""
    text = _answer_text(answer)
    all_words = _words_with_probabilities(text, answer)
    parse_words = _grammar_tokens(all_words)
    checked_words = [
        word
        for word in parse_words
        if word.probability is None or word.probability >= LOW_CONFIDENCE
    ]
    word_count = len(checked_words)
    if word_count < MIN_WORDS_TO_CHECK:
        return _result("too_short", word_count)

    checked_text = list(text)
    parse_indexes = {word.index for word in parse_words}
    for word in all_words:
        if word.index not in parse_indexes:
            checked_text[word.start : word.end] = " " * (word.end - word.start)

    try:
        pipeline = _load_pipeline()
    except ImportError:
        return _result(
            "grammar_unavailable",
            word_count,
            message=(
                "Install the SpeakSure requirements and the en_core_web_sm "
                "English model to enable offline grammar checks."
            ),
        )
    except OSError:
        return _result(
            "grammar_unavailable",
            word_count,
            message=(
                "The en_core_web_sm English model is unavailable; install it "
                "to enable offline grammar checks."
            ),
        )

    doc = pipeline("".join(checked_text))
    words_by_index = {word.index: word for word in all_words}
    token_word_indexes = {
        token.i: word.index
        for token in doc
        if not token.is_space
        and (word := _word_at_offset(words_by_index.values(), token.idx)) is not None
    }
    checked_word_indexes = {word.index for word in checked_words}
    issues = _find_issues(
        doc, token_word_indexes, words_by_index, checked_word_indexes
    )
    issues.sort(key=lambda issue: (int(issue["word_index"]), str(issue["category"])))
    error_count = len(issues)
    return {
        "analyzer_version": GRAMMAR_ANALYZER_VERSION,
        "status": "ok",
        "word_count_checked": word_count,
        "error_count": error_count,
        "errors_per_100_words": round(error_count * 100 / word_count, 2),
        "issues": issues,
    }


@lru_cache(maxsize=1)
def _load_pipeline() -> Any:
    """Load the small English parser without downloading anything at runtime."""
    import spacy

    return spacy.load("en_core_web_sm")


def _result(
    status: str, word_count: int, *, message: str | None = None
) -> dict[str, object]:
    result: dict[str, object] = {
        "analyzer_version": GRAMMAR_ANALYZER_VERSION,
        "status": status,
        "word_count_checked": word_count,
        "error_count": 0,
        "errors_per_100_words": None,
        "issues": [],
    }
    if message is not None:
        result["message"] = message
    return result


def _answer_text(answer: TranscriptionResult | None) -> str:
    """Return transcript text when available."""
    if answer is None:
        return ""
    value = getattr(answer, "text", "")
    return value if isinstance(value, str) else ""


def _all_words(text: str) -> list[_TranscriptWord]:
    return [
        _TranscriptWord(
            text=match.group(),
            normalized=normalize_token(match.group()),
            start=match.start(),
            end=match.end(),
            index=index,
            probability=None,
        )
        for index, match in enumerate(_WORD_PATTERN.finditer(text))
    ]


def _words_with_probabilities(
    text: str, answer: TranscriptionResult | None
) -> list[_TranscriptWord]:
    words = _all_words(text)
    probabilities = _word_probabilities(words, answer)
    return [
        _TranscriptWord(
            word.text,
            word.normalized,
            word.start,
            word.end,
            word.index,
            probability,
        )
        for word, probability in zip(words, probabilities)
    ]


def _grammar_tokens(words: list[_TranscriptWord]) -> list[_TranscriptWord]:
    """Remove fillers and immediate repeats while retaining uncertain parse context."""
    retained: list[_TranscriptWord] = []
    for word in words:
        if word.normalized in HESITATION_FILLERS:
            continue
        if retained and retained[-1].normalized == word.normalized:
            continue
        retained.append(word)
    return retained


def _word_probabilities(
    words: list[_TranscriptWord], answer: TranscriptionResult | None
) -> list[float | None]:
    probabilities: list[float | None] = [None] * len(words)
    if answer is None:
        return probabilities
    try:
        metadata_words = answer.words or ()
    except (AttributeError, TypeError):
        return probabilities

    cursor = 0
    for metadata in metadata_words:
        metadata_text = getattr(metadata, "text", "")
        probability = _finite_probability(getattr(metadata, "probability", None))
        if not isinstance(metadata_text, str):
            continue
        for match in _WORD_PATTERN.finditer(metadata_text):
            normalized = normalize_token(match.group())
            for index in range(cursor, len(words)):
                if words[index].normalized == normalized:
                    probabilities[index] = probability
                    cursor = index + 1
                    break
    return probabilities


def _finite_probability(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _word_at_offset(
    words: Any, offset: int
) -> _TranscriptWord | None:
    return next((word for word in words if word.start <= offset < word.end), None)


def _find_issues(
    doc: Any,
    token_word_indexes: dict[int, int],
    words_by_index: dict[int, _TranscriptWord],
    checked_word_indexes: set[int],
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()

    def add(
        category: str,
        span_text: str,
        suggestion: str,
        explanation: str,
        word_index: int,
    ) -> None:
        key = (category, word_index)
        if key in seen:
            return
        seen.add(key)
        issues.append(
            {
                "category": category,
                "span_text": span_text,
                "suggestion": suggestion,
                "explanation": explanation,
                "word_index": word_index + 1,
            }
        )

    for subject in doc:
        if subject.dep_ not in _SUBJECT_DEPS:
            continue
        subject_word_index = token_word_indexes.get(subject.i)
        if subject_word_index not in checked_word_indexes:
            continue
        predicate = subject.head
        verbs = [
            token
            for token in [predicate, *predicate.children]
            if token.pos_ in {"AUX", "VERB"}
            and (
                token == predicate
                or token.dep_ in {"aux", "auxpass", "cop"}
            )
        ]
        for verb in verbs:
            if (word_index := token_word_indexes.get(verb.i)) is None:
                continue
            if word_index not in checked_word_indexes:
                continue
            source_word = words_by_index.get(word_index)
            if source_word is None:
                continue
            expected = _expected_agreement_form(subject, verb)
            if expected is not None and expected.casefold() != verb.text.casefold():
                add(
                    "subject_verb_agreement",
                    source_word.text,
                    expected,
                    "The verb form does not agree with its subject.",
                    word_index,
                )

            tense_suggestion = _tense_suggestion(
                verb,
                subject,
                token_word_indexes,
                words_by_index,
                checked_word_indexes,
            )
            if tense_suggestion is not None:
                add(
                    "verb_tense",
                    source_word.text,
                    tense_suggestion,
                    "The time expression suggests a different simple tense.",
                    word_index,
                )

    for noun in doc:
        if noun.pos_ not in {"NOUN"} or noun.tag_ == "NNS":
            continue
        if noun.lemma_.casefold() in _MASS_NOUNS:
            continue
        if noun.dep_ == "pobj" and noun.head.text.casefold() == "during":
            continue
        # Fixed phrases: "for example", "for instance" are correct as spoken.
        if (
            noun.lemma_.casefold() in {"example", "instance"}
            and noun.dep_ == "pobj"
            and noun.head.text.casefold() == "for"
        ):
            continue
        if noun.dep_ not in {
            "attr",
            "dobj",
            "obj",
            "oprd",
            "pobj",
            "nsubj",
            "nsubjpass",
        } and not (
            noun.dep_ == "ROOT"
            and any(
                child.dep_ == "cop" or child.text.casefold() in {"am", "is", "are", "was", "were"}
                for child in noun.children
            )
        ):
            continue
        word_index = token_word_indexes.get(noun.i)
        source_word = words_by_index.get(word_index) if word_index is not None else None
        if source_word is None:
            continue
        determiners = [child for child in noun.children if child.dep_ == "det"]
        if determiners:
            if any(
                token_word_indexes.get(determiner.i) not in checked_word_indexes
                for determiner in determiners
            ):
                continue
            determiner = determiners[0]
            if determiner.text.casefold() in {"a", "an"}:
                expected = _article_for(noun.text)
                if determiner.text.casefold() != expected:
                    add(
                        "article",
                        source_word.text,
                        f"{expected} {source_word.text}",
                        "Use 'a' before a consonant sound and 'an' before a vowel sound.",
                        word_index,
                    )
            continue
        if any(
            child.pos_ in {"PROPN", "NUM"}
            and child.dep_ in {"compound", "nmod", "nummod"}
            for child in noun.children
        ):
            continue
        if any(child.dep_ in {"poss", "nmod:poss"} for child in noun.children):
            continue
        if noun.morph.get("Number") and "Sing" not in noun.morph.get("Number"):
            continue
        add(
            "article",
            source_word.text,
            f"{_article_for(noun.text)} {source_word.text}",
            "A singular count noun generally needs an article in this context.",
            word_index,
        )
    return issues


def _expected_agreement_form(subject: Any, verb: Any) -> str | None:
    text = verb.text.casefold()
    subject_text = subject.text.casefold()
    third_singular = _is_third_person_singular(subject)
    lemma = verb.lemma_.casefold()
    if lemma == "be":
        if text in {"'m", "’m"} and subject_text == "i":
            return verb.text
        if text in {"'re", "’re"} and not _is_singular_subject(subject):
            return verb.text
        if text in {"'s", "’s"} and third_singular:
            return verb.text
    if text in {"am", "is", "are"}:
        if subject_text == "i":
            return "am"
        return "is" if third_singular else "are"
    if text in {"was", "were"}:
        return "was" if _is_singular_subject(subject) else "were"
    if lemma in {"be", "do", "have"} and verb.tag_ in {"VB", "VBP", "VBZ"}:
        if lemma == "be":
            return "am" if subject_text == "i" else "is" if third_singular else "are"
        if lemma == "do":
            return "does" if third_singular else "do"
        return "has" if third_singular else "have"
    if verb.tag_ not in {"VBP", "VBZ"}:
        return None
    expected_third = _present_form(lemma, True)
    expected_other = _present_form(lemma, False)
    return expected_third if third_singular else expected_other


def _is_third_person_singular(subject: Any) -> bool:
    text = subject.text.casefold()
    if text in {"he", "she", "it"}:
        return True
    if text in {"i", "you", "we", "they"}:
        return False
    number = subject.morph.get("Number")
    if "Plur" in number:
        return False
    return subject.pos_ in {"NOUN", "PROPN"} and subject.tag_ not in {"NNS", "NNPS"}


def _is_singular_subject(subject: Any) -> bool:
    text = subject.text.casefold()
    if text in {"you", "we", "they"}:
        return False
    if text in {"i", "he", "she", "it"}:
        return True
    number = subject.morph.get("Number")
    if "Plur" in number or subject.tag_ in {"NNS", "NNPS"}:
        return False
    return True


def _tense_suggestion(
    verb: Any,
    subject: Any,
    token_word_indexes: dict[int, int],
    words_by_index: dict[int, _TranscriptWord],
    checked_word_indexes: set[int],
) -> str | None:
    sentence_words = [
        token.text.casefold()
        for token in verb.sent
        if token.is_alpha
        and (
            (word_index := token_word_indexes.get(token.i)) is None
            or word_index in checked_word_indexes
            or words_by_index.get(word_index) is None
            or words_by_index[word_index].probability is None
            or words_by_index[word_index].probability >= LOW_CONFIDENCE
        )
    ]
    has_past_marker = any(
        word == "yesterday"
        or word == "ago"
        or (
            word == "last"
            and index + 1 < len(sentence_words)
            and sentence_words[index + 1] in _PAST_TIME_WORDS
        )
        for index, word in enumerate(sentence_words)
    )
    has_present_marker = any(
        word in _PRESENT_HABIT_WORDS
        or (
            word in {"each", "every"}
            and index + 1 < len(sentence_words)
            and sentence_words[index + 1] in _PAST_TIME_WORDS
        )
        for index, word in enumerate(sentence_words)
    )
    lemma = verb.lemma_.casefold()
    if has_past_marker and verb.tag_ in {"VBP", "VBZ"}:
        if lemma == "be":
            return "was" if _is_singular_subject(subject) else "were"
        return _past_form(lemma)
    if has_present_marker and verb.tag_ == "VBD" and lemma != "be":
        return _present_form(lemma, _is_third_person_singular(subject))
    if has_present_marker and verb.tag_ == "VBD" and lemma == "be":
        return "is" if _is_third_person_singular(subject) else "am" if subject.text.casefold() == "i" else "are"
    return None


def _past_form(lemma: str) -> str:
    if lemma in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[lemma]
    if lemma.endswith("e"):
        return f"{lemma}d"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return f"{lemma[:-1]}ied"
    return f"{lemma}ed"


def _present_form(lemma: str, third_singular: bool) -> str:
    if not third_singular:
        return lemma
    if lemma in _IRREGULAR_PRESENT_THIRD:
        return _IRREGULAR_PRESENT_THIRD[lemma]
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return f"{lemma}es"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return f"{lemma[:-1]}ies"
    return f"{lemma}s"


def _article_for(word: str) -> str:
    normalized = word.casefold()
    if normalized.startswith(("uni", "use", "euro", "one", "once")):
        return "a"
    if normalized.startswith(("hour", "honest", "honor", "heir")):
        return "an"
    return "an" if normalized[:1] in "aeiou" else "a"
