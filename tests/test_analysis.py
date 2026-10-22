"""Tests for the pure transcript fluency and quality analyzers."""

import json

import pytest

from src.speaksure.analysis.fluency import (
    LONG_PAUSE_SECONDS,
    PAUSE_MIN_SECONDS,
    analyze_fluency,
)
from src.speaksure.analysis.lexicons import (
    DISCOURSE_MARKERS,
    HEDGES,
    HESITATION_FILLERS,
)
from src.speaksure.analysis.session_analysis import analyze_session
from src.speaksure.analysis.text_utils import (
    count_phrases,
    normalize_token,
    split_sentences,
)
from src.speaksure.analysis.transcript_quality import (
    LOW_CONFIDENCE,
    analyze_transcript_quality,
)
from src.speaksure.session_log import InterviewSession
from src.speaksure.speech_to_text import TranscriptWord, TranscriptionResult


def make_answer(
    text: str,
    words: tuple[TranscriptWord, ...] = (),
) -> TranscriptionResult:
    """Build a synthetic transcription result."""
    return TranscriptionResult(text, words, 5.0, "en", 1.0, "test")


def timed_words(
    tokens: list[str],
    starts: list[float],
    ends: list[float],
    probabilities: list[float] | None = None,
) -> tuple[TranscriptWord, ...]:
    """Create word timings from explicit start/end times."""
    probs = probabilities or [0.9] * len(tokens)
    return tuple(
        TranscriptWord(token, start, end, probability)
        for token, start, end, probability in zip(tokens, starts, ends, probs)
    )


def test_normalize_token_and_split_sentences():
    """Token normalization preserves inner punctuation and sentence splitting omits empties."""
    assert normalize_token("“Don't-stop,”") == "don't-stop"
    assert split_sentences("One.  Two?! ... Three!") == ["One", "Two", "Three"]


def test_count_phrases_counts_phrases_without_overlapping_matches():
    """Phrase matches advance beyond consumed tokens and avoid overlap."""
    assert count_phrases(
        ["i", "think", "kind", "of", "this"],
        HEDGES,
    ) == 2
    assert count_phrases(
        ["you", "know", "you", "know"],
        DISCOURSE_MARKERS,
    ) == 2
    assert count_phrases(["i", "mean"], (("i",), ("i", "mean"))) == 1


def test_word_sentence_counts_and_hand_computed_speech_rate():
    """Ten timed words spanning four seconds produce 150 words per minute."""
    tokens = "one two three four five six seven eight nine ten".split()
    words = timed_words(
        tokens,
        [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6],
        [0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 4.0],
    )

    metrics = analyze_fluency(make_answer("One two three four five. Six seven eight nine ten!", words), 1.25)

    assert metrics["word_count"] == 10
    assert metrics["sentence_count"] == 2
    assert metrics["speech_seconds"] == pytest.approx(4.0)
    assert metrics["words_per_minute"] == pytest.approx(150.0)
    assert metrics["thinking_time_seconds"] == 1.25


@pytest.mark.parametrize(
    ("count", "span"),
    [(4, 4.0), (5, 1.9)],
)
def test_wpm_is_none_for_too_few_timed_words_or_short_span(count, span):
    """WPM requires five timed words and at least a two-second speech span."""
    words = timed_words(
        [f"w{index}" for index in range(count)],
        [index * span / count for index in range(count)],
        [(index + 1) * span / count for index in range(count)],
    )

    assert analyze_fluency(make_answer(" ".join(word.text for word in words), words), None)[
        "words_per_minute"
    ] is None


def test_pause_boundaries_and_totals():
    """Two qualifying gaps include one long pause; a 0.49-second gap is ignored."""
    tokens = ["a", "b", "c", "d"]
    words = timed_words(
        tokens,
        [0.0, 0.8, 2.6, 3.29],
        [0.2, 1.0, 2.8, 3.49],
    )

    metrics = analyze_fluency(make_answer("a b c d", words), None)

    assert PAUSE_MIN_SECONDS == 0.5
    assert LONG_PAUSE_SECONDS == 1.5
    assert metrics["pause_count"] == 2
    assert metrics["long_pause_count"] == 1
    assert metrics["total_pause_seconds"] == pytest.approx(2.2)
    assert metrics["longest_pause_seconds"] == pytest.approx(1.6)
    assert metrics["pause_ratio"] == pytest.approx(2.2 / 3.49)


def test_gap_of_049_seconds_is_not_a_pause():
    """A gap below the configured pause threshold is excluded."""
    words = timed_words(["one", "two"], [0, 0.69], [0.2, 0.89])

    metrics = analyze_fluency(make_answer("one two", words), None)

    assert metrics["pause_count"] == 0
    assert metrics["total_pause_seconds"] == 0


def test_fillers_ignore_case_and_edge_punctuation():
    """Filler words are detected after case-folding and punctuation stripping."""
    words = "UM, then uh... we continued".split()
    metrics = analyze_fluency(make_answer(" ".join(words)), None)

    assert "um" in HESITATION_FILLERS and "uh" in HESITATION_FILLERS
    assert metrics["filler_count"] == 2
    assert metrics["fillers_per_100_words"] == pytest.approx(40)


def test_hedges_and_discourse_markers_count_multiword_phrases_once():
    """Multi-token hedges and discourse markers are counted as phrases."""
    text = "I think we can, kind of, improve this, you know, basically."
    metrics = analyze_fluency(make_answer(text), None)

    assert ("i", "think") in HEDGES and ("kind", "of") in HEDGES
    assert ("you", "know") in DISCOURSE_MARKERS
    assert metrics["hedge_count"] == 2
    assert metrics["discourse_marker_count"] == 2
    assert metrics["hedges_per_100_words"] is not None


def test_immediate_repetitions_and_whitelist():
    """Repeated normalized tokens count except whitelisted had and that."""
    metrics = analyze_fluency(make_answer("Very, very good had had that that yes yes."), None)

    assert metrics["immediate_repetitions"] == 2


def test_typed_answer_without_word_timings_uses_text_counts():
    """Text-only answers retain lexical metrics but have no timing metrics."""
    metrics = analyze_fluency(make_answer("Hello there. This is typed."), 0.8)

    assert metrics["has_answer"] is True
    assert metrics["word_count"] == 5
    assert metrics["sentence_count"] == 2
    assert metrics["speech_seconds"] is None
    assert metrics["words_per_minute"] is None
    assert metrics["pause_ratio"] is None


@pytest.mark.parametrize("answer", [None, make_answer("")])
def test_none_and_empty_answers_return_safe_empty_metrics(answer):
    """Missing and empty answers do not raise and have no lexical activity."""
    metrics = analyze_fluency(answer, None)

    assert metrics["has_answer"] is False
    assert metrics["word_count"] == 0
    assert metrics["sentence_count"] == 0
    assert metrics["speech_seconds"] is None
    assert metrics["words_per_minute"] is None
    assert metrics["pause_ratio"] is None
    assert metrics["fillers_per_100_words"] is None
    assert metrics["hedges_per_100_words"] is None


def test_transcript_quality_known_probabilities():
    """Quality metrics report mean, minimum, ratio, and low words in order."""
    words = timed_words(
        ["one", "two", "three", "four"],
        [0, 0.2, 0.4, 0.6],
        [0.1, 0.3, 0.5, 0.7],
        [0.9, 0.4, 0.2, 0.8],
    )

    metrics = analyze_transcript_quality(make_answer("one two three four", words))

    assert LOW_CONFIDENCE == 0.5
    assert metrics["has_timed_words"] is True
    assert metrics["mean_word_probability"] == pytest.approx(0.575)
    assert metrics["low_confidence_ratio"] == pytest.approx(0.5)
    assert metrics["low_confidence_words"] == [
        {"word": "two", "probability": 0.4},
        {"word": "three", "probability": 0.2},
    ]
    assert metrics["min_word_probability"] == 0.2


def test_transcript_quality_without_timed_words_returns_none_probabilities():
    """Text-only results have no word-probability statistics."""
    metrics = analyze_transcript_quality(make_answer("typed words"))

    assert metrics["has_timed_words"] is False
    assert metrics["mean_word_probability"] is None
    assert metrics["low_confidence_ratio"] == 0.0
    assert metrics["low_confidence_words"] == []
    assert metrics["min_word_probability"] is None


def test_analyze_session_deep_copies_preserves_turn_data_and_summarizes():
    """Session analysis adds metrics without mutating or replacing existing data."""
    session = InterviewSession.new("hr", "standard", 3, {"source": "test"})
    timed = timed_words(
        ["um", "i", "think", "yes", "yes", "indeed"],
        [0.0, 0.6, 1.0, 1.5, 2.0, 3.8],
        [0.2, 0.8, 1.2, 1.7, 2.2, 4.0],
        [0.9, 0.8, 0.7, 0.4, 0.4, 0.9],
    )
    session.add_turn(
        "First",
        answer=make_answer("Um, I think yes yes indeed.", timed),
        thinking_time_seconds=1.0,
        analysis={"interviewer": {"kind": "next_question"}},
    )
    session.add_turn("Second", answer=None, thinking_time_seconds=None)
    session.add_turn("Third", answer=make_answer("Typed only answer."), thinking_time_seconds=3.0)
    original = json.loads(json.dumps(session.to_dict()))

    analyzed = analyze_session(session)

    assert session.to_dict() == original
    assert analyzed is not session
    assert analyzed.turns[0] is not session.turns[0]
    assert analyzed.turns[0].analysis["interviewer"] == {"kind": "next_question"}
    assert "fluency" in analyzed.turns[0].analysis
    summary = analyzed.analysis["fluency_summary"]
    assert summary["answered_turns"] == 2
    assert summary["unanswered_turns"] == 1
    assert summary["total_fillers"] == 1
    assert summary["total_hedges"] == 1
    assert summary["total_immediate_repetitions"] == 1
    assert summary["wpm_by_turn"][1] is None
    assert summary["pause_ratio_by_turn"][1] is None
    assert summary["thinking_time_by_turn"] == [1.0, None, 3.0]
    assert summary["mean_thinking_time_seconds"] == 2.0
    assert summary["mean_transcript_probability"] == pytest.approx(0.6833333333)
    assert summary["analysis_versions"] == {
        "fluency": 1,
        "transcript_quality": 1,
        "audio_delivery": 1,
    }
