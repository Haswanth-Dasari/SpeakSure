"""Offline tests for ASR text, manifest, and aggregate metrics."""

import csv

import pytest

from src.speaksure.evaluation.asr import (
    load_manifest,
    name_hit_rate,
    normalize_text,
    pooled_wer,
    summarize,
    word_level_errors,
)


def test_normalize_text_preserves_only_internal_apostrophes():
    assert normalize_text("  DON'T—stop, it's   fine!  ") == "don't stop it's fine"


def test_extended_normalization_handles_spelling_time_and_ordinals():
    from src.speaksure.evaluation.asr import normalize_extended
    assert normalize_extended("Center at 9 a.m., 28th") == "center at 9 am 28"
    assert normalize_extended("colour programme p.m.") == "color program pm"


def test_bootstrap_seed_is_repeatable_for_tiny_hand_computable_case():
    from src.speaksure.evaluation.asr import bootstrap_wer
    # Two one-word clips (WER 0 and 1) have bootstrap samples with WER 0, 0.5, or 1.
    pairs = [("a", "a"), ("a", "b")]
    assert bootstrap_wer(pairs, resamples=20, seed=9) == bootstrap_wer(pairs, resamples=20, seed=9)


def test_pooled_wer_weights_clips_by_reference_words():
    # Clip WERs are 1/4 and 2/1 (mean 1.125); pooled WER is 3/5 = 0.6.
    result = pooled_wer(
        [
            ("a b c d", "a x c d"),
            ("a", "x y"),
        ]
    )

    assert result == {"errors": 3, "reference_words": 5, "wer": pytest.approx(0.6)}


def test_name_hit_rate_supports_whole_tokens_phrases_case_and_absent_names():
    assert name_hit_rate(
        "MAYA PATEL joined Northbridge University yesterday",
        ["Maya", "Maya Patel", "NORTHBRIDGE UNIVERSITY"],
    ) == 1.0
    assert name_hit_rate("Mayamore visited Northbridge", ["Maya"]) == 0.0
    assert name_hit_rate("Anything", []) is None


def test_load_manifest_resolves_clips_and_parses_optional_fields(tmp_path):
    (tmp_path / "voice.wav").write_bytes(b"fake WAV")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "clip,reference,names,condition\nvoice.wav,Hello,\"Maya; Northbridge\",noisy\n",
        encoding="utf-8",
    )

    rows = load_manifest(manifest)

    assert rows == [
        {
            "clip": (tmp_path / "voice.wav").resolve(),
            "reference": "Hello",
            "names": ["Maya", "Northbridge"],
            "condition": "noisy",
            "input_device": "",
            "input_device_id": "",
            "category": "",
            "notes": "",
        }
    ]


def test_load_manifest_supports_missing_optional_columns_and_reports_missing_clip(
    tmp_path,
):
    (tmp_path / "voice.wav").touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("clip,reference\nvoice.wav,hello\n", encoding="utf-8")
    assert load_manifest(manifest)[0]["names"] == []
    assert load_manifest(manifest)[0]["condition"] == ""
    assert load_manifest(manifest)[0]["input_device"] == ""

    manifest.write_text("clip,reference\nabsent.wav,hello\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing clip"):
        load_manifest(manifest)

    manifest.write_text("clip,reference\nvoice.wav,   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty reference"):
        load_manifest(manifest)


def test_summarize_excludes_warmup_from_speed_and_device_mismatch_from_model():
    results = [
        {
            "model": "base",
            "reference": "a b",
            "hypothesis": "a b",
            "wer": 0.0,
            "warmup": True,
            "load_plus_first_run_seconds": 12.0,
            "real_time_factor": 10.0,
            "name_hits": 1,
            "name_count": 2,
            "word_probability_sum": 1.8,
            "word_count": 2,
            "actual_device": "cuda",
        },
        {
            "model": "base",
            "reference": "a b",
            "hypothesis": "a x",
            "wer": 0.5,
            "warmup": False,
            "real_time_factor": 0.4,
            "name_hits": 2,
            "name_count": 2,
            "word_probability_sum": 0.9,
            "word_count": 1,
            "actual_device": "cuda",
        },
        {
            "model": "small",
            "reference": "a",
            "hypothesis": "wrong",
            "wer": 1.0,
            "warmup": False,
            "real_time_factor": 9.0,
            "device_mismatch": True,
        },
    ]

    result = summarize(results)

    assert list(result) == ["base"]
    assert result["base"]["warm_real_time_factor_median"] == 0.4
    assert result["base"]["load_plus_first_run_seconds"] == 12.0
    assert result["base"]["name_hit_rate"] == pytest.approx(3 / 4)
    assert result["base"]["mean_word_probability"] == pytest.approx(0.9)
    assert result["base"]["actual_devices"] == ["cuda"]
    assert result["base"]["reference_words"] == 4


def test_word_level_errors_marks_substitutions_deletions_and_insertions():
    assert word_level_errors("a b c", "a x") == "[b->x] [del c]"
    assert word_level_errors("a", "a b") == "[ins b]"
    assert word_level_errors("a b", "a b") == "none"


def test_summarize_groups_by_device_and_category_and_reports_reference_words():
    results = [
        {
            "model": "base",
            "input_device": "USB mic",
            "category": "names",
            "reference": "Maya Patel",
            "hypothesis": "Maya Patel",
            "wer": 0.0,
        },
        {
            "model": "base",
            "input_device": "Laptop mic",
            "category": "names",
            "reference": "Maya",
            "hypothesis": "Maya",
            "wer": 0.0,
        },
    ]

    result = summarize(results, ("input_device", "category"))

    assert result["input_device=USB mic, category=names"]["reference_words"] == 2
    assert result["input_device=Laptop mic, category=names"]["reference_words"] == 1
