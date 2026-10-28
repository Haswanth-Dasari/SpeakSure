"""Pure helpers for preparing and summarizing ASR benchmark results."""

from __future__ import annotations

import csv
import unicodedata
import random
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

import jiwer


def normalize_text(text: str) -> str:
    """Lowercase text, preserve word-internal apostrophes, and normalize spacing."""
    lowered = text.lower()
    characters: list[str] = []
    for index, character in enumerate(lowered):
        if character == "'":
            previous_is_word = index > 0 and lowered[index - 1].isalnum()
            next_is_word = index + 1 < len(lowered) and lowered[index + 1].isalnum()
            characters.append(character if previous_is_word and next_is_word else " ")
        elif character.isspace() or character.isalnum():
            characters.append(character)
        elif unicodedata.category(character).startswith("M"):
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


# Extended-only equivalence table: common US/UK spellings and time notation.
SPELLING_PAIRS = (
    ("center", "centre"), ("color", "colour"), ("organize", "organise"),
    ("organization", "organisation"), ("recognize", "recognise"),
    ("analyze", "analyse"), ("behavior", "behaviour"),
    ("favor", "favour"), ("favorite", "favourite"), ("neighbor", "neighbour"),
    ("theater", "theatre"), ("meter", "metre"), ("gray", "grey"),
    ("license", "licence"), ("defense", "defence"), ("traveling", "travelling"),
    ("program", "programme"),
)
_SPELLING_CANON = {variant: pair[0] for pair in SPELLING_PAIRS for variant in pair}


def normalize_extended(text: str) -> str:
    """Apply basic cleanup plus documented spelling, clock, and ordinal equivalences."""
    import re
    tokens = normalize_text(text).split()
    normalized = []
    for token in tokens:
        if token in {"a.m", "am", "a", "m"}:
            # Handle punctuation-spaced forms after basic cleanup using source-level regex below.
            normalized.append(token)
        else:
            normalized.append(_SPELLING_CANON.get(token, re.sub(r"(?<=\d)(?:st|nd|rd|th)$", "", token)))
    cleaned = " ".join(normalized)
    cleaned = re.sub(r"\b(?:a\s*\.\s*m\.?|a\s+m)\b", "am", cleaned)
    cleaned = re.sub(r"\b(?:p\s*\.\s*m\.?|p\s+m)\b", "pm", cleaned)
    return cleaned


def _accepted_reference(reference: str, accept: str, extended: bool) -> str:
    """Normalize each reference token using optional pipe-delimited alternatives."""
    words = normalize_extended(reference) .split() if extended else normalize_text(reference).split()
    if accept:
        alternatives = [normalize_extended(item) if extended else normalize_text(item) for item in accept.split("|")]
        if len(words) == 1 and alternatives:
            # Single-token aliases are the common manifest use; multiword aliases
            # are handled by canonicalizing the whole reference/hypothesis phrase.
            return alternatives[0]
    return " ".join(words)


def bootstrap_wer(pairs: Sequence[tuple[str, str]], *, resamples: int = 2000, seed: int = 1234,
                  extended: bool = False) -> tuple[float, float]:
    """Clip bootstrap CI. Tiny check: one perfect and one wholly-wrong 1-word clip yields only 0/0.5/1 samples."""
    if not pairs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    cached = [_word_error_counts(ref, hyp, extended=extended) for ref, hyp in pairs]
    values = []
    for _ in range(resamples):
        sample = [cached[rng.randrange(len(cached))] for _ in cached]
        errors = sum(int(item["errors"]) for item in sample)
        words = sum(int(item["reference_words"]) for item in sample)
        values.append(errors / words if words else 0.0)
    values.sort()
    return values[int(0.025 * (resamples - 1))], values[int(0.975 * (resamples - 1))]


def _word_error_counts(reference: str, hypothesis: str, *, extended: bool = False, accept: str = "") -> dict[str, int | float]:
    """Return JiWER's word-level edit counts for one normalized clip pair."""
    norm = normalize_extended if extended else normalize_text
    result = jiwer.process_words(_accepted_reference(reference, accept, extended), norm(hypothesis))
    return {
        "substitutions": result.substitutions,
        "deletions": result.deletions,
        "insertions": result.insertions,
        "errors": result.substitutions + result.deletions + result.insertions,
        "reference_words": result.hits + result.substitutions + result.deletions,
        "wer": result.wer,
    }


def word_level_errors(reference: str, hypothesis: str) -> str:
    """Format non-matching aligned words as substitutions, deletions, and insertions."""
    normalized_reference = normalize_text(reference).split()
    normalized_hypothesis = normalize_text(hypothesis).split()
    result = jiwer.process_words(
        " ".join(normalized_reference), " ".join(normalized_hypothesis)
    )
    errors: list[str] = []
    for chunk in result.alignments[0]:
        reference_words = normalized_reference[
            chunk.ref_start_idx : chunk.ref_end_idx
        ]
        hypothesis_words = normalized_hypothesis[
            chunk.hyp_start_idx : chunk.hyp_end_idx
        ]
        if chunk.type == "substitute":
            errors.extend(
                f"[{reference_word}->{hypothesis_word}]"
                for reference_word, hypothesis_word in zip(
                    reference_words, hypothesis_words
                )
            )
        elif chunk.type == "delete":
            errors.extend(f"[del {word}]" for word in reference_words)
        elif chunk.type == "insert":
            errors.extend(f"[ins {word}]" for word in hypothesis_words)
    return " ".join(errors) if errors else "none"


def pooled_wer(pairs: Iterable[tuple[str, str]], *, extended: bool = False) -> dict[str, int | float]:
    """Return pooled word errors and WER, weighted by total reference words."""
    substitutions = deletions = insertions = reference_words = 0
    for reference, hypothesis in pairs:
        counts = _word_error_counts(reference, hypothesis, extended=extended)
        substitutions += int(counts["substitutions"])
        deletions += int(counts["deletions"])
        insertions += int(counts["insertions"])
        reference_words += int(counts["reference_words"])
    errors = substitutions + deletions + insertions
    return {
        "errors": errors,
        "reference_words": reference_words,
        "wer": errors / reference_words if reference_words else 0.0,
    }


def name_hit_rate(hypothesis: str, names: Sequence[str]) -> float | None:
    """Return the fraction of case-insensitive whole-name phrases found."""
    normalized_hypothesis = f" {normalize_text(hypothesis)} "
    normalized_names = [normalize_text(name) for name in names]
    normalized_names = [name for name in normalized_names if name]
    if not normalized_names:
        return None
    hits = sum(f" {name} " in normalized_hypothesis for name in normalized_names)
    return hits / len(normalized_names)


def load_manifest(path: str | Path) -> list[dict[str, object]]:
    """Load and validate a CSV manifest, resolving clip paths beside that CSV."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"ASR manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"clip", "reference"}
        columns = set(reader.fieldnames or ())
        missing_columns = required - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"ASR manifest is missing required column(s): {missing}.")

        rows: list[dict[str, object]] = []
        for line_number, source_row in enumerate(reader, start=2):
            clip_text = (source_row.get("clip") or "").strip()
            reference = (source_row.get("reference") or "").strip()
            if not clip_text:
                raise ValueError(f"ASR manifest line {line_number} has an empty clip path.")
            if not reference:
                raise ValueError(f"ASR manifest line {line_number} has an empty reference.")
            clip_path = Path(clip_text).expanduser()
            if not clip_path.is_absolute():
                clip_path = manifest_path.parent / clip_path
            clip_path = clip_path.resolve()
            if not clip_path.is_file():
                raise FileNotFoundError(
                    f"ASR manifest line {line_number} references missing clip: {clip_path}"
                )
            rows.append(
                {
                    "clip": clip_path,
                    "reference": reference,
                    "names": [
                        name.strip()
                        for name in (source_row.get("names") or "").split(";")
                        if name.strip()
                    ],
                    "condition": (source_row.get("condition") or "").strip(),
                    "input_device": (source_row.get("input_device") or "").strip(),
                    "input_device_id": (source_row.get("input_device_id") or "").strip(),
                    "category": (source_row.get("category") or "").strip(),
                    "notes": (source_row.get("notes") or "").strip(),
                    "accept": (source_row.get("accept") or "").strip(),
                }
            )
    return rows


def summarize(
    results: Iterable[Mapping[str, object]],
    group_by: Sequence[str] = ("model",),
) -> dict[str, dict[str, object]]:
    """Summarize valid benchmark rows by selected metadata fields."""
    valid_fields = {"model", "input_device", "category"}
    group_fields = tuple(group_by)
    if not group_fields or any(field not in valid_fields for field in group_fields):
        raise ValueError("group_by must contain model, input_device, and/or category.")
    grouped_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in results:
        if bool(row.get("device_mismatch", False)):
            continue
        values = [
            str(row.get(field, "")).strip() or "(unspecified)"
            for field in group_fields
        ]
        if group_fields == ("model",):
            key = values[0]
        else:
            key = ", ".join(
                f"{field}={value}" for field, value in zip(group_fields, values)
            )
        grouped_rows.setdefault(key, []).append(row)

    summaries: dict[str, dict[str, object]] = {}
    for group, rows in grouped_rows.items():
        pairs = [
            (str(row.get("reference", "")), str(row.get("hypothesis", "")))
            for row in rows
        ]
        clip_wers = [float(row.get("wer", 0.0)) for row in rows]
        warm_rows = [row for row in rows if bool(row.get("warmup", False))]
        warm_times = [
            float(row["load_plus_first_run_seconds"])
            for row in warm_rows
            if row.get("load_plus_first_run_seconds") not in (None, "")
        ]
        warm_rtf = [
            float(row["real_time_factor"])
            for row in rows
            if not bool(row.get("warmup", False))
            and row.get("real_time_factor") not in (None, "")
        ]
        names_total = sum(int(row.get("name_count", 0) or 0) for row in rows)
        names_hit = sum(int(row.get("name_hits", 0) or 0) for row in rows)
        probabilities = [
            (float(row["word_probability_sum"]), int(row["word_count"]))
            for row in rows
            if row.get("word_probability_sum") not in (None, "")
            and int(row.get("word_count", 0) or 0) > 0
        ]
        probability_count = sum(count for _, count in probabilities)
        probability_mean = (
            sum(total for total, _ in probabilities) / probability_count
            if probability_count
            else None
        )
        devices = sorted(
            {
                str(row.get("actual_device", "")).strip()
                for row in rows
                if str(row.get("actual_device", "")).strip()
            }
        )
        pooled = pooled_wer(pairs)
        extended = pooled_wer(pairs, extended=True)
        summaries[group] = {
            "pooled_wer": pooled["wer"],
            "extended_wer": extended["wer"],
            "clips": len(rows),
            "reference_words": pooled["reference_words"],
            "mean_clip_wer": mean(clip_wers) if clip_wers else 0.0,
            "median_clip_wer": median(clip_wers) if clip_wers else 0.0,
            "name_hit_rate": names_hit / names_total if names_total else None,
            "warm_real_time_factor_median": median(warm_rtf) if warm_rtf else None,
            "load_plus_first_run_seconds": warm_times[0] if warm_times else None,
            "mean_word_probability": probability_mean,
            "actual_devices": devices,
        }
    return summaries
