"""Record prompt-reading or spontaneous-answer clips for the ASR evaluation set."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.audio_capture import (  # noqa: E402
    CapturedAnswer,
    CaptureSettings,
    FloatFrame,
    describe_input_device,
    record_answer,
    wav_bytes,
)
from speaksure.question_bank import select_questions  # noqa: E402

EVAL_DIRECTORY = PROJECT_ROOT / "data" / "eval"
MANIFEST_FIELDS = (
    "clip",
    "reference",
    "names",
    "condition",
    "input_device",
    "input_device_id",
    "category",
    "notes",
)
VALID_CATEGORIES = frozenset({"plain", "names", "numbers", "natural"})


def parse_prompt_line(line: str) -> tuple[str, str, str]:
    """Parse a prompt and optional category/names metadata separated by pipes."""
    parts = [part.strip() for part in line.split("|")]
    reference = parts[0]
    if not reference:
        raise ValueError("Prompt text cannot be empty.")
    metadata: dict[str, str] = {}
    for part in parts[1:]:
        key, separator, value = part.partition("=")
        key = key.strip().lower()
        if not separator or key not in {"category", "names"}:
            raise ValueError(f"Unsupported prompt metadata: {part}")
        if key in metadata:
            raise ValueError(f"Duplicate prompt metadata: {key}")
        metadata[key] = value.strip()
    category = metadata.get("category", "")
    if category and category not in VALID_CATEGORIES:
        raise ValueError(
            "Prompt category must be plain, names, numbers, or natural."
        )
    return reference, category, metadata.get("names", "")


def _ensure_manifest_fields(manifest_path: Path) -> None:
    """Add newer optional columns to an existing manifest without dropping data."""
    if not manifest_path.exists() or manifest_path.stat().st_size == 0:
        return
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        existing_fields = reader.fieldnames
        if existing_fields is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        rows = list(reader)
    if all(field in existing_fields for field in MANIFEST_FIELDS):
        return
    fields = list(existing_fields)
    fields.extend(field for field in MANIFEST_FIELDS if field not in fields)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=manifest_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, manifest_path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def append_evaluation_clip(
    directory: Path,
    samples: FloatFrame,
    sample_rate: int,
    reference: str,
    *,
    names: str = "",
    condition: str = "",
    input_device: str = "",
    input_device_id: int | str | None = None,
    category: str = "",
    notes: str = "",
) -> Path:
    """Save a new numbered WAV and append its reference metadata to the manifest."""
    if not reference.strip():
        raise ValueError("Reference text cannot be empty.")
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.csv"
    clip_number = 1
    while True:
        clip_name = f"clip_{clip_number:02d}.wav"
        clip_path = directory / clip_name
        try:
            with clip_path.open("xb") as clip_file:
                clip_file.write(wav_bytes(samples, sample_rate))
            break
        except FileExistsError:
            clip_number += 1

    try:
        _ensure_manifest_fields(manifest_path)
        needs_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
        with manifest_path.open("a", encoding="utf-8", newline="") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "clip": clip_name,
                    "reference": reference.strip(),
                    "names": names.strip(),
                    "condition": condition.strip(),
                    "input_device": input_device.strip(),
                    "input_device_id": (
                        "" if input_device_id is None else str(input_device_id)
                    ),
                    "category": category.strip(),
                    "notes": notes.strip(),
                }
            )
    except OSError:
        clip_path.unlink(missing_ok=True)
        raise
    return clip_path


def _keep_recording(input_fn: Callable[[str], str] = input) -> bool | None:
    """Ask whether to keep, re-record, or skip the current recording."""
    while True:
        choice = input_fn("Enter = keep, r = re-record, s = skip: ").strip().lower()
        if choice == "":
            return True
        if choice == "r":
            return False
        if choice == "s":
            return None
        print("Choose Enter, r, or s.")


def _review_reference(
    reference: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> str:
    """Let the recorder keep or correct the reference before saving a clip."""
    while True:
        output_fn(f"Reference: {reference}")
        choice = input_fn("Enter = keep reference, e = edit: ").strip().lower()
        if choice == "":
            return reference
        if choice == "e":
            edited_reference = input_fn("Correct reference text: ").strip()
            if edited_reference:
                reference = edited_reference
            else:
                output_fn("Reference text cannot be empty.")
            continue
        output_fn("Choose Enter or e.")


def _record_until_choice(
    *,
    input_device: int | None,
    prompt: str,
    input_fn: Callable[[str], str],
    capture_fn: Callable[..., CapturedAnswer],
    output_fn: Callable[[str], None],
) -> CapturedAnswer | None:
    """Record one utterance and repeat or skip according to the student's choice."""
    capture_settings = CaptureSettings(end_silence_seconds=2.5)
    while True:
        output_fn(prompt)
        captured = capture_fn(settings=capture_settings, device=input_device)
        choice = _keep_recording(input_fn)
        if choice is True:
            return captured
        if choice is None:
            return None


def main(argv: Sequence[str] | None = None) -> int:
    """Capture evaluation clips from prompts or question-bank free responses."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompts", type=Path)
    mode.add_argument("--free", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-device", type=int)
    args = parser.parse_args(argv)

    input_device = describe_input_device(args.input_device)
    print(f"Input device: {input_device.get('name', 'unknown')}")
    if args.free:
        prompts = [(select_questions(2, seed=args.seed)[1].text, "", "")]
    else:
        prompt_path = args.prompts
        if prompt_path is None or not prompt_path.is_file():
            parser.error(f"Prompt file does not exist: {prompt_path}")
        prompt_lines = [
            line.strip()
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not prompt_lines:
            parser.error(f"Prompt file contains no prompts: {prompt_path}")
        try:
            prompts = [parse_prompt_line(line) for line in prompt_lines]
        except ValueError as error:
            parser.error(str(error))

    for prompt, prompt_category, prompt_names in prompts:
        captured = _record_until_choice(
            input_device=args.input_device,
            prompt=f"Read aloud: {prompt}" if not args.free else f"Answer aloud: {prompt}",
            input_fn=input,
            capture_fn=record_answer,
            output_fn=print,
        )
        if captured is None:
            print("Skipped.")
            continue
        reference = prompt
        if args.free:
            while True:
                reference = input(
                    "Type what you actually said (do not use a transcript): "
                ).strip()
                if reference:
                    break
                print("Reference text cannot be empty.")
        reference = _review_reference(reference)
        condition = input("Condition label (optional): ").strip()
        names = prompt_names
        if not names:
            names = input("Names to check, separated by ';' (optional): ").strip()
        category = prompt_category or input(
            "Category (plain, names, numbers, natural; optional): "
        ).strip()
        if category and category not in VALID_CATEGORIES:
            parser.error("Category must be plain, names, numbers, or natural.")
        notes = input("Notes (optional): ").strip()
        clip_path = append_evaluation_clip(
            EVAL_DIRECTORY,
            captured.samples,
            captured.sample_rate,
            reference,
            names=names,
            condition=condition,
            input_device=str(input_device.get("name", "unknown")),
            input_device_id=input_device.get("index"),
            category=category,
            notes=notes,
        )
        print(f"Saved {clip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
