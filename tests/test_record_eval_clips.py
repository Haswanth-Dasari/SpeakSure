"""Offline tests for evaluation clip persistence."""

import csv
from pathlib import Path

import numpy as np

from scripts.eval.record_eval_clips import (
    _review_reference,
    append_evaluation_clip,
    parse_prompt_line,
)


def test_recorder_appends_manifest_and_never_overwrites_existing_clip(tmp_path):
    existing_clip = tmp_path / "clip_01.wav"
    existing_clip.write_bytes(b"keep this existing clip")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file, fieldnames=("clip", "reference", "names", "condition")
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip": "clip_01.wav",
                "reference": "Existing reference",
                "names": "",
                "condition": "",
            }
        )

    added_clip = append_evaluation_clip(
        tmp_path,
        np.zeros(1600, dtype=np.float32),
        16000,
        "New independent reference",
        names="Maya Patel",
        condition="hesitant",
        input_device="USB microphone",
        input_device_id=2,
        category="names",
        notes="quiet room",
    )

    assert added_clip.name == "clip_02.wav"
    assert existing_clip.read_bytes() == b"keep this existing clip"
    with manifest.open("r", encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    assert len(rows) == 2
    assert rows[0]["clip"] == "clip_01.wav"
    assert rows[0]["input_device"] == ""
    assert rows[1] == {
        "clip": "clip_02.wav",
        "reference": "New independent reference",
        "names": "Maya Patel",
        "condition": "hesitant",
        "input_device": "USB microphone",
        "input_device_id": "2",
        "category": "names",
        "notes": "quiet room",
    }


def test_prompt_line_parses_category_and_preentered_names():
    assert parse_prompt_line(
        "Maya Patel joined Northbridge University. | category=names | "
        "names=Maya Patel;Northbridge University"
    ) == (
        "Maya Patel joined Northbridge University.",
        "names",
        "Maya Patel;Northbridge University",
    )


def test_reference_review_can_correct_what_was_actually_said():
    answers = iter(["e", "I started early.", ""])
    messages = []

    reference = _review_reference(
        "I start early.",
        input_fn=lambda _prompt: next(answers),
        output_fn=messages.append,
    )

    assert reference == "I started early."
    assert messages == [
        "Reference: I start early.",
        "Reference: I started early.",
    ]


def test_example_prompts_have_requested_categories_and_word_counts():
    prompt_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "prompts.example.txt"
    prompts = [
        parse_prompt_line(line)
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = {
        category: [len(reference.split()) for reference, item_category, _ in prompts
                   if item_category == category]
        for category in ("plain", "names", "numbers", "natural")
    }

    assert len(prompts) == 14
    assert len(counts["plain"]) == 6
    assert all(12 <= count <= 20 for count in counts["plain"])
    assert len(counts["names"]) == 4
    assert len(counts["numbers"]) == 2
    assert len(counts["natural"]) == 2
    assert counts["natural"] == [30, 30]
    assert any("Maya Patel" in names for _, category, names in prompts if category == "names")
    assert any(
        "Northbridge University" in names
        for _, category, names in prompts
        if category == "names"
    )
