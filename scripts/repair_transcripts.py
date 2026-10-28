"""Repair merged transcript spacing in copied session JSON files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS = PROJECT_ROOT / "data" / "sessions"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sessions_repaired"


def _word_tokens(text: str) -> tuple[str, ...]:
    """Extract lexical tokens while ignoring punctuation-only spacing."""
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold()))


def repair_session(data: object) -> tuple[dict[str, object], int]:
    """Rebuild answer text from timed words when only whitespace differs."""
    if not isinstance(data, dict):
        raise ValueError("Session JSON root must be an object.")
    turns = data.get("turns")
    if not isinstance(turns, list):
        raise ValueError("Session JSON turns must be a list.")
    changed_turns = 0
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"Session turn {turn_index} must be an object.")
        answer = turn.get("answer")
        if answer is None:
            continue
        if not isinstance(answer, dict):
            raise ValueError(f"Session turn {turn_index} answer must be an object or null.")
        stored_text = answer.get("text")
        words = answer.get("words")
        if not isinstance(stored_text, str) or not isinstance(words, list):
            continue
        word_texts: list[str] = []
        for word_index, word in enumerate(words):
            if not isinstance(word, dict) or not isinstance(word.get("text"), str):
                raise ValueError(
                    f"Session turn {turn_index} word {word_index} must contain text."
                )
            word_texts.append(word["text"].strip())
        rebuilt_text = " ".join(text for text in word_texts if text)
        if (
            stored_text != rebuilt_text
            and "".join(stored_text.split()) == "".join(rebuilt_text.split())
            and _word_tokens(stored_text) != _word_tokens(rebuilt_text)
        ):
            answer["text"] = rebuilt_text
            changed_turns += 1
    return data, changed_turns


def _is_same_or_nested(first: Path, second: Path) -> bool:
    """Return whether either resolved path is equal to or nested under the other."""
    try:
        first.relative_to(second)
        return True
    except ValueError:
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False


def repair_directory(
    sessions_dir: Path,
    output_dir: Path,
    *,
    write: bool = False,
) -> tuple[int, int]:
    """Inspect sessions and optionally write repaired copies to a separate tree."""
    source = sessions_dir.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Sessions directory does not exist: {source}")
    if _is_same_or_nested(source, destination):
        raise ValueError("Output directory must be separate from the sessions directory.")

    files = sorted(source.rglob("*.json"))
    changed_turns = 0
    changed_sessions = 0
    for session_path in files:
        data = json.loads(session_path.read_text(encoding="utf-8"))
        repaired, changed = repair_session(data)
        changed_turns += changed
        if changed:
            changed_sessions += 1
            print(f"{session_path.relative_to(source)}: {changed} changed turn(s)")
        if write:
            output_path = destination / session_path.relative_to(source)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8", newline="\n") as output_file:
                json.dump(repaired, output_file, ensure_ascii=False, indent=2)
                output_file.write("\n")
    return changed_turns, changed_sessions


def main(argv: Sequence[str] | None = None) -> int:
    """Run a dry-run scan or write repaired session copies when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        changed_turns, changed_sessions = repair_directory(
            args.sessions_dir, args.output_dir, write=args.write
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not repair transcripts: {error}", file=sys.stderr)
        return 1
    action = "Wrote copies to" if args.write else "Dry run; no files written"
    print(
        f"{action}. Changed turns: {changed_turns} across "
        f"{changed_sessions} session file(s)."
    )
    if args.write:
        print(f"Output directory: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
