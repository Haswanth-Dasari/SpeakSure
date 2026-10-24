"""Print each word's timing and the gap to the next word for one turn of a saved session.

Usage (from the project root, venv active):
    python scripts/show_word_gaps.py data/sessions/<file>.json <turn_number>

Purpose: check whether Whisper's word timestamps contain real silent gaps between
words, or whether every word simply runs into the next one (which would hide pauses).
"""

import json
import sys
from pathlib import Path

CONTIGUOUS_SECONDS = 0.02  # a gap at or below this means the next word starts as the previous one ends
LONG_WORD_SECONDS = 0.8    # a single word lasting longer than this probably contains a pause


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    try:
        data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        turn = data["turns"][int(sys.argv[2]) - 1]
    except (OSError, ValueError, KeyError, IndexError) as error:
        print(f"Could not read that turn: {error}")
        return 1

    words = (turn.get("answer") or {}).get("words") or []
    if len(words) < 2:
        print("This turn has fewer than two timed words.")
        return 1

    print(f"{'word':<16}{'start':>7}{'end':>7}{'dur':>6}{'gap':>6}")
    contiguous = 0
    long_words = 0
    for index, word in enumerate(words):
        duration = word["end"] - word["start"]
        gap = words[index + 1]["start"] - word["end"] if index + 1 < len(words) else None
        note = ""
        if duration > LONG_WORD_SECONDS:
            note = "  <- long word"
            long_words += 1
        if gap is not None and gap <= CONTIGUOUS_SECONDS:
            contiguous += 1
        gap_text = f"{gap:.2f}" if gap is not None else "-"
        print(f"{word['text']:<16}{word['start']:>7.2f}{word['end']:>7.2f}{duration:>6.2f}{gap_text:>6}{note}")

    pairs = len(words) - 1
    print(f"\n{contiguous} of {pairs} word pairs are contiguous (gap <= {CONTIGUOUS_SECONDS:.2f} s).")
    print(f"{long_words} words last longer than {LONG_WORD_SECONDS:.1f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
