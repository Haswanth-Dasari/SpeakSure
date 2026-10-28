"""Compare audio levels and recognition confidence across analyzed sessions.

Usage (from the project root, venv active):
    python scripts/show_audio_levels.py data/analysis/<a>.analyzed.json [<b>.analyzed.json ...]

Purpose: find out whether poor transcripts go together with quiet or distant audio.
Each row is one answer. Columns: peak (loudest sample, 0-1), mean_dB (average level of the
voiced frames, dB relative to full scale; closer to 0 is louder), std_dB (level variation),
mean_prob (average word recognition probability), and any audio quality flags.
"""

import json
import sys
from pathlib import Path


def fmt(value, digits: int = 2) -> str:
    """Format a number, or '-' when the value is missing."""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "-"


def mean(values: list) -> float | None:
    """Mean of the numeric values, or None when there are none."""
    numbers = [v for v in values if isinstance(v, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1

    print(f"{'session':<10}{'turn':>5}{'words':>7}{'peak':>8}{'mean_dB':>9}{'std_dB':>8}{'mean_prob':>11}  flags")
    for raw_path in paths:
        try:
            data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"Could not read {raw_path}: {error}")
            return 1

        session = str(data.get("session_id", ""))[:8] or "?"
        metadata = data.get("metadata") or {}
        peaks, levels, probs = [], [], []
        for turn in data.get("turns", []):
            analysis = turn.get("analysis") or {}
            audio = analysis.get("audio_delivery") or {}
            quality = analysis.get("transcript_quality") or {}
            fluency = analysis.get("fluency") or {}
            flags = ", ".join(audio.get("quality_flags") or []) or "-"
            print(
                f"{session:<10}{turn.get('index', '?'):>5}{fluency.get('word_count', '-'):>7}"
                f"{fmt(audio.get('peak')):>8}{fmt(audio.get('mean_level_db'), 1):>9}"
                f"{fmt(audio.get('level_std_db'), 1):>8}{fmt(quality.get('mean_word_probability')):>11}  {flags}"
            )
            peaks.append(audio.get("peak"))
            levels.append(audio.get("mean_level_db"))
            probs.append(quality.get("mean_word_probability"))
        print(
            f"{session:<10}{'avg':>5}{'':>7}{fmt(mean(peaks)):>8}{fmt(mean(levels), 1):>9}{'':>8}"
            f"{fmt(mean(probs)):>11}  stt_model={metadata.get('stt_model', '?')}"
            f" input_device={metadata.get('input_device', 'not recorded')}"
        )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
