"""A second, more accurate transcription pass used only for the feedback report.

The live interview needs a fast speech model so the conversation feels natural.
The report has no such deadline, so after the interview the saved recordings are
transcribed again with a larger model. Every analyzer then reads the better text.
The live transcript is kept in each turn's ``analysis["transcript_passes"]``.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from pathlib import Path

from .session_log import InterviewSession
from .speech_to_text import SpeechToTextProvider

LOGGER = logging.getLogger(__name__)

MIN_WORDS_FOR_SANITY_CHECK = 5
MIN_KEEP_RATIO = 0.5
MAX_GROWTH_RATIO = 2.0


def choose_transcript(live_text: str, report_text: str) -> tuple[bool, str]:
    """Decide whether the report-pass text may replace the live text.

    Returns ``(use_report, reason)``. A pass that returns nothing, collapses to a
    fraction of the live text, or balloons (a typical repetition failure) is
    rejected so one bad run cannot damage the report.
    """
    live_words = len(live_text.split())
    report_words = len(report_text.split())
    if report_words == 0:
        return False, "report pass returned no text"
    if live_words >= MIN_WORDS_FOR_SANITY_CHECK:
        if report_words < live_words * MIN_KEEP_RATIO:
            return False, "report transcript was much shorter than the live one"
        if report_words > live_words * MAX_GROWTH_RATIO + 3:
            return False, "report transcript was much longer (possible repetition)"
    return True, "report transcript used"


def retranscribe_session(
    session: InterviewSession,
    provider: SpeechToTextProvider,
    *,
    report_model: str,
    live_model: str,
    hotwords: str | None = None,
    audio_root: Path = Path("."),
    progress_callback: Callable[[str, float], None] | None = None,
    skip_if_cpu: bool = True,
) -> InterviewSession:
    """Return a deep copy of ``session`` with answers re-transcribed by ``provider``.

    Turns without audio or without a live answer are left untouched. Failures on
    one turn keep that turn's live transcript and are recorded, never raised.
    """
    result = copy.deepcopy(session)
    warmup = getattr(provider, "warmup", None)
    if callable(warmup):
        warmup()
    if skip_if_cpu and getattr(provider, "actual_device", "") == "cpu":
        raise RuntimeError(
            f"The report model '{report_model}' would run on the CPU, which is too "
            "slow; keeping the live transcripts."
        )

    used_report = 0
    total = len(result.turns)
    for position, turn in enumerate(result.turns, start=1):
        if progress_callback is not None:
            progress_callback("transcribing", (position - 1) / max(1, total))
        live_text = turn.answer.text if turn.answer is not None else ""
        record: dict[str, object] = {
            "live_model": live_model,
            "report_model": report_model,
            "live_text": live_text,
            "used": "live",
        }
        turn.analysis["transcript_passes"] = record
        if turn.answer is None or not turn.audio_path:
            record["reason"] = "no answer or no saved audio"
            continue
        audio_path = Path(turn.audio_path)
        if not audio_path.is_absolute():
            audio_path = audio_root / audio_path
        if not audio_path.is_file():
            record["reason"] = "audio file not found"
            continue
        kwargs: dict[str, object] = {
            "filename": audio_path.name,
            "content_type": "audio/wav",
        }
        if hotwords:
            kwargs["hotwords"] = hotwords
        started = time.perf_counter()
        try:
            new_answer = provider.transcribe_detailed(audio_path.read_bytes(), **kwargs)
        except Exception as error:  # keep the live transcript on any failure
            LOGGER.warning("Report transcription failed for turn %s: %s", turn.index, error)
            record["reason"] = f"report pass failed: {type(error).__name__}"
            continue
        record["report_pass_seconds"] = round(time.perf_counter() - started, 3)
        accept, reason = choose_transcript(live_text, new_answer.text)
        record["reason"] = reason
        if accept:
            turn.answer = new_answer
            record["used"] = "report"
            used_report += 1
    if progress_callback is not None:
        progress_callback("transcribing", 1.0)
    if used_report:
        result.metadata["report_stt_model"] = report_model
    return result
