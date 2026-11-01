# Changelog

## 1.0.0 — live spoken interview coach

- Browser and terminal spoken interviews with local Whisper and Piper and a Groq
  interviewer; end-of-answer detection with noise calibration.
- Resume upload with local text extraction, contact redaction, grounded profile
  and an editable review panel; resume questions with rubric hints.
- Varied question planner (pool of 16, shuffled order, varied openers), follow-ups
  that never assume a denied experience, deeper follow-ups on projects.
- Two-pass transcription: medium live, large-v3 for the report, per-answer guard
  that keeps the live text when it holds more of the candidate's names and terms.
- Content scoring by rubric with a lenient repair step and median of three ratings.
- Report that opens with a written review (verified quotes and numbers) followed by
  the evidence; rule-based grammar section hidden by default (precision about 5%).
- Evaluation toolkit and results: E1 speech models, E2 grammar flags, E3 content
  scoring agreement, E4 review accuracy (E2 to E4 annotated by an AI assistant).
- Repository reorganised: scripts grouped into demos, eval and tools; documentation
  grouped into technical, course and results.
- Bug fixes found along the way: browser typos that blocked joining and recording,
  voice-detector sample-rate and calibration errors, thinking-time measurement,
  concurrent model loading, report-model GPU memory, quote truncation.

## 0.5.0 — Streamlit prototype

- Text-first prototype (record, transcribe, edit, submit) with OpenAI follow-ups.
  Archived in `legacy/`.
