# SpeakSure architecture and data contracts

This document describes the current implementation; JSON objects may contain
additional per-version metadata where the schema permits arbitrary mappings.

## Module map

- `src/speaksure/__init__.py` — package identity; does not expose archived prototype imports.
- `src/speaksure/configuration.py` — loads root `.env` and resolves configuration precedence.
- `src/speaksure/audio_capture.py` — microphone capture, VAD, WAV serialization, and capture timings.
- `src/speaksure/speech_to_text.py` — local faster-whisper and optional OpenAI transcription providers.
- `src/speaksure/text_to_speech.py` — local Piper configuration, synthesis, and playback.
- `src/speaksure/question_bank.py` — typed interview questions and question selection.
- `src/speaksure/interviewer_prompts.py` — interviewer system and response prompt text.
- `src/speaksure/interviewer.py` — interview plan, turn selection, follow-up, and finish state.
- `src/speaksure/llm_client.py` — provider-neutral LLM protocol and Groq client.
- `src/speaksure/live_interview.py` — shared terminal/web answer processing and terminal runner.
- `src/speaksure/session_log.py` — validated session/turn schema and atomic JSON persistence.
- `src/speaksure/analysis/__init__.py` — analysis package marker.
- `src/speaksure/analysis/audio_delivery.py` — pause, level, clipping, and audio-quality heuristics.
- `src/speaksure/analysis/content.py` — optional question-rubric LLM scoring and evidence validation.
- `src/speaksure/analysis/content_prompts.py` — content-scoring prompt templates.
- `src/speaksure/analysis/fluency.py` — transcript timing, pause, filler, and repetition metrics.
- `src/speaksure/analysis/grammar.py` — offline spaCy-assisted grammar heuristics.
- `src/speaksure/analysis/lexicons.py` — filler, discourse-marker, and hedge vocabularies.
- `src/speaksure/analysis/question_labels.py` — maps asked turns to main-question groups.
- `src/speaksure/analysis/session_analysis.py` — composes per-turn and session analysis results.
- `src/speaksure/analysis/text_utils.py` — shared transcript normalization and token helpers.
- `src/speaksure/analysis/transcript_quality.py` — word-confidence and recognition-quality measures.
- `src/speaksure/evaluation/__init__.py` — evaluation package marker.
- `src/speaksure/evaluation/asr.py` — ASR manifest, WER, timing, and grouped benchmark logic.
- `src/speaksure/report/__init__.py` — report package marker and public renderer.
- `src/speaksure/report/aggregate.py` — groups delivery measurements for report charts.
- `src/speaksure/report/charts.py` — self-contained SVG chart generation.
- `src/speaksure/report/coaching.py` — selects report findings from analysis results.
- `src/speaksure/report/render.py` — escaped, standalone HTML feedback report renderer.
- `src/speaksure/web/__init__.py` — web package marker.
- `src/speaksure/web/app.py` — FastAPI routes, model warmup, turn processing, and report lifecycle.
- `src/speaksure/web/sessions.py` — thread-safe in-memory web session registry.

## One interview-turn data flow

1. The terminal recorder or browser VAD captures one mono answer; the browser
   uploads WAV to `POST /api/sessions/{id}/answer`.
2. The shared processor transcribes non-silent audio, applies the existing
   minimum-answer handling, computes live fluency/audio signals, and calls the
   interviewer for the next utterance.
3. The web path caches the Piper response before returning an audio URL; the
   terminal path synthesizes and plays the line directly. The turn is stored
   in the session with timing and interviewer metadata.
4. On finish, session analysis adds grammar, transcript-confidence, fluency,
   and delivery outputs. Optional content scoring groups turns by main
   question, calls the configured LLM, and records the result.
5. The report renderer turns the analyzed session into standalone HTML.

## Session JSON schema

`InterviewSession.SCHEMA_VERSION` is `1`. `session_log.py` validates the
following top-level object:

| Field | JSON type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Session schema version (`1`). |
| `session_id` | string | UUID hex identifier. |
| `created_at` | string | UTC ISO-8601 timestamp. |
| `mode` | string | Interview mode. |
| `difficulty` | string | Difficulty label. |
| `planned_questions` | integer | Number of main questions planned; non-negative. |
| `metadata` | object | Capture, model, role, and lifecycle metadata. |
| `turns` | array of turn objects | Ordered recorded answers and derived turn analysis. |
| `analysis` | object | Whole-session analysis results; empty before analysis. |

Each turn object has `index` (integer), `question` (string), `asked_at`
(UTC ISO-8601 string), `answer` (null or transcription object),
`audio_path` (string or null), `thinking_time_seconds` (number or null),
`answer_duration_seconds` (number or null), and `analysis` (object).
The transcription object contains `text` (string), `words` (array of
`{text: string, start: number, end: number, probability: number}`),
`duration_seconds` (number), `language` (string),
`language_probability` (number), and `device` (string). Floating-point values
are rounded to three decimal places when serialized.

## Analysis result schema

The `analysis` object on a turn is intentionally extensible. The shared
processor writes:

- `timing`: old timing keys (`capture_stop_reason`, `thinking_time_seconds`,
  `answer_duration_seconds`, `stt_seconds`, `llm_seconds`,
  `tts_synthesis_seconds`, `response_gap_estimate_seconds`,
  `end_silence_seconds`) plus `end_wait_s`, `stt_s`, `llm_s`, and `tts_s`.
  The web path also adds `upload_s`; its browser timing endpoint replaces
  `end_wait_s` and `upload_s` with measured seconds.
- `interviewer`: `kind`, `question_id`, `rationale`, `used_fallback`,
  `fallback_reason`, and `llm_latency_seconds`.
- `asked`: `main_question_id` and `is_follow_up`.
- `capture`: `seconds`, `peak`, `rms_dbfs`, `stop_reason`, and `attempts`.
- `fluency`, `audio_delivery`, `low_confidence_words`,
  `transcription_errors`, and `empty_attempts`.
- After full analysis: `grammar`, `transcript_quality`, and optionally
  `content_ref` (`main_question_id`, `group_index`). A speech-output failure
  may add `speech_output_failed`.

The session-level analysis object receives `fluency_summary` and
`delivery_summary`. With content scoring it also receives `content_summary`
and `content_by_question`. Each content group identifies its main question,
turn indexes, label source, rating state, ASR mean/reliability, and analyzer
result. Unrated groups carry a reason/error; rated groups carry rubric
components, score, evidence, and optional attempt count.

## Web API contract

All routes are served by the local FastAPI application. Error responses use
`{"error": string, "message": string}`.

| Method and path | Request | Successful response |
| --- | --- | --- |
| `GET /` | — | Static interview setup HTML. |
| `GET /static/{file_path}` | — | Static CSS, JavaScript, or HTML asset. |
| `GET /api/health` | — | `status`, per-component `components`, `stt_model`, `stt_device_setting`, `tts_voice`, and `llm_configured`. |
| `POST /api/sessions` | JSON: `client` (input device label and constraints), optional `questions`, `role`, `length`, `candidate_name`, `vocabulary_hints`. Explicit `questions` takes precedence over `length`. | `session_id`, `first_question` (`id`, `text`, cached `audio_url`), `question_index`, `total_main_questions`. |
| `GET /api/sessions/{id}/audio/{name}.wav` | — | Previously cached interviewer WAV. |
| `POST /api/sessions/{id}/answer` | WAV body with `audio/wav` content type. Browser supplies upload/end-wait measurements in `X-Client-Upload-Ms` and `X-Client-End-Wait-Ms` on the timing update request. | `turn_index`, `transcript`, `words_count`, `flags`, `interviewer` (`action`, `text`, cached `audio_url`), `finished`, `turn_recorded`. |
| `POST /api/sessions/{id}/finish` | Optional JSON `{"content": boolean}`. | Lifecycle `status` (`analyzing` or current state). |
| `GET /api/sessions/{id}/status` | — | `state`, `message`, optional `stage`, `progress`, and `report_url`. |
| `PATCH /api/sessions/{id}/turns/{turn_index}/timing` | Headers `X-Client-Upload-Ms` and `X-Client-End-Wait-Ms`; both finite, non-negative milliseconds. | `{"status": "saved"}`. |
| `GET /report/{id}` | — | Generated standalone HTML report, once ready. |
| `GET /api/sessions` | — | Recent session summaries and report links where available. |

## Timing interpretation

The terminal capture records end wait from the final voiced frame to the VAD
stop frame. Both paths record STT, LLM, and TTS synthesis seconds in the same
turn-analysis timing map. Browser upload and end-wait values are measured by
the client and persisted through the timing PATCH route. The latency report
totals end wait + upload when available + STT + LLM + TTS synthesis; it omits
total samples whose required stage measurements are missing.
