# SpeakSure

SpeakSure is a local-first, voice-first interview coach for practising spoken
job interviews. It asks questions aloud, listens, decides when an answer has
ended, transcribes it with local speech recognition, replies with a follow-up
or the next question, and finishes with a feedback report that opens with a
plain-language review and then shows the evidence. It can tailor questions to
the candidate's own resume.

## Table of contents

- [What it does](#what-it-does)
- [Demo flow](#demo-flow)
- [Architecture overview](#architecture-overview)
- [Windows PowerShell setup](#windows-powershell-setup)
- [Configuration](#configuration)
- [Run the web interview](#run-the-web-interview)
- [Resume-based interviews](#resume-based-interviews)
- [Run the terminal interview](#run-the-terminal-interview)
- [Analyze a session](#analyze-a-session)
- [Generate a report](#generate-a-report)
- [Evaluation toolkit](#evaluation-toolkit)
- [Privacy](#privacy)
- [Limitations](#limitations)
- [Project structure](#project-structure)
- [Licences and models](#licences-and-models)
- [Documentation](#documentation)
- [Use of AI assistance](#use-of-ai-assistance)

## What it does

- **Spoken interview in the browser or terminal.** Local Whisper transcription,
  local Piper voice, and an interviewer language model (Groq free tier).
- **Resume-aware.** Upload a PDF, DOCX or TXT resume; contact details are removed
  before anything is sent to a model; you review the extracted profile; resume
  questions are mixed with a varied pool of general ones.
- **Natural conversation.** Follow-ups build on what you said, never assume
  something you denied, and the question order differs every time.
- **Accurate transcripts.** A fast model runs live; a larger model re-transcribes
  the saved audio for the report, with your names and resume terms as hints.
- **A report that reads like a friend.** A short review comes first (what went
  well, what to work on, language notes with rewrites, a practice plan), followed
  by the evidence: scores, delivery measurements and transcripts.
- **Honest evaluation.** Measured speech-model choice (E1), content-scoring
  agreement (E3), grammar-flag precision (E2) and review accuracy (E4); see
  [docs/results](docs/results).

## Demo flow

In the **terminal flow**, the candidate chooses a role and question count, then
answers through a microphone. The interviewer waits for the VAD end-of-speech
window, transcribes locally, asks a follow-up or the next question, and speaks
the reply. In the **web flow**, the candidate selects a microphone and options
in the browser, joins the interview room, and speaks without a review-before-
send step. Both flows use the shared interviewer-turn processor and save
session JSON and audio locally. At the end, analysis can produce an HTML report.

## Architecture overview

```text
Resume (optional) ─> local text extraction ─> contact details removed ─> verified profile
                                                                              │ (you review it)
                                                                              ▼
                                         varied question plan  ◄── question pool (16) + resume questions
                                                  │
Browser capture ─┐                                ▼
                 ├─> WAV ─> live STT (medium) ─> interviewer LLM ─> Piper voice ─> next question
Terminal capture ┘                │
                                  └─> session JSON + local audio
                                              │
                    large-v3 re-transcription (kept per answer only if better)
                                              │
              fluency / audio-delivery / transcript-quality / content scoring (median of 3)
                                              │
                          written review (verified quotes) + evidence ─> standalone HTML report
```

The browser and terminal use the same `process_interview_turn` path. Offline
analyzers produce delivery, fluency and transcript-confidence measures. When
enabled, content scoring uses a language model and per-question rubrics, and a
written review is generated from the computed facts with every quote and number
validated. The rule-based grammar check is still computed and saved, but its
section is hidden from the report by default because it was mostly false alarms
on spoken text (see [docs/results/e2_grammar.md](docs/results/e2_grammar.md)).
The HTML report needs no hosted UI.

## Windows PowerShell setup

Run these commands from the SpeakSure project directory. Python 3.12 is the
tested interpreter:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
New-Item -ItemType Directory -Force models\piper | Out-Null
```

`requirements-dev.txt` includes the runtime requirements and adds pytest and
httpx. The spaCy English model is pinned in the runtime requirements. The first
faster-whisper inference downloads the selected STT model to the local model
cache. The Piper ONNX voice files are stored under the ignored `models/`
directory and are not committed. Obtain and place a voice model and its JSON
configuration there only after confirming that the voice's upstream license
permits your use; the currently configured Lessac voice points to a research
license requiring approval.

The app loads `.env` from the project root without overriding values already
set in the process environment. `.env.example` documents the recommended
`SPEECH_TO_TEXT_MODEL_SIZE=medium` setting for a 6 GB GPU; the code default
remains `base`. Use `medium` only if GPU memory permits. For a CPU-only system,
set `SPEECH_TO_TEXT_DEVICE=cpu`.

## Configuration

For each setting, precedence is **process environment > project-root `.env` >
built-in default**. Blank values generally select the default. The `OPENAI_*`
settings are for the optional OpenAI speech transcription backend or the
archived prototype; the current live interviewer and content analysis use the
Groq settings.

| Environment variable                      | Default                          | Meaning                                                                                                                                                                                  |
| ----------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`                          | empty                            | Credential for the optional OpenAI STT provider and archived prototype.                                                                                                                  |
| `OPENAI_MODEL`                            | `gpt-4o-mini`                    | Chat model setting retained for the archived prototype.                                                                                                                                  |
| `OPENAI_TRANSCRIPTION_MODEL`              | `whisper-1`                      | Model used when `SPEECH_TO_TEXT_PROVIDER=openai`.                                                                                                                                        |
| `GROQ_API_KEY`                            | empty                            | Enables the current Groq interviewer and optional content scoring.                                                                                                                       |
| `LLM_MODEL`                               | `openai/gpt-oss-20b`             | Groq model for live interviewer responses.                                                                                                                                               |
| `LLM_REASONING_EFFORT`                    | `low`                            | Groq interviewer reasoning effort: empty, low, medium, or high.                                                                                                                          |
| `LLM_TIMEOUT_SECONDS`                     | `20`                             | Groq request timeout, from 1 to 120 seconds.                                                                                                                                             |
| `LLM_ANALYSIS_MODEL`                      | `openai/gpt-oss-120b`            | Groq model used for content analysis.                                                                                                                                                    |
| `LLM_ANALYSIS_REASONING_EFFORT`           | `medium`                         | Groq content-analysis reasoning effort: empty, low, medium, or high.                                                                                                                     |
| `SPEECH_TO_TEXT_PROVIDER`                 | `local`                          | `local` selects faster-whisper; `openai` sends audio to OpenAI.                                                                                                                          |
| `SPEECH_TO_TEXT_MODEL_SIZE`               | `base`                           | Local faster-whisper model size; `medium` is recommended for a 6 GB GPU if it fits.                                                                                                      |
| `SPEECH_TO_TEXT_DEVICE`                   | `auto`                           | Local inference device: `auto`, `cuda`, or `cpu`.                                                                                                                                        |
| `SPEECH_TO_TEXT_LANGUAGE`                 | `en`                             | Language code for local transcription.                                                                                                                                                   |
| `SPEECH_TO_TEXT_VAD_FILTER`               | `true`                           | Enable faster-whisper VAD filtering for local transcription.                                                                                                                             |
| `SPEECH_TO_TEXT_HOTWORDS`                 | empty                            | Optional domain terms passed to local transcription.                                                                                                                                     |
| `SPEECH_TO_TEXT_INITIAL_PROMPT`           | empty                            | Optional local transcription prompt, up to 300 characters.                                                                                                                               |
| `REPORT_SPEECH_TO_TEXT_MODEL`             | `large-v3`                       | Larger model that re-transcribes saved audio for the report (`off` disables; must differ from the live model). Loaded as `int8_float16` so it fits next to the live model on a 6 GB GPU. |
| `CONTENT_RATING_SAMPLES`                  | `1` in code, `3` in `run_web.py` | How many times each question is rated; the median result is reported (1 to 5).                                                                                                           |
| `SHOW_RULE_GRAMMAR`                       | `1` in code, `0` in `run_web.py` | Set to `1` to show the rule-based grammar section in the report.                                                                                                                         |
| `TEXT_TO_SPEECH_VOICE`                    | `en_US-lessac-medium`            | Piper voice name.                                                                                                                                                                        |
| `TEXT_TO_SPEECH_MODELS_DIR`               | `models/piper`                   | Directory containing Piper ONNX voice files.                                                                                                                                             |
| `TEXT_TO_SPEECH_LENGTH_SCALE`             | `1.0`                            | Piper speaking-rate scale, from 0.5 to 2.0.                                                                                                                                              |
| `TEXT_TO_SPEECH_LEADING_SILENCE_SECONDS`  | `0.15`                           | Silence added before synthesized audio, from 0 to 1 second.                                                                                                                              |
| `TEXT_TO_SPEECH_TRAILING_SILENCE_SECONDS` | `0.5`                            | Silence added after synthesized audio, from 0 to 2 seconds.                                                                                                                              |
| `PATH`                                    | Windows process/system path      | Used to locate CUDA DLLs; SpeakSure may prepend discovered package DLL folders to the Python process PATH, not the machine-wide PATH.                                                    |

## Run the web interview

Start the local server and open the printed address in Chrome or Edge:

```powershell
.\.venv\Scripts\python.exe scripts\run_web.py --host 127.0.0.1 --port 8000
```

The setup screen reports model readiness. Grant microphone access, test the
selected input, and join the room. Camera preview is optional and is never
uploaded or stored. Captions are off by default. When the interview ends, the
room prepares the feedback and displays the report in an embedded frame.

## Resume-based interviews

On the setup screen, choose a PDF, DOCX or TXT resume and click **Analyse**
(PDF support needs `pypdf`, listed in `requirements.txt`). The resume is read on
your computer; emails, phone numbers and links are removed before any text goes
to the language model, and only the short profile shown under "Check what the
interviewer will use" is kept with the session. Edit or delete anything, then
join. Project names, tools and institutions from the profile are also added to
the speech recogniser's vocabulary.

## Run the terminal interview

The terminal loop requires a microphone and output device, a working Piper
voice, and `GROQ_API_KEY` for live model-generated interviewer turns. Without a
Groq key, startup reports a configuration error rather than silently switching
to another provider.

```powershell
.\.venv\Scripts\python.exe scripts\live_interview.py --questions 3 --role 'software engineering fresher' --name 'Candidate'
```

Use `--help` to see question, follow-up, device, end-silence, and save options.
Completed and interrupted interviews are saved beneath `data/sessions`.

## Analyze a session

Analysis adds local fluency, grammar, transcript-confidence, and WAV delivery
signals. Add `--content` to request per-question content scoring; this requires
`GROQ_API_KEY` and sends transcript text to the configured model.

```powershell
$session = Get-ChildItem data\sessions -Filter 'session_*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $session) { throw 'Run an interview first; no session JSON was found.' }
.\.venv\Scripts\python.exe scripts\analyze_session.py $session.FullName --content --role 'software engineering fresher' --force
```

Analyzed JSON is written under `data/analysis` unless `--out` is specified.
For timing summaries, pass a session file or the sessions directory to:

```powershell
.\.venv\Scripts\python.exe scripts\eval\latency_report.py data\sessions
```

This prints a Markdown table of stage median, nearest-rank 90th percentile, and
maximum. It writes `latency_report.md` beside the input. Total response latency
adds end-of-speech wait, browser upload when measured, STT, LLM, and TTS
synthesis time.

## Generate a report

Render a standalone HTML report from the newest analyzed session:

```powershell
$analyzed = Get-ChildItem data\analysis -Filter '*.analyzed.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $analyzed) { throw 'Analyze a session first; no analyzed JSON was found.' }
.\.venv\Scripts\python.exe scripts\make_report.py $analyzed.FullName
```

The HTML report is saved under `data/reports` by default and can be opened in a
browser or printed.

## Evaluation toolkit

A step-by-step guide to the four experiments (E1 to E4), with exact commands and what to record for the course report, is in [`docs/technical/EVALUATION.md`](docs/technical/EVALUATION.md). A draft report is in [`docs/course/COURSE_REPORT.md`](docs/course/COURSE_REPORT.md) and viva preparation is in [`docs/course/VIVA_QA.md`](docs/course/VIVA_QA.md).

Evaluation prompts, manifests, grammar cases, and collected clips are
course-specific local files under the ignored `data/eval` directory and are
not distributed in the repository. After preparing `prompts.txt`, record a
manifest and benchmark the selected models:

```powershell
if (-not (Test-Path data\eval\prompts.txt)) { throw 'Create the course prompt list at data\eval\prompts.txt first.' }
.\.venv\Scripts\python.exe scripts\eval\record_eval_clips.py --prompts data\eval\prompts.txt
if (-not (Test-Path data\eval\manifest.csv)) { throw 'Recording must produce data\eval\manifest.csv before benchmarking.' }
.\.venv\Scripts\python.exe scripts\eval\asr_benchmark.py --manifest data\eval\manifest.csv --models base,small --group-by model,input_device,category
```

After an interview and analysis, use the most recent session paths for grammar
review and content evaluation:

```powershell
$session = Get-ChildItem data\sessions -Filter 'session_*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $session) { throw 'Run an interview first; no session JSON was found.' }
$analyzedPath = Join-Path data\analysis ($session.BaseName + '.analyzed.json')
if (-not (Test-Path $analyzedPath)) { throw 'Analyze the session before continuing.' }
if (-not (Test-Path data\eval\grammar_cases.csv)) { throw 'Prepare labelled cases at data\eval\grammar_cases.csv first.' }
.\.venv\Scripts\python.exe scripts\eval\eval_grammar.py
.\.venv\Scripts\python.exe scripts\eval\review_grammar_flags.py $analyzedPath
.\.venv\Scripts\python.exe scripts\eval\make_rating_sheet.py $analyzedPath
```

Complete the generated blind ratings CSV, then calculate content-score agreement
or repeat scoring for stability:

```powershell
$ratings = Get-ChildItem data\eval\ratings -Filter '*.ratings.csv' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $ratings) { throw 'Complete a generated rating sheet first.' }
.\.venv\Scripts\python.exe scripts\eval\eval_content.py $ratings.FullName $analyzedPath
.\.venv\Scripts\python.exe scripts\analyze_session.py $session.FullName --content --repeat 5 --out data\analysis\session_runs.json --force
.\.venv\Scripts\python.exe scripts\eval\content_stability.py data\analysis\session_runs.json
```

Grammar case labels are by construction, not independent human gold labels.
Human ratings must be completed blind before agreement statistics are
calculated.

## Privacy

The default STT path is local faster-whisper. Under that setting, recorded audio
and transcripts are kept locally under `data/sessions`; no audio is sent to a
speech-recognition service. The explicitly selected `openai` STT provider does
send audio to OpenAI. When `GROQ_API_KEY` is configured, transcript text is
sent to Groq for live interviewer responses; enabling content scoring also
sends transcripts for rubric scoring. Piper TTS runs locally. Session JSON,
audio, model caches, and reports are ignored by Git.

## Limitations

- Recognition can miss names, technical terms, accents, and speech in noise.
- The capture and analysis paths assume a single speaker; overlapping speech
  and multiple speakers are not diarized.
- Grammar feedback is a small rule-based heuristic, not a complete grammar
  judgment; ASR errors can cause misleading flags.
- Content scores are approximate and can vary between LLM runs.
- The app measures delivery signals such as pauses and level variation; it does
  not infer emotion or personality.

## Project structure

```text
src/speaksure/       Live interview, speech, analysis, web, and report modules
scripts/
  demos/             Interactive and component demos
  eval/              Evaluation and reporting tools
  tools/             Audio, hardware, and transcript utilities
  README.md          Purpose and example command for every script
  analyze_session.py Analyze a saved interview session
  live_interview.py  Run the terminal interview
  make_report.py     Render an HTML report
  run_web.py         Start the web application
tests/               Automated tests; tests/fixtures/ holds the WAV test sample
legacy/              Archived Streamlit prototype and its excluded tests
docs/
  course/            Course report and viva preparation
  technical/         Architecture, evaluation guide, and UI reference
  results/           Evaluation results and supporting data
data/                Ignored local session, evaluation, analysis, and report files
models/              Ignored local voice/model weights
requirements.txt     Pinned runtime dependencies
requirements-dev.txt Runtime dependencies plus pytest and httpx
```

See [scripts/README.md](scripts/README.md) for a purpose and example command
for every script, and [docs/technical/ARCHITECTURE.md](docs/technical/ARCHITECTURE.md)
for the module map, schemas, data flow, and web API.

## Documentation

Start at [docs/README.md](docs/README.md). See [scripts/README.md](scripts/README.md)
for a purpose and example command for every script, and
[docs/technical/ARCHITECTURE.md](docs/technical/ARCHITECTURE.md) for the module map,
schemas, data flow, and web API. Project history is in [CHANGELOG.md](CHANGELOG.md).

## Licences and models

The Piper engine and the Piper voice collection repository declare MIT
licences, but voice data can have additional source-dataset terms. The model
card for `en_US-lessac-medium` names the Lessac Blizzard 2013 dataset and links
to a research licence that requires approval; the collection's repository
licence does not replace that dataset licence. Check the exact voice and
dataset terms before using or redistributing its weights. The installed
`en_core_web_sm` spaCy model is MIT licensed. faster-whisper and the OpenAI
Whisper project are MIT licensed; downloaded weights and all other
third-party assets remain subject to their own upstream terms.

- Piper engine licence: <https://github.com/rhasspy/piper/blob/master/LICENSE.md>
- Piper voice collection: <https://huggingface.co/rhasspy/piper-voices>
- Lessac medium voice model card: <https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD>
- Lessac dataset research licence: <https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html>
- spaCy model package: <https://spacy.io/models/en>
- faster-whisper licence: <https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE>
- Whisper licence: <https://github.com/openai/whisper/blob/main/LICENSE>
