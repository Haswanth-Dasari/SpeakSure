# SpeakSure

SpeakSure helps people build confidence in English communication through realistic practice. Version 0.5.0 adds local speech recognition with faster-whisper, alongside the optional OpenAI transcription backend.

## Features

- Configure 10, 15, 30, or 60 minute sessions across four practice modes and three difficulty levels.
- Submit answers, retain session history across Streamlit reruns, and review a completed transcript.
- When configured, AI asks one concise follow-up based on the selected mode, difficulty, previous questions, and answers. HR interview mode uses a professional interviewer voice.
- Without an API key, the app clearly labels and cycles through curated scripted prompts for each practice mode. If an AI request fails, retry it or switch to scripted prompts; the submitted answer remains recorded once and the selected session configuration is preserved.
- Record a response in the browser, transcribe it locally by default, and review or edit the recognized text before submitting. Typed answers remain available. OpenAI transcription remains available as an explicitly selected backend.

The chosen duration is a session setting and does not enforce a timer. Audio is handled in memory for transcription and is not permanently stored by SpeakSure. Local audio inference stays on this computer; first use downloads the selected model from Hugging Face and therefore needs internet access. Speech synthesis uses Piper (GPL-licensed engine); each voice model has its own license, so see its model card on Hugging Face. Speech/emotion analysis is not implemented.

## Setup on Windows PowerShell

From the SpeakSure project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

SpeakSure's requirements install spaCy and the pinned `en_core_web_sm` model
wheel; this one-time setup needs internet access. Grammar feedback then runs
offline, uses the CPU, and does not need Java or an API key. If either dependency
is missing after setup, the report marks grammar analysis unavailable instead
of attempting a download at runtime.

Local transcription is the default and does not need `OPENAI_API_KEY` or OpenAI API credits. Configure it in the project-root `.env` or process environment:

```dotenv
SPEECH_TO_TEXT_PROVIDER=local
SPEECH_TO_TEXT_MODEL_SIZE=base
SPEECH_TO_TEXT_DEVICE=auto
```

The `base` model is a practical starting point for a 6 GB GPU and CPU fallback. `small` may be more accurate but uses more memory and runs slower. `SPEECH_TO_TEXT_DEVICE=auto` uses CUDA only when CTranslate2 reports a CUDA device; otherwise it uses CPU `int8`. If CUDA model initialization or inference fails in auto mode, local inference retries with CPU `int8`. Set the device to `cuda` to require CUDA (no CPU fallback), or `cpu` to force CPU. CUDA acceleration requires compatible NVIDIA drivers and CUDA/cuDNN libraries recognized by CTranslate2; an installed NVIDIA GPU alone does not guarantee CUDA inference. On Windows, SpeakSure discovers DLL folders in installed NVIDIA Python packages and adds them to the current process's PATH before importing CTranslate2 or Faster-Whisper. It does not modify the machine-wide PATH.

The first local use downloads model weights to the faster-whisper/Hugging Face cache and needs an internet connection. Later runs reuse that cache. Local recordings and transcripts are processed in memory; SpeakSure does not send audio to a transcription service or store audio recordings.

To use paid API transcription instead, explicitly select `SPEECH_TO_TEXT_PROVIDER=openai`, configure `OPENAI_API_KEY`, and optionally set `OPENAI_TRANSCRIPTION_MODEL` (default `whisper-1`). `OPENAI_MODEL` selects the separate AI follow-up model (default `gpt-4o-mini`). The OpenAI backend sends recordings to OpenAI and may incur charges. Backend selection does not silently fall back: a selected provider error is shown rather than invoking the other backend.

SpeakSure loads the project-root `.env` independently of the shell's current directory, without overriding values already set in the process environment. `.env.example` contains placeholders/defaults and is not loaded automatically. Restart Streamlit after changing configuration.

## Configuration precedence

For every setting, SpeakSure uses the process environment first, then the
project-root `.env`, then the built-in default. For example, this PowerShell
override selects `tiny` for one run of the transcription demo:

```powershell
$env:SPEECH_TO_TEXT_MODEL_SIZE = "tiny"
python scripts/transcribe_demo.py .\harvard.wav
Remove-Item Env:SPEECH_TO_TEXT_MODEL_SIZE
```

Run this safe diagnostic from PowerShell in the project root. It reports the selected provider/model, device available to CTranslate2, initialized device, and dependency presence. Before the first successful transcription in that Python process, `actual_device` is `not-initialized`; the app also displays the device used after local inference. The diagnostic never prints the API key:

```powershell
Set-Location '<project-root>'
& '.\.venv\Scripts\python.exe' -c "import json; from src.speaksure.speech_to_text import diagnose_transcription_configuration; print(json.dumps(diagnose_transcription_configuration(), indent=2))"
```

Microphone recording uses Streamlit's built-in `st.audio_input` and requires a supported, up-to-date browser and permission to use the microphone. The app accepts audio formats supported by the transcription provider (including the widget's WAV recording) up to 25 MB. If microphone access or transcription is unavailable, users can continue by typing.

Install dependencies in the project virtual environment:

```powershell
Set-Location '<project-root>'
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

For CPU-only operation, set `SPEECH_TO_TEXT_DEVICE=cpu`; faster-whisper uses `int8` CPU inference. If CUDA is explicitly requested and CTranslate2 cannot initialize it, configure CPU mode. Model download failures need an internet connection on the first load; model loading errors, unavailable CUDA, unsupported audio, and inference errors are reported without switching to the OpenAI backend.

CPU fallback remains available when CUDA libraries are missing or CUDA initialization fails. Review and correct the editable transcript before submitting. GPU availability and successful inference depend on the local driver and installed CUDA libraries.

## Tests

```powershell
python -m pytest -q
```

Provider unit tests mock faster-whisper and the OpenAI SDK. They do not download model weights, make network calls, require real API credentials, or execute on a GPU. Browser microphone permission, actual model download/inference, and hardware device selection still need to be checked manually on the target machine.

To try a typed-text HR mock interview with the configured Groq language model, run `python scripts/interviewer_demo.py --questions 3 --name Haswanth`. Use `--follow-ups`, `--role`, `--seed`, or `--no-save` to adjust the session; by default, completed and interrupted sessions are saved under `data/sessions`.

## Evaluation toolkit

E1 ASR: record prompted clips and benchmark them by model, device, and category.
```powershell
Copy-Item data/eval/prompts.example.txt data/eval/prompts.txt
python scripts/record_eval_clips.py --prompts data/eval/prompts.txt
python scripts/asr_benchmark.py --manifest data/eval/manifest.csv --models base,small --group-by model,input_device,category
```
E2 grammar: evaluate the labelled, by-construction cases; the labels are not an independent gold standard.
`python scripts/eval_grammar.py`
E2 real speech: export flags with `python scripts/review_grammar_flags.py analyzed.json`, fill verdicts, then calculate precision with `python scripts/review_grammar_flags.py --score`.
E3 content: create blind sheets with `python scripts/make_rating_sheet.py data/analysis/session.analyzed.json`, fill ratings, then run `python scripts/eval_content.py ratings.csv data/analysis/session.analyzed.json --merge more-ratings.csv`.
E3 stability: repeat content scoring with `python scripts/analyze_session.py session.json --content --repeat 5 --out data/analysis/session_runs.json`, then summarize with `python scripts/content_stability.py data/analysis/session_runs.json`.
The E1 recorder stores the selected microphone and category; `e` edits references to match what was actually spoken.
E2 case labels are deliberately injected for evaluation and must not be read as human-validated annotation.
E3 student ratings are never populated by the rating-sheet generator; score agreement only after blind ratings are complete.

## Web app

Start the local server with `python scripts/run_web.py` and open `http://127.0.0.1:8000` in Google Chrome or Microsoft Edge.
Microphone advice: Prefer a laptop microphone or wired input; avoid Bluetooth hands-free profiles due to muffled audio.

Manual test checklist (10 steps):
1. Setup: Check STT model status, set 3 questions, enter candidate name and optional vocabulary hints.
2. Mic grant: Click "Grant microphone" and select your laptop microphone from the enumerated list.
3. Mic test: Click "Test microphone", speak for 3 seconds, confirm "Good level", and play back test audio.
4. Begin interview: Click "Start interview", listen to the interviewer audio, and verify automatic listening turn start.
5. Normal turn: Speak an answer; verify live level meter and automatic answer submission after 2.5 s of silence.
6. Transcript & flags: Verify "What I heard:" displays your transcript with any relevant delivery flags.
7. Repeat request: In a turn, ask for clarification or test 15 s silence timeout ("I did not hear you" retry).
8. Early end: Click "End interview early", confirm the prompt, and verify transition to the results screen.
9. Report display: Observe status polling until "ready", then review the report rendered inside the sandboxed iframe.
10. Past sessions: Click "Start new interview", verify the completed session appears in the list, and open its report link.
