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
