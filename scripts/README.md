# Scripts index

Run commands from the repository root with the project virtual environment.
Most CLI tools accept `--help`; commands below use the runner where a safe
sample requires hardware, credentials, or a previously generated session.
Use real locally-created session and evaluation paths when passing data files.

```powershell
$session = Get-ChildItem data\sessions -Filter 'session_*.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$analyzed = Get-ChildItem data\analysis -Filter '*.analyzed.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

| Script | Purpose | Example command |
| --- | --- | --- |
| `analyze_session.py` | Add offline analysis and optionally LLM content scoring to a saved session. | `.\.venv\Scripts\python.exe scripts\analyze_session.py --help` |
| `asr_benchmark.py` | Benchmark local Whisper models against a CSV manifest. | `.\.venv\Scripts\python.exe scripts\asr_benchmark.py --help` |
| `capture_demo.py` | Capture one answer and optionally transcribe it. | `.\.venv\Scripts\python.exe scripts\capture_demo.py --help` |
| `content_stability.py` | Summarize score variation across repeated analysis runs. | `.\.venv\Scripts\python.exe scripts\content_stability.py --help` |
| `cuda_check.py` | Diagnose CTranslate2/CUDA DLL availability on Windows. | `.\.venv\Scripts\python.exe scripts\cuda_check.py` |
| `eval_content.py` | Compare blind human rubric ratings with LLM scores. | `.\.venv\Scripts\python.exe scripts\eval_content.py --help` |
| `eval_grammar.py` | Evaluate rule-based grammar results on labelled cases. | `.\.venv\Scripts\python.exe scripts\eval_grammar.py --help` |
| `evaluate_stt.py` | Measure local STT WER and warm-run latency on `tests/fixtures/harvard.wav`. | `.\.venv\Scripts\python.exe scripts\evaluate_stt.py` |
| `filler_probe.py` | Compare local transcription with and without a disfluency prompt. | `.\.venv\Scripts\python.exe scripts\filler_probe.py --help` |
| `interviewer_demo.py` | Run an interactive typed HR interview. | `.\.venv\Scripts\python.exe scripts\interviewer_demo.py --help` |
| `live_interview.py` | Run the live terminal voice interview. | `.\.venv\Scripts\python.exe scripts\live_interview.py --help` |
| `llm_demo.py` | Send a test prompt to the configured Groq LLM. | `.\.venv\Scripts\python.exe scripts\llm_demo.py --help` |
| `log_demo.py` | Transcribe a WAV and demonstrate session JSON persistence. | `.\.venv\Scripts\python.exe scripts\log_demo.py --help` |
| `make_rating_sheet.py` | Generate a blind Markdown rating sheet and empty CSV. | `.\.venv\Scripts\python.exe scripts\make_rating_sheet.py --help` |
| `make_report.py` | Render an analyzed session into standalone HTML feedback. | `.\.venv\Scripts\python.exe scripts\make_report.py --help` |
| `mic_probe.py` | Record or reuse a WAV while probing STT settings. | `.\.venv\Scripts\python.exe scripts\mic_probe.py --help` |
| `record_eval_clips.py` | Record prompted or spontaneous clips for ASR evaluation. | `.\.venv\Scripts\python.exe scripts\record_eval_clips.py --help` |
| `repair_transcripts.py` | Repair merged transcript spacing in copied session JSON. | `.\.venv\Scripts\python.exe scripts\repair_transcripts.py --help` |
| `review_grammar_flags.py` | Export flags for human review or score completed verdicts. | `.\.venv\Scripts\python.exe scripts\review_grammar_flags.py --help` |
| `run_web.py` | Start the local FastAPI interview application. | `.\.venv\Scripts\python.exe scripts\run_web.py --help` |
| `show_audio_levels.py` | Inspect audio-level and word-confidence measures per turn. | `.\.venv\Scripts\python.exe scripts\show_audio_levels.py $analyzed.FullName` |
| `show_word_gaps.py` | Print recognized word timestamps and adjacent gaps. | `.\.venv\Scripts\python.exe scripts\show_word_gaps.py $session.FullName 1` |
| `transcribe_demo.py` | Print detailed local transcription output for a WAV. | `.\.venv\Scripts\python.exe scripts\transcribe_demo.py tests\fixtures\harvard.wav` |
| `tts_demo.py` | Synthesize and play a test utterance with the configured Piper voice. | `.\.venv\Scripts\python.exe scripts\tts_demo.py --help` |
| `latency_report.py` | Summarize per-stage and end-of-answer-to-reply latency from saved sessions. | `.\.venv\Scripts\python.exe scripts\latency_report.py data\sessions` |

`capture_demo.py`, `evaluate_stt.py`, `live_interview.py`, `mic_probe.py`,
`record_eval_clips.py`, and `tts_demo.py` may require audio hardware or model
files. LLM-dependent tools need `GROQ_API_KEY`. ASR benchmark and evaluation
commands require matching files under the ignored `data/` directory.
