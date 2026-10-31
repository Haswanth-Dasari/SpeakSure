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
| `eval/asr_benchmark.py` | Benchmark local Whisper models against a CSV manifest. | `.\.venv\Scripts\python.exe scripts\eval\asr_benchmark.py --help` |
| `demos/capture_demo.py` | Capture one answer and optionally transcribe it. | `.\.venv\Scripts\python.exe scripts\demos\capture_demo.py --help` |
| `eval/content_stability.py` | Summarize score variation across repeated analysis runs. | `.\.venv\Scripts\python.exe scripts\eval\content_stability.py --help` |
| `tools/cuda_check.py` | Diagnose CTranslate2/CUDA DLL availability on Windows. | `.\.venv\Scripts\python.exe scripts\tools\cuda_check.py` |
| `eval/eval_content.py` | Compare blind human rubric ratings with LLM scores. | `.\.venv\Scripts\python.exe scripts\eval\eval_content.py --help` |
| `eval/eval_grammar.py` | Evaluate rule-based grammar results on labelled cases. | `.\.venv\Scripts\python.exe scripts\eval\eval_grammar.py --help` |
| `eval/evaluate_stt.py` | Measure local STT WER and warm-run latency on `tests/fixtures/harvard.wav`. | `.\.venv\Scripts\python.exe scripts\eval\evaluate_stt.py` |
| `tools/filler_probe.py` | Compare local transcription with and without a disfluency prompt. | `.\.venv\Scripts\python.exe scripts\tools\filler_probe.py --help` |
| `demos/interviewer_demo.py` | Run an interactive typed HR interview. | `.\.venv\Scripts\python.exe scripts\demos\interviewer_demo.py --help` |
| `live_interview.py` | Run the live terminal voice interview. | `.\.venv\Scripts\python.exe scripts\live_interview.py --help` |
| `demos/llm_demo.py` | Send a test prompt to the configured Groq LLM. | `.\.venv\Scripts\python.exe scripts\demos\llm_demo.py --help` |
| `demos/log_demo.py` | Transcribe a WAV and demonstrate session JSON persistence. | `.\.venv\Scripts\python.exe scripts\demos\log_demo.py --help` |
| `eval/make_rating_sheet.py` | Generate a blind Markdown rating sheet and empty CSV. | `.\.venv\Scripts\python.exe scripts\eval\make_rating_sheet.py --help` |
| `make_report.py` | Render an analyzed session into standalone HTML feedback. | `.\.venv\Scripts\python.exe scripts\make_report.py --help` |
| `tools/mic_probe.py` | Record or reuse a WAV while probing STT settings. | `.\.venv\Scripts\python.exe scripts\tools\mic_probe.py --help` |
| `eval/record_eval_clips.py` | Record prompted or spontaneous clips for ASR evaluation. | `.\.venv\Scripts\python.exe scripts\eval\record_eval_clips.py --help` |
| `tools/repair_transcripts.py` | Repair merged transcript spacing in copied session JSON. | `.\.venv\Scripts\python.exe scripts\tools\repair_transcripts.py --help` |
| `eval/review_grammar_flags.py` | Export flags for human review or score completed verdicts. | `.\.venv\Scripts\python.exe scripts\eval\review_grammar_flags.py --help` |
| `eval/review_narrative.py` | Export the written review's claims for human checking, or score the verdicts (E4). | `.\.venv\Scripts\python.exe scripts\eval\review_narrative.py --help` |
| `run_web.py` | Start the local FastAPI interview application. | `.\.venv\Scripts\python.exe scripts\run_web.py --help` |
| `tools/show_audio_levels.py` | Inspect audio-level and word-confidence measures per turn. | `.\.venv\Scripts\python.exe scripts\tools\show_audio_levels.py $analyzed.FullName` |
| `tools/show_word_gaps.py` | Print recognized word timestamps and adjacent gaps. | `.\.venv\Scripts\python.exe scripts\tools\show_word_gaps.py $session.FullName 1` |
| `demos/transcribe_demo.py` | Print detailed local transcription output for a WAV. | `.\.venv\Scripts\python.exe scripts\demos\transcribe_demo.py tests\fixtures\harvard.wav` |
| `demos/tts_demo.py` | Synthesize and play a test utterance with the configured Piper voice. | `.\.venv\Scripts\python.exe scripts\demos\tts_demo.py --help` |
| `eval/latency_report.py` | Summarize per-stage and end-of-answer-to-reply latency from saved sessions. | `.\.venv\Scripts\python.exe scripts\eval\latency_report.py data\sessions` |

`capture_demo.py`, `evaluate_stt.py`, `live_interview.py`, `mic_probe.py`,
`record_eval_clips.py`, and `tts_demo.py` may require audio hardware or model
files. LLM-dependent tools need `GROQ_API_KEY`. ASR benchmark and evaluation
commands require matching files under the ignored `data/` directory.
