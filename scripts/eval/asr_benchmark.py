"""Benchmark local Whisper model accuracy and speed against a clip manifest."""

from __future__ import annotations

import argparse
import csv
import gc
import io
import random
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import jiwer

from speaksure import speech_to_text
from speaksure.evaluation.asr import (
    load_manifest,
    name_hit_rate,
    normalize_text,
    summarize,
    word_level_errors,
)
from speaksure.speech_to_text import (
    LocalSpeechToTextProvider,
    LocalTranscriptionConfig,
    TranscriptionResult,
    describe_stt_configuration,
)

CSV_FIELDS = (
    "model",
    "clip",
    "condition",
    "reference",
    "hypothesis",
    "wer",
    "substitutions",
    "deletions",
    "insertions",
    "name_hit",
    "name_hits",
    "name_count",
    "transcribe_seconds",
    "load_plus_first_run_seconds",
    "audio_seconds",
    "real_time_factor",
    "mean_word_probability",
    "word_probability_sum",
    "word_count",
    "actual_device",
    "warmup",
    "device_mismatch",
    "peak_gpu_memory_mb",
    "hotwords",
    "input_device",
    "input_device_id",
    "category",
    "notes",
    "accept",
    "word_level_errors",
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "manifest.csv"
DEFAULT_MODELS = ("base", "small", "medium", "large-v3")
GROUP_FIELDS = frozenset({"model", "input_device", "category"})


class BenchmarkProvider(Protocol):
    """Provider interface consumed by the benchmark runner."""

    actual_device: str

    def transcribe_detailed(
        self, audio: bytes, *, filename: str, content_type: str
    ) -> TranscriptionResult:
        """Transcribe one clip and return text, word scores, and device metadata."""
        ...


def _audio_duration(audio: bytes, clip: Path) -> float:
    """Return clip duration in seconds from its WAV stream."""
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError) as error:
        raise ValueError(f"Could not read WAV duration for {clip}: {error}") from error


def _torch_module() -> object | None:
    """Import Torch only when installed, keeping it optional for CPU-only systems."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _gpu_peak_memory_mb(torch_module: object | None) -> float | None:
    """Return Torch's CUDA peak allocation in MiB when CUDA is available."""
    if torch_module is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    return float(cuda.max_memory_allocated()) / (1024 * 1024)


def _clear_torch_cuda_cache(torch_module: object | None) -> None:
    """Release Torch CUDA cache when an importable Torch build exposes CUDA."""
    if torch_module is None:
        return
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def run_benchmark(
    manifest_rows: Sequence[Mapping[str, object]],
    models: Sequence[str],
    *,
    device: str = "cuda",
    language: str = "en",
    hotwords: str = "",
    provider_factory: Callable[
        [LocalTranscriptionConfig], BenchmarkProvider
    ] = LocalSpeechToTextProvider,
    torch_module: object | None = None,
) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    """Run each model over all clips, isolating failures and device mismatches."""
    torch_module = _torch_module() if torch_module is None else torch_module
    result_rows: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    for model_size in models:
        config = LocalTranscriptionConfig(
            model_size=model_size,
            device=device,
            language=language,
            hotwords=hotwords,
        )
        description = describe_stt_configuration(config)
        print(
            f"STT: model={description['model']} "
            f"device={description['device_setting']}"
        )
        model_rows: list[dict[str, object]] = []
        provider: BenchmarkProvider | None = None
        try:
            provider = provider_factory(config)
            cuda = getattr(torch_module, "cuda", None) if torch_module else None
            if cuda is not None and cuda.is_available():
                cuda.reset_peak_memory_stats()

            for clip_index, manifest_row in enumerate(manifest_rows):
                clip_path = Path(str(manifest_row["clip"]))
                reference = str(manifest_row["reference"])
                names_value = manifest_row.get("names", [])
                names = (
                    [str(name) for name in names_value]
                    if isinstance(names_value, (list, tuple))
                    else []
                )
                audio = clip_path.read_bytes()
                audio_seconds = _audio_duration(audio, clip_path)
                started = time.perf_counter()
                transcription = provider.transcribe_detailed(
                    audio,
                    filename=clip_path.name,
                    content_type="audio/wav",
                )
                elapsed = time.perf_counter() - started
                actual_device = str(
                    getattr(provider, "actual_device", transcription.device)
                )
                if not actual_device or actual_device == "not-initialized":
                    actual_device = str(transcription.device)
                counts = jiwer.process_words(
                    normalize_text(reference), normalize_text(transcription.text)
                )
                valid_names = [name for name in names if normalize_text(name)]
                hit_rate = name_hit_rate(transcription.text, valid_names)
                hit_count = (
                    round(hit_rate * len(valid_names)) if hit_rate is not None else 0
                )
                word_probabilities = [
                    float(word.probability) for word in transcription.words
                ]
                probability_sum = sum(word_probabilities)
                is_warmup = clip_index == 0
                mismatch = device.lower() == "cuda" and actual_device.lower() != "cuda"
                model_rows.append(
                    {
                        "model": model_size,
                        "clip": str(clip_path),
                        "condition": str(manifest_row.get("condition", "")),
                        "input_device": str(manifest_row.get("input_device", "")),
                        "input_device_id": str(
                            manifest_row.get("input_device_id", "")
                        ),
                        "category": str(manifest_row.get("category", "")),
                        "notes": str(manifest_row.get("notes", "")),
                        "accept": str(manifest_row.get("accept", "")),
                        "reference": reference,
                        "hypothesis": transcription.text,
                        "word_level_errors": word_level_errors(
                            reference, transcription.text
                        ),
                        "wer": counts.wer,
                        "substitutions": counts.substitutions,
                        "deletions": counts.deletions,
                        "insertions": counts.insertions,
                        "name_hit": hit_rate,
                        "name_hits": hit_count,
                        "name_count": len(valid_names),
                        "transcribe_seconds": elapsed,
                        "load_plus_first_run_seconds": elapsed if is_warmup else "",
                        "audio_seconds": audio_seconds,
                        "real_time_factor": elapsed / audio_seconds if audio_seconds else "",
                        "mean_word_probability": (
                            probability_sum / len(word_probabilities)
                            if word_probabilities
                            else ""
                        ),
                        "word_probability_sum": probability_sum,
                        "word_count": len(word_probabilities),
                        "actual_device": actual_device,
                        "warmup": is_warmup,
                        "device_mismatch": mismatch,
                        "peak_gpu_memory_mb": "",
                        "hotwords": hotwords,
                    }
                )

            peak_memory = _gpu_peak_memory_mb(torch_module)
            if peak_memory is not None:
                for row in model_rows:
                    row["peak_gpu_memory_mb"] = peak_memory
            if any(bool(row["device_mismatch"]) for row in model_rows):
                for row in model_rows:
                    row["device_mismatch"] = True
                print(
                    f"Skipping {model_size} summary: requested {device}, "
                    f"provider reported {', '.join(sorted({str(r['actual_device']) for r in model_rows}))}."
                )
            result_rows.extend(model_rows)
        except Exception as error:
            failures.append((model_size, f"{type(error).__name__}: {error}"))
            print(f"Skipping {model_size}: {type(error).__name__}: {error}", file=sys.stderr)
        finally:
            if provider is not None:
                del provider
            speech_to_text._cached_local_model.cache_clear()
            gc.collect()
            _clear_torch_cuda_cache(torch_module)
    return result_rows, failures


def _format_metric(value: object, digits: int = 3) -> str:
    """Format an optional numeric table value without inventing measurements."""
    if value is None or value == "":
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def summary_table(
    results: Sequence[Mapping[str, object]],
    group_by: Sequence[str] = ("model",),
) -> str:
    """Render the selected benchmark group summaries as a Markdown table."""
    summaries = summarize(results, group_by)
    headers = (
        "Group",
        "Reference words",
        "Pooled WER",
        "Extended WER",
        "Mean clip WER",
        "Median clip WER",
        "Name hit rate",
        "Median warm RTF",
        "Load + first run (s)",
        "Mean word probability",
        "Actual device(s)",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for model, data in summaries.items():
        values = (
            model,
            str(data["reference_words"]),
            _format_metric(data["pooled_wer"]),
            _format_metric(data["extended_wer"]),
            _format_metric(data["mean_clip_wer"]),
            _format_metric(data["median_clip_wer"]),
            _format_metric(data["name_hit_rate"]),
            _format_metric(data["warm_real_time_factor_median"]),
            _format_metric(data["load_plus_first_run_seconds"]),
            _format_metric(data["mean_word_probability"]),
            ", ".join(data["actual_devices"]),
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def per_clip_report(results: Sequence[Mapping[str, object]]) -> str:
    """Render each clip's source text, recognized text, and aligned word errors."""
    lines: list[str] = []
    for row in results:
        lines.extend(
            (
                f"- {row.get('model', '')} / {Path(str(row.get('clip', ''))).name}",
                f"  Reference: {row.get('reference', '')}",
                f"  Hypothesis: {row.get('hypothesis', '')}",
                f"  Word errors: {row.get('word_level_errors', 'none')}",
            )
        )
    return "\n".join(lines)


def write_outputs(
    output_path: Path,
    results: Sequence[Mapping[str, object]],
    group_by: Sequence[str] = ("model",),
    *, excluded: Sequence[str] = (), normalize: str = "basic", seed: int = 1234,
) -> Path:
    """Write benchmark CSV and its adjacent Markdown summary without overwriting."""
    csv_path = output_path.expanduser()
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = csv_path.with_suffix(".md")
    if csv_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing benchmark output: {csv_path} or {markdown_path}"
        )
    with csv_path.open("x", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    with markdown_path.open("x", encoding="utf-8", newline="") as output_file:
        output_file.write("# ASR benchmark summary\n\n")
        output_file.write(summary_table(results, group_by))
        output_file.write(f"\n\nExcluded clips: {', '.join(excluded) if excluded else 'none'}\n")
        output_file.write("\n\n## Per-clip transcripts and word errors\n\n")
        output_file.write(per_clip_report(results))
        output_file.write("\n")
    if normalize in {"basic", "extended"}:
        report_dir = PROJECT_ROOT / "docs" / "results"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "e1_asr.md"
        if not report_path.exists():
            report_path.write_text(e1_report(results, excluded, seed), encoding="utf-8")
    return markdown_path


def _ci(rows: Sequence[Mapping[str, object]], *, extended: bool, seed: int) -> tuple[float, float]:
    from speaksure.evaluation.asr import bootstrap_wer
    pairs = [(str(row.get("reference", "")), str(row.get("hypothesis", ""))) for row in rows]
    return bootstrap_wer(pairs, resamples=2000, seed=seed, extended=extended)


def _fmt_ci(rows: Sequence[Mapping[str, object]], extended: bool, seed: int) -> str:
    from speaksure.evaluation.asr import pooled_wer
    pairs = [(str(row.get("reference", "")), str(row.get("hypothesis", ""))) for row in rows]
    value = float(pooled_wer(pairs, extended=extended)["wer"])
    low, high = _ci(rows, extended=extended, seed=seed)
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def e1_report(results: Sequence[Mapping[str, object]], excluded: Sequence[str], seed: int) -> str:
    """Build the requested E1 result tables from benchmark CSV rows."""
    from speaksure.evaluation.asr import normalize_text, normalize_extended, pooled_wer
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in results:
        if str(row.get("device_mismatch", "")).lower() in {"true", "1"}:
            continue
        groups.setdefault(str(row.get("model", "")), []).append(row)
    lines = ["# E1 ASR benchmark results", "", f"Bootstrap: 2,000 clip resamples; seed={seed}. Excluded clips: {', '.join(excluded) if excluded else 'none'}.", "", "## Model comparison", "", "| Model | Clips | Ref words | Raw WER (95% CI) | Extended WER (95% CI) | RTF | Mean word probability | Load + first run (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, rows in groups.items():
        words = sum(len(normalize_text(str(r.get("reference", ""))).split()) for r in rows)
        rtf = [float(r["real_time_factor"]) for r in rows if r.get("real_time_factor") not in (None, "") and str(r.get("warmup", "")).lower() not in {"true", "1"}]
        probs = [float(r["mean_word_probability"]) for r in rows if r.get("mean_word_probability") not in (None, "")]
        loads = [float(r["load_plus_first_run_seconds"]) for r in rows if r.get("load_plus_first_run_seconds") not in (None, "")]
        lines.append(f"| {model} | {len(rows)} | {words} | {_fmt_ci(rows, False, seed)} | {_fmt_ci(rows, True, seed)} | {sum(rtf)/len(rtf):.3f} | {sum(probs)/len(probs):.3f} | {sum(loads)/len(loads):.3f} |" if rtf and probs and loads else f"| {model} | {len(rows)} | {words} | {_fmt_ci(rows, False, seed)} | {_fmt_ci(rows, True, seed)} | - | - | - |")
    for label, field in (("Per-category", "category"), ("Per-device", "input_device")):
        lines += ["", f"## {label}", "", "| Group | Model | Clips | Ref words | Raw WER | Extended WER |", "|---|---|---:|---:|---:|---:|"]
        buckets: dict[tuple[str, str], list[Mapping[str, object]]] = {}
        for model, rows in groups.items():
            for row in rows: buckets.setdefault((str(row.get(field, "") or "unspecified"), model), []).append(row)
        for (group, model), rows in buckets.items():
            words = sum(len(normalize_text(str(r.get("reference", ""))).split()) for r in rows)
            lines.append(f"| {group} | {model} | {len(rows)} | {words} | {_fmt_ci(rows, False, seed)} | {_fmt_ci(rows, True, seed)} |")
    lines += ["", "## Paired comparisons", "", "| First model vs second | Mean WER difference (first − second) | 95% interval | Fraction first better |", "|---|---:|---:|---:|"]
    by_model_clip = {(str(r.get("model")), Path(str(r.get("clip", ""))).name): r for r in results}
    for first, second in (("medium", "large-v3"), ("small", "medium")):
        clips = sorted({clip for model, clip in by_model_clip if model == first} & {clip for model, clip in by_model_clip if model == second})
        diffs = []
        if clips:
            rng = random.Random(seed)
            for _ in range(2000):
                sample = [rng.choice(clips) for _ in clips]
                diffs.append(sum(float(by_model_clip[(first,c)].get("wer", 0))-float(by_model_clip[(second,c)].get("wer", 0)) for c in sample)/len(sample))
            ordered = sorted(diffs)
            mean_diff = sum(diffs)/len(diffs)
            lines.append(f"| {first} vs {second} | {mean_diff:.3f} | [{ordered[49]:.3f}, {ordered[1949]:.3f}] | {sum(d < 0 for d in diffs)/len(diffs):.3f} |")
    lines += ["", "## First-word analysis", "", "| Model | First word wrong/missing | Other ref words wrong (substitution/deletion) |", "|---|---:|---:|"]
    for model, rows in groups.items():
        first_bad = other_bad = first_n = other_n = 0
        for row in rows:
            ref = normalize_text(str(row.get("reference", ""))).split()
            hyp = normalize_text(str(row.get("hypothesis", ""))).split()
            if not ref: continue
            first_n += 1
            first_bad += int(not hyp or ref[0] != hyp[0])
            # Token-position comparison is intentionally a transparent approximation to aligned S/D counts.
            other_n += max(0, len(ref)-1)
            other_bad += sum(1 for i, word in enumerate(ref[1:], 1) if i >= len(hyp) or hyp[i] != word)
        lines.append(f"| {model} | {first_bad}/{first_n} ({first_bad/first_n:.3f}) | {other_bad}/{other_n} ({other_bad/other_n:.3f}) |" if first_n and other_n else f"| {model} | {first_bad}/{first_n} | {other_bad}/{other_n} |")
    hotword_sets = {str(r.get("hotwords", "")) for r in results}
    if len(hotword_sets) > 1:
        lines += ["", "## Hotword comparison", "", "Hotword values present: " + ", ".join(sorted(hotword_sets))]
    lines += ["", "## Reading guide", "", "Raw WER counts every token mismatch after lowercase and punctuation cleanup.", "Extended WER also equates listed US/UK spellings, clock forms, and digit ordinals.", "Intervals resample whole clips, so each clip contributes its complete reference and hypothesis.", "A negative paired difference means the first listed model has lower WER.", "First-word rates flag a first-position mismatch; other-word rates count remaining reference positions."]
    return "\n".join(lines) + "\n"


def _default_output_path() -> Path:
    """Return a unique timestamped benchmark CSV path under the project data folder."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "data" / "eval" / f"asr_results_{timestamp}.csv"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI options, run the benchmark, and write result and summary files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--language", default="en")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--normalize", choices=("basic", "extended"), default="basic")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--exclude-clips", default="", help="Comma-separated clip names or paths to omit.")
    parser.add_argument("--report-only", type=Path, help="Report from an existing benchmark CSV.")
    parser.add_argument(
        "--group-by",
        default="model",
        help="Comma-separated grouping fields: model, input_device, category (default: model).",
    )
    args = parser.parse_args(argv)

    try:
        manifest_rows = load_manifest(args.manifest)
    except (OSError, ValueError) as error:
        print(f"Could not load ASR manifest: {error}", file=sys.stderr)
        return 1
    excluded = [item.strip() for item in args.exclude_clips.split(",") if item.strip()]
    excluded_names = {Path(item).name for item in excluded}
    manifest_rows = [row for row in manifest_rows if Path(str(row["clip"])).name not in excluded_names and str(row["clip"]) not in excluded]
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    if not models:
        parser.error("--models must contain at least one model size.")
    group_by = [field.strip() for field in args.group_by.split(",") if field.strip()]
    if not group_by or any(field not in GROUP_FIELDS for field in group_by):
        parser.error("--group-by must contain model, input_device, and/or category.")
    if args.report_only:
        with args.report_only.open("r", encoding="utf-8-sig", newline="") as source:
            results = list(csv.DictReader(source))
        failures = []
        if excluded:
            results = [row for row in results if Path(str(row.get("clip", ""))).name not in excluded_names and str(row.get("clip", "")) not in excluded]
    else:
        results, failures = run_benchmark(
        manifest_rows,
        models,
        device=args.device,
        language=args.language,
        hotwords=args.hotwords,
        )
    output_path = args.out or _default_output_path()
    try:
        markdown_path = write_outputs(output_path, results, group_by, excluded=excluded, normalize=args.normalize, seed=args.seed)
    except OSError as error:
        print(f"Could not write ASR benchmark output: {error}", file=sys.stderr)
        return 1
    table = summary_table(results, group_by)
    print("\nASR benchmark summary")
    print(table)
    print("\nPer-clip transcripts and word errors")
    print(per_clip_report(results))
    print(f"\nCSV: {output_path.resolve()}")
    print(f"Markdown: {markdown_path.resolve()}")
    print(f"Skipped models: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
