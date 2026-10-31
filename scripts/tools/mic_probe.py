"""Record from the default microphone (or reuse a saved clip) and transcribe it
with settings you can change from the command line.

Purpose: find out WHY real-microphone transcription is worse than the clean
Harvard sample, by changing one setting at a time on the SAME recording.

Examples (run from the project root, venv active):
    python scripts/tools/mic_probe.py --seconds 15 --out data/probe/clip1.wav
    python scripts/tools/mic_probe.py --reuse data/probe/clip1.wav --language auto
    python scripts/tools/mic_probe.py --reuse data/probe/clip1.wav --language en
    python scripts/tools/mic_probe.py --reuse data/probe/clip1.wav --language en --model small
    python scripts/tools/mic_probe.py --reuse data/probe/clip1.wav --language en --hotwords "Pradeep, SpeakSure, Tirupati"
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000  # Whisper models expect 16 kHz mono audio.


def record(seconds: int) -> np.ndarray:
    """Record mono 16-bit audio from the default input device."""
    import sounddevice as sd

    print(f"Recording {seconds} s - start speaking now...")
    samples = sd.rec(
        int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16"
    )
    sd.wait()
    print("Recording finished.")
    return samples.reshape(-1)


def save_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1:
            raise ValueError("Probe clips must be 16 kHz mono WAV (use this script to record them).")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)


def report_levels(samples: np.ndarray) -> None:
    """Print loudness statistics so bad input levels are visible."""
    scaled = samples.astype(np.float32) / 32768.0
    peak = float(np.abs(scaled).max())
    rms = float(np.sqrt(np.mean(scaled**2)))
    print(f"Duration: {len(samples) / SAMPLE_RATE:.1f} s | peak: {peak:.3f} | rms: {rms:.4f}")
    if peak >= 0.99:
        print("WARNING: clipping - the input is too loud or the mic gain is too high.")
    if rms < 0.01:
        print("WARNING: very quiet - raise the mic level or move closer to the mic.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=int, default=15, help="recording length")
    parser.add_argument("--out", type=Path, default=Path("data/probe/clip.wav"), help="where to save a new recording")
    parser.add_argument("--reuse", type=Path, help="transcribe this existing clip instead of recording")
    parser.add_argument("--model", default="base", help="tiny, base, small, medium ...")
    parser.add_argument("--language", default="en", help="'en' (default) or 'auto' to let Whisper detect it")
    parser.add_argument("--hotwords", default="", help="comma-separated names/terms to bias towards")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    if args.reuse:
        samples = load_wav(args.reuse)
        print(f"Reusing {args.reuse}")
    else:
        samples = record(args.seconds)
        save_wav(args.out, samples)
        print(f"Saved to {args.out} (this folder is git-ignored, keep it private)")
    report_levels(samples)

    # Reuse the app's own Windows CUDA setup (adds the NVIDIA pip packages' DLL
    # folders to this process's PATH). Without it, cublas64_12.dll is not found.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from speaksure.speech_to_text import _configure_windows_cuda_dlls

    _configure_windows_cuda_dlls()

    from faster_whisper import WhisperModel

    compute_type = "float16" if args.device == "cuda" else "int8"
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    audio = samples.astype(np.float32) / 32768.0
    language = None if args.language == "auto" else args.language

    start = time.perf_counter()
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language=language,
        hotwords=args.hotwords or None,
        vad_filter=True,
        word_timestamps=True,
    )
    segments = list(segments)  # transcription actually runs while iterating
    elapsed = time.perf_counter() - start

    print("\n--- settings ---")
    print(f"model={args.model} device={args.device} language={args.language} hotwords={args.hotwords!r}")
    print(f"detected language: {info.language} (probability {info.language_probability:.2f})")
    print(f"transcription time: {elapsed:.2f} s")

    print("\n--- transcript ---")
    print(" ".join(seg.text.strip() for seg in segments))

    print("\n--- words Whisper was unsure about (probability < 0.60) ---")
    shaky = [(w.word.strip(), w.probability) for seg in segments for w in seg.words if w.probability < 0.60]
    if shaky:
        for word, prob in shaky:
            print(f"{word:<20} {prob:.2f}")
    else:
        print("(none)")


if __name__ == "__main__":
    main()