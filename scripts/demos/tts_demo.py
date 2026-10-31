"""Synthesize and play text using the configured offline Piper voice."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speaksure.audio_capture import save_wav  # noqa: E402
from speaksure.text_to_speech import (  # noqa: E402
    SpeechSynthesisConfig,
    SpeechSynthesisError,
    build_speech_synthesizer,
    ensure_voice,
    list_output_devices,
    play_audio,
)


def main() -> int:
    """Run offline speech synthesis, optionally saving or playing the audio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--voice")
    parser.add_argument("--length-scale", type=float)
    parser.add_argument("--leading-silence", type=float)
    parser.add_argument("--trailing-silence", type=float)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--list-output-devices", action="store_true")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    if args.list_output_devices:
        devices = list_output_devices()
        if not devices:
            print("No output devices found.")
        for index, name in devices:
            print(f"{index}: {name}")
        return 0
    if args.text is None:
        parser.error("text is required unless --list-output-devices is used.")

    try:
        config = SpeechSynthesisConfig.from_env()
        overrides = {}
        if args.voice is not None:
            overrides["voice_name"] = args.voice
        if args.length_scale is not None:
            overrides["length_scale"] = args.length_scale
        if args.leading_silence is not None:
            overrides["leading_silence_seconds"] = args.leading_silence
        if args.trailing_silence is not None:
            overrides["trailing_silence_seconds"] = args.trailing_silence
        config = replace(config, **overrides)
        print(
            "Playback padding: "
            f"{config.leading_silence_seconds:.3f}s leading, "
            f"{config.trailing_silence_seconds:.3f}s trailing"
        )

        if args.download:
            print(
                f"Ensuring voice '{config.voice_name}' is available in "
                f"{config.models_dir}..."
            )
            model_path = ensure_voice(config.voice_name, config.models_dir)
            print(f"Voice ready: {model_path}")

        synthesizer = build_speech_synthesizer(config)
        audio = synthesizer.synthesize(args.text)
        print(f"Synthesis time: {audio.synthesis_seconds:.3f} seconds")
        print(f"Audio duration: {audio.duration_seconds:.3f} seconds")
        print(f"Real-time factor: {audio.real_time_factor:.3f}")

        if args.save is not None:
            saved_path = save_wav(args.save, audio.samples, audio.sample_rate)
            print(f"Saved: {saved_path}")

        playback_seconds = 0.0
        if not args.no_play:
            playback_seconds = play_audio(
                audio,
                device=args.device,
                leading_silence_seconds=config.leading_silence_seconds,
                trailing_silence_seconds=config.trailing_silence_seconds,
            )
        print(f"Playback time: {playback_seconds:.3f} seconds")
        return 0
    except (SpeechSynthesisError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
