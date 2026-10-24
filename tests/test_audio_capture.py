import io
import sys
import wave

import numpy as np
import pytest

from src.speaksure.audio_capture import (
    FRAME_SAMPLES,
    FRAME_SECONDS,
    SAMPLE_RATE,
    CaptureSettings,
    NoiseProfile,
    _build_noise_profile,
    calibrate_microphone,
    capture_answer,
    describe_input_device,
    wav_bytes,
)


def _frame(rms: float, phase: float = 0.0) -> np.ndarray:
    angle = np.arange(FRAME_SAMPLES, dtype=np.float32) * (2 * np.pi / 48) + phase
    samples = np.sin(angle) + 0.5 * np.sin(2 * angle) + 0.2 * np.sin(3 * angle)
    return (samples * (rms / np.sqrt(np.mean(samples**2)))).astype(np.float32)


def _sequence(kind: str, count: int, rms: float = 0.1) -> list[np.ndarray]:
    if kind == "silence":
        return [_frame(0.002, index) for index in range(count)]
    if kind == "speech":
        return [_frame(rms, index) for index in range(count)]
    if kind == "noise":
        random = np.random.default_rng(1234)
        return [
            (random.normal(0, rms, FRAME_SAMPLES)).astype(np.float32)
            for _ in range(count)
        ]
    raise AssertionError(kind)


def _settings(**overrides: object) -> CaptureSettings:
    values = {
        "calibration_seconds": 1.0,
        "threshold_multiplier": 3.0,
        "min_threshold": 0.008,
        "release_ratio": 0.7,
        "onset_frames": 4,
        "preroll_seconds": 0.3,
        "tail_seconds": 0.3,
        "end_silence_seconds": 0.3,
        "no_speech_timeout_seconds": 5.0,
        "max_answer_seconds": 10.0,
    }
    values.update(overrides)
    return CaptureSettings(**values)


def test_calibration_quiet_speech_and_quiet_detects_end_of_speech():
    frames = (
        _sequence("silence", 33)
        + _sequence("silence", 67)
        + _sequence("speech", 100)
        + _sequence("silence", 67)
    )

    result = capture_answer(
        iter(frames),
        _settings(end_silence_seconds=0.6, no_speech_timeout_seconds=8.0),
    )

    assert result.stop_reason == "silence"
    assert result.thinking_time_seconds == pytest.approx(2.0, abs=0.1)
    assert result.answer_duration_seconds == pytest.approx(3.0, abs=0.1)


def test_pause_inside_speech_does_not_end_answer():
    frames = (
        _sequence("silence", 34)
        + _sequence("speech", 50)
        + _sequence("silence", 27)
        + _sequence("speech", 50)
        + _sequence("silence", 55)
    )

    result = capture_answer(
        iter(frames),
        _settings(end_silence_seconds=1.5, no_speech_timeout_seconds=5.0),
    )

    assert result.stop_reason == "silence"
    assert result.answer_duration_seconds == pytest.approx(3.81, abs=0.1)


def test_three_frame_click_does_not_start_speech():
    frames = _sequence("silence", 34) + _sequence("speech", 3) + _sequence("silence", 166)

    result = capture_answer(
        iter(frames), _settings(no_speech_timeout_seconds=5.0)
    )

    assert result.stop_reason == "no_speech"
    assert result.samples.size == 0


def test_silence_only_source_times_out_without_speech():
    frames = iter(_sequence("silence", 34 + 100))

    result = capture_answer(
        frames, _settings(no_speech_timeout_seconds=1.5)
    )

    assert result.stop_reason == "no_speech"
    assert result.thinking_time_seconds == 1.5
    assert result.answer_duration_seconds == 0.0
    assert result.samples.size == 0


def test_continuous_speech_stops_at_max_answer_duration():
    frames = iter(
        _sequence("silence", 34)
        + _sequence("speech", 100)
    )

    result = capture_answer(
        frames,
        _settings(max_answer_seconds=1.0, no_speech_timeout_seconds=5.0),
    )

    assert result.stop_reason == "max_duration"
    assert result.answer_duration_seconds == pytest.approx(1.0, abs=FRAME_SECONDS)
    assert result.samples.size >= FRAME_SAMPLES * 33


def test_noisier_room_raises_threshold_but_detects_louder_speech():
    random = np.random.default_rng(42)
    calibration_noise = [
        random.normal(0, 0.02, FRAME_SAMPLES).astype(np.float32)
        for _ in range(34)
    ]
    room_noise = [
        random.normal(0, 0.02, FRAME_SAMPLES).astype(np.float32)
        for _ in range(100)
    ]
    noise_only = capture_answer(
        iter(calibration_noise + room_noise),
        _settings(no_speech_timeout_seconds=1.5),
    )
    assert noise_only.stop_reason == "no_speech"

    random = np.random.default_rng(42)
    room_noise = [
        random.normal(0, 0.02, FRAME_SAMPLES).astype(np.float32)
        for _ in range(34 + 40)
    ]
    frames = room_noise + _sequence("speech", 30, rms=0.2) + _sequence("silence", 20)

    result = capture_answer(
        iter(frames),
        _settings(
            end_silence_seconds=0.3,
            no_speech_timeout_seconds=5.0,
            max_answer_seconds=5.0,
        ),
    )

    assert result.noise_rms > 0.015
    assert result.threshold > 0.008
    assert result.stop_reason == "silence"
    assert result.answer_duration_seconds > 0.8


def test_source_ending_mid_answer_returns_captured_samples():
    frames = iter(_sequence("silence", 34) + _sequence("speech", 40))

    result = capture_answer(
        frames, _settings(no_speech_timeout_seconds=5.0)
    )

    assert result.stop_reason == "source_ended"
    assert result.answer_duration_seconds > 1.0
    assert result.samples.size > 0


def test_capture_answer_closes_generator_source():
    closed = False

    def generate():
        nonlocal closed
        try:
            yield from _sequence("silence", 34 + 5)
        finally:
            closed = True

    result = capture_answer(
        generate(), _settings(no_speech_timeout_seconds=0.1)
    )

    assert result.stop_reason == "no_speech"
    assert closed


def test_precalibrated_profile_skips_frames_and_times_thinking_from_first_frame():
    """A supplied profile leaves all response frames available for detection."""
    profile = NoiseProfile(noise_rms=0.002, threshold=0.008)
    events = []
    frames = (
        _sequence("silence", 67)
        + _sequence("speech", 100)
        + _sequence("silence", 67)
    )

    result = capture_answer(
        iter(frames),
        _settings(end_silence_seconds=0.6, no_speech_timeout_seconds=8.0),
        on_event=lambda event, details: events.append((event, details)),
        noise_profile=profile,
    )

    assert result.stop_reason == "silence"
    assert result.thinking_time_seconds == pytest.approx(2.0, abs=0.1)
    assert result.answer_duration_seconds == pytest.approx(3.0, abs=0.1)
    assert result.noise_rms == profile.noise_rms
    assert result.threshold == profile.threshold
    assert events[0] == (
        "calibrated",
        {"noise_rms": profile.noise_rms, "threshold": profile.threshold},
    )


def test_shared_profile_helper_matches_in_capture_calibration_for_same_frames():
    """Shared calibration arithmetic produces identical captured profile values."""
    settings = _settings(end_silence_seconds=0.6, no_speech_timeout_seconds=8.0)
    calibration_frames = _sequence("silence", 33)
    response_frames = (
        _sequence("silence", 67)
        + _sequence("speech", 100)
        + _sequence("silence", 67)
    )
    profile = _build_noise_profile(calibration_frames, settings)

    captured = capture_answer(
        iter(calibration_frames + response_frames),
        settings,
    )

    assert captured.noise_rms == profile.noise_rms
    assert captured.threshold == profile.threshold


def test_calibrate_microphone_consumes_calibration_frames_and_uses_shared_helper(
    monkeypatch,
):
    """Microphone calibration reads the configured frames through shared math."""
    settings = _settings(calibration_seconds=0.3)
    frames = _sequence("silence", 10)
    consumed = []

    def generate():
        for frame in frames:
            consumed.append(frame)
            yield frame

    monkeypatch.setattr(
        "src.speaksure.audio_capture.microphone_frames",
        lambda device=None: generate(),
    )

    profile = calibrate_microphone(settings=settings, device=2)

    expected = _build_noise_profile(consumed, settings)
    assert len(consumed) == round(settings.calibration_seconds / FRAME_SECONDS)
    assert profile == expected


@pytest.mark.parametrize(
    "setting",
    [
        {"calibration_seconds": 0},
        {"threshold_multiplier": -1},
        {"min_threshold": 0},
        {"release_ratio": 0},
        {"release_ratio": 1.01},
        {"onset_frames": 0},
        {"preroll_seconds": -0.1},
        {"tail_seconds": 0},
        {"end_silence_seconds": 0},
        {"no_speech_timeout_seconds": 0},
        {"max_answer_seconds": float("nan")},
    ],
)
def test_capture_settings_reject_invalid_values(setting):
    with pytest.raises(ValueError):
        CaptureSettings(**setting)


def test_wav_bytes_round_trip_is_mono_16_bit_with_expected_length():
    samples = np.linspace(-1.2, 1.2, SAMPLE_RATE // 2, dtype=np.float32)
    data = wav_bytes(samples)

    with wave.open(io.BytesIO(data), "rb") as wav_file:
        assert wav_file.getframerate() == SAMPLE_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == len(samples)
        assert len(wav_file.readframes(wav_file.getnframes())) == len(samples) * 2


def test_describe_input_device_by_index_and_default(monkeypatch):
    devices = {
        2: {
            "name": "USB microphone",
            "default_samplerate": 48000.0,
            "hostapi": 1,
        },
        0: {
            "name": "Default microphone",
            "default_samplerate": 16000.0,
            "hostapi": 0,
        },
    }
    sounddevice = type(
        "SoundDevice",
        (),
        {
            "default": type("Default", (), {"device": (0, 3)})(),
            "query_devices": staticmethod(lambda index: devices[index]),
            "query_hostapis": staticmethod(
                lambda: [{"name": "MME"}, {"name": "WASAPI"}]
            ),
        },
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)

    assert describe_input_device(2) == {
        "name": "USB microphone",
        "index": 2,
        "default_samplerate": 48000.0,
        "host_api": "WASAPI",
    }
    assert describe_input_device(None) == {
        "name": "Default microphone",
        "index": 0,
        "default_samplerate": 16000.0,
        "host_api": "MME",
    }


@pytest.mark.parametrize("missing", [False, True])
def test_describe_input_device_returns_unknown_on_failures(monkeypatch, missing):
    if missing:
        monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
        monkeypatch.setitem(sys.modules, "sounddevice", None)
    else:
        sounddevice = type(
            "SoundDevice",
            (),
            {
                "default": type("Default", (), {"device": (0, 0)})(),
                "query_devices": staticmethod(
                    lambda _index: (_ for _ in ()).throw(RuntimeError("unavailable"))
                ),
            },
        )
        monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)

    assert describe_input_device(None) == {"name": "unknown"}
