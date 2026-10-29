"""Tests for the memory-friendly report-pass speech model loader."""

from __future__ import annotations

import sys
import types

import pytest

from src.speaksure import speech_to_text as stt


class FakeWhisperModel:
    created: list[dict[str, str]] = []
    fail_on_cuda = False

    def __init__(self, size: str, device: str, compute_type: str) -> None:
        if device == "cuda" and FakeWhisperModel.fail_on_cuda:
            raise RuntimeError("CUDA out of memory")
        FakeWhisperModel.created.append(
            {"size": size, "device": device, "compute_type": compute_type}
        )


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr(stt, "_configure_windows_cuda_dlls", lambda: None)
    FakeWhisperModel.created = []
    FakeWhisperModel.fail_on_cuda = False
    stt._cached_report_model.cache_clear()
    yield
    stt._cached_report_model.cache_clear()


def test_gpu_report_model_uses_int8_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stt, "resolve_local_device", lambda preference: "cuda")
    _model, device, compute = stt.load_report_model("large-v3", "auto")
    assert (device, compute) == ("cuda", "int8_float16")
    assert FakeWhisperModel.created == [
        {"size": "large-v3", "device": "cuda", "compute_type": "int8_float16"}
    ]


def test_gpu_failure_raises_instead_of_falling_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stt, "resolve_local_device", lambda preference: "cuda")
    FakeWhisperModel.fail_on_cuda = True
    with pytest.raises(stt.LocalModelError):
        stt.load_report_model("large-v3", "auto")
    assert FakeWhisperModel.created == []  # no silent CPU model was created


def test_cpu_device_reports_cpu_so_the_pass_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stt, "resolve_local_device", lambda preference: "cpu")
    _model, device, compute = stt.load_report_model("large-v3", "cpu")
    assert (device, compute) == ("cpu", "int8")


def test_report_model_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stt, "resolve_local_device", lambda preference: "cuda")
    first = stt.load_report_model("large-v3", "auto")
    second = stt.load_report_model("large-v3", "auto")
    assert first[0] is second[0]
    assert len(FakeWhisperModel.created) == 1
