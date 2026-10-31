"""Diagnose why CTranslate2 cannot load cublas64_12.dll on Windows.

Run from the project root with the venv active:
    python scripts/tools/cuda_check.py

It reports (A) where the NVIDIA DLLs are and whether other copies sit on PATH,
(B) whether each DLL can be loaded and, if not, the exact Windows error, and
(C) whether a real GPU transcription call works once the DLLs are preloaded.
"""

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

# DLLs that faster-whisper's GPU path needs, in dependency order.
WANTED = ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll")


def find_dll_dirs() -> dict:
    """Map every folder under site-packages/nvidia that holds DLLs to its DLL names."""
    spec = importlib.util.find_spec("nvidia")
    found: dict = {}
    if spec is None or not spec.submodule_search_locations:
        return found
    for root in spec.submodule_search_locations:
        for dll in Path(root).rglob("*.dll"):
            found.setdefault(dll.parent, []).append(dll.name)
    return found


def main() -> None:
    if sys.platform != "win32":
        print("This diagnostic is for Windows only.")
        return

    print(f"Python {sys.version.split()[0]} | venv: {sys.prefix}")

    print("\n=== A. Where are the DLLs? ===")
    dll_dirs = find_dll_dirs()
    if not dll_dirs:
        print("No DLLs found under site-packages/nvidia.")
    for folder, names in sorted(dll_dirs.items()):
        wanted_here = [n for n in names if n in WANTED]
        print(f"{folder}  ({len(names)} DLLs; needed here: {wanted_here or 'none'})")

    print("\nOther copies of cublas64_12.dll already on PATH (possible version clashes):")
    clashes = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and (Path(p) / "cublas64_12.dll").exists()]
    print("\n".join(f"  {p}" for p in clashes) if clashes else "  (none)")

    print("\n=== B. Load each DLL directly ===")
    for folder in dll_dirs:
        os.add_dll_directory(str(folder))
    os.environ["PATH"] = os.pathsep.join([str(d) for d in dll_dirs] + [os.environ.get("PATH", "")])
    for name in WANTED:
        path = next((d / name for d in dll_dirs if (d / name).exists()), None)
        if path is None:
            print(f"{name}: NOT FOUND in the nvidia packages")
            continue
        try:
            ctypes.WinDLL(str(path))  # stays loaded; later lookups by name reuse it
            print(f"{name}: loaded OK")
        except OSError as error:
            print(f"{name}: FAILED -> {error}")

    print("\n=== C. Real GPU transcription call (2 s of silence, model 'base') ===")
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("base", device="cuda", compute_type="float16")
        segments, _ = model.transcribe(np.zeros(32000, dtype=np.float32), language="en")
        list(segments)  # transcription runs while iterating
        print("GPU transcription call: OK")
    except Exception as error:  # report anything, this is a diagnostic
        print(f"GPU transcription call: FAILED -> {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
