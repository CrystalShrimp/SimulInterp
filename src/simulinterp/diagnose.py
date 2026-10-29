"""Print local installation checks. This command never opens a recording stream."""
from __future__ import annotations

import importlib.metadata
import sys

from .runtime import bootstrap
from .config import model_status, DATA_ROOT


def main() -> int:
    bootstrap()
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Data: {DATA_ROOT}")
    failed = False
    for name in ("PySide6-Essentials", "faster-whisper", "ctranslate2", "transformers", "PyAudioWPatch", "webrtcvad-wheels", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: not installed")
    try:
        import ctranslate2
        print(f"CUDA devices: {ctranslate2.get_cuda_device_count()}")
        print(f"CPU compute types: {sorted(ctranslate2.get_supported_compute_types('cpu'))}")
        if ctranslate2.get_cuda_device_count():
            print(f"CUDA compute types: {sorted(ctranslate2.get_supported_compute_types('cuda'))}")
    except Exception as exc:
        print(f"Inference runtime error: {exc}")
        failed = True
    try:
        from .audio import list_audio_devices
        for device in list_audio_devices():
            print(f"Audio {device.index}: [{device.kind}] {device.name} ({device.sample_rate} Hz){' [default]' if device.is_default else ''}")
    except Exception as exc:
        print(f"Audio error: {exc}")
        failed = True
    for size in ("large-v3-turbo", "small"):
        ready, message = model_status(size)
        print(f"{size}: {'ready' if ready else 'missing'} — {message}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
