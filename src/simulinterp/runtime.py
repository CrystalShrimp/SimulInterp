"""Locate project-local CUDA DLLs before importing CTranslate2 on Windows."""
from __future__ import annotations

import os
import site
from pathlib import Path

_DLL_HANDLES: list = []


def bootstrap(*, offline: bool = True) -> None:
    from .config import DATA_ROOT

    os.environ.setdefault("HF_HOME", str(DATA_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.name != "nt" or _DLL_HANDLES:
        return
    roots = [Path(p) for p in site.getsitepackages()]
    paths = []
    for root in roots:
        for relative in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin", "torch/lib"):
            directory = root / relative
            if directory.is_dir():
                paths.append(str(directory))
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    if paths:
        os.environ["PATH"] = os.pathsep.join(paths + [os.environ.get("PATH", "")])
