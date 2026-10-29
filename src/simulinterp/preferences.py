"""Validated local settings, independent of Qt and the Windows registry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from . import config


def default_preferences() -> dict[str, Any]:
    return {
        "language": None,  # First use requires an explicit course language.
        "model_size": "large-v3-turbo",
        "device": "cuda",
        "asr_profile": config.DEFAULT_ASR_PROFILE,
        "translate_enabled": True,
        "translation_threads": config.DEFAULT_TRANSLATION_THREADS,
    }


def validated_preferences(values: Any) -> dict[str, Any]:
    result = default_preferences()
    if not isinstance(values, dict):
        return result
    if values.get("language") in ("de", "en"):
        result["language"] = values["language"]
    if isinstance(values.get("model_size"), str) and values["model_size"] in config.ASR_REPOS:
        result["model_size"] = values["model_size"]
    if values.get("device") in ("cuda", "cpu"):
        result["device"] = values["device"]
    profile = values.get("asr_profile")
    if isinstance(profile, str) and profile in config.ASR_PROFILES:
        result["asr_profile"] = profile
    if type(values.get("translate_enabled")) is bool:
        result["translate_enabled"] = values["translate_enabled"]
    threads = values.get("translation_threads")
    if type(threads) is int and threads in config.TRANSLATION_THREAD_OPTIONS:
        result["translation_threads"] = threads
    return result


def load_preferences(path: Path | None = None) -> dict[str, Any]:
    path = path if path is not None else config.DATA_ROOT / "settings.json"
    try:
        if path.stat().st_size > 64 * 1024:
            return default_preferences()
        values = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default_preferences()
    return validated_preferences(values)


def save_preferences(values: dict[str, Any], path: Path | None = None) -> None:
    """Publish a complete JSON atomically; leave the old file on write failure."""
    path = path if path is not None else config.DATA_ROOT / "settings.json"
    clean = validated_preferences(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=".settings-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(clean, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
