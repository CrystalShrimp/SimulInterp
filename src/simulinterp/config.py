from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SIMULINTERP_HOME", PROJECT_ROOT)).resolve()
MODEL_ROOT = DATA_ROOT / "models"
# Short DE/EN CPU benchmark: 8 threads lowered median latency vs 6 by ~12%.
# The settings dialog keeps 4/6 available when lower CPU load is preferred.
DEFAULT_TRANSLATION_THREADS = 8
TRANSLATION_THREAD_OPTIONS = (4, 6, 8)
DEFAULT_ASR_PROFILE = "adaptive"
# Keep model precision, beam and both speech filters identical. Only phrase
# boundaries change; each profile has a fixed memory/latency bound.
ASR_PROFILES = {
    "adaptive": {"max_seconds": 8.0, "silence_seconds": 0.8,
                 "pre_roll_seconds": 0.2, "min_speech_seconds": 0.18,
                 "soft_min_seconds": 4.5, "soft_silence_seconds": 0.3},
    "coherent": {"max_seconds": 8.0, "silence_seconds": 0.8,
                 "pre_roll_seconds": 0.2, "min_speech_seconds": 0.18},
    "fast": {"max_seconds": 5.0, "silence_seconds": 0.5,
             "pre_roll_seconds": 0.2, "min_speech_seconds": 0.18},
}


def segmentation_options(profile: str = DEFAULT_ASR_PROFILE) -> dict:
    if not isinstance(profile, str) or profile not in ASR_PROFILES:
        raise ValueError("字幕节奏必须为清晰收音 adaptive、较差收音 coherent 或低延迟 fast")
    return dict(ASR_PROFILES[profile])


ASR_REPOS = {
    "large-v3-turbo": "dropbox-dash/faster-whisper-large-v3-turbo",
    "small": "Systran/faster-whisper-small",
}
TRANSLATION_REPO = "facebook/m2m100_418M"
# Community conversion of the MIT-licensed Meta model. Pin the complete snapshot
# and every inference asset, rather than trusting a mutable repository branch.
TRANSLATION_CT2_REPO = "jncraton/m2m100_418M-ct2-int8"
TRANSLATION_CT2_REVISION = "7c1b2620a4e58dacecbd8bf89cfd6da7eb9eb7b0"
TRANSLATION_SHA256 = {
    "model.bin": "d6703dd9f920ff896e45c3d97b490761bed5944937b90bbe6a7245f5652542d4",
    "config.json": "4244772990e30069563e3ddfb4ad6dc95bdfd2ac3de667ea8858c9b0a8433fa8",
    "generation_config.json": "aed76366507333ddbb8bd49960f23c82fe6446b3319a46a54befdb45324ccf61",
    "shared_vocabulary.json": "7eb5d0ff184c6095c7c10f9911c0aea492250abd12854f9c3d787c64b1c6397e",
    "special_tokens_map.json": "c1a4f86c3874d279ae1b2a05162858db5dd6c61665d84223ed886cbcff08fda6",
    "tokenizer_config.json": "ae54f15f0649bb05041cdadad8485ba1faf40bc33e6b4c2a74ae2d1ae5710fa2",
    "vocab.json": "b6e77e474aeea8f441363aca7614317c06381f3eacfe10fb9856d5081d1074cc",
    "sentencepiece.bpe.model": "d8f7c76ed2a5e0822be39f0a4f95a55eb19c78f4593ce609e2edbc2aea4d380a",
}
ASR_REQUIRED = ("model.bin", "config.json", "tokenizer.json")
TRANSLATION_REQUIRED = (
    "model.bin", "config.json", "shared_vocabulary.json", "sentencepiece.bpe.model",
    "vocab.json", "tokenizer_config.json", "special_tokens_map.json",
)


def asr_path(model_size: str) -> Path:
    if model_size not in ASR_REPOS:
        raise ValueError(f"Unsupported ASR model: {model_size}")
    return MODEL_ROOT / f"whisper-{model_size}"


def translation_path() -> Path:
    return MODEL_ROOT / "m2m100-418m-int8"


def missing_files(path: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (path / name).is_file() or (path / name).stat().st_size == 0]


def model_status(model_size: str, *, translate_enabled: bool = True) -> tuple[bool, str]:
    if type(translate_enabled) is not bool:
        raise ValueError("translate_enabled 必须为布尔值")
    asr_missing = missing_files(asr_path(model_size), ASR_REQUIRED)
    mt_missing = missing_files(translation_path(), TRANSLATION_REQUIRED) if translate_enabled else []
    problems = []
    if asr_missing:
        problems.append("语音识别模型")
    if mt_missing:
        problems.append("中文翻译模型")
    return (False, "缺少" + "、".join(problems) + "，请先准备模型") if problems else (True, "本地模型已就绪 · 可离线使用")
