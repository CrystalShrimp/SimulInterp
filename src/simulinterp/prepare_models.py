"""Explicit online setup; normal inference never downloads models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

from .config import (
    ASR_REPOS, TRANSLATION_REPO, TRANSLATION_CT2_REPO, TRANSLATION_CT2_REVISION,
    TRANSLATION_SHA256, TRANSLATION_REQUIRED, asr_path, translation_path,
    model_status, missing_files,
)


def announce(message: str) -> None:
    print(message, flush=True)


def prepare_asr(model_size: str) -> None:
    from huggingface_hub import snapshot_download

    destination = asr_path(model_size)
    announce(f"下载 / 校验 Whisper {model_size} …")
    snapshot_download(
        ASR_REPOS[model_size], local_dir=str(destination),
        allow_patterns=["model.bin", "config.json", "tokenizer.json", "vocabulary.*", "preprocessor_config.json", "README.md", "LICENSE*"],
        max_workers=2,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_mismatches(directory: Path, *, existing_only: bool = False) -> list[str]:
    """Check the full pinned manifest without loading any model into memory."""
    mismatches = []
    for name, expected in TRANSLATION_SHA256.items():
        path = directory / name
        if not path.is_file():
            if not existing_only:
                mismatches.append(name)
        elif sha256_file(path) != expected:
            mismatches.append(name)
    return mismatches


def publish_directory(staging: Path, destination: Path) -> None:
    """Publish a verified sibling directory, retaining any previous install.

    Windows cannot replace a populated directory in one rename. Preserve the
    existing directory as a backup and restore it if the second rename fails.
    No directory is recursively deleted.
    """
    root = destination.parent.resolve()
    resolved_source = staging.resolve()
    resolved_destination = destination.resolve()
    if (
        resolved_source.parent != root
        or resolved_destination.parent != root
        or resolved_source == resolved_destination
    ):
        raise ValueError("模型发布目录必须位于同一模型文件夹下")
    backup = None
    if destination.exists():
        backup = destination.with_name(destination.name + ".previous-" + uuid.uuid4().hex[:8])
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    if backup is not None:
        announce(f"原模型已保留：{backup}")


def write_verified_provenance(directory: Path) -> None:
    (directory / "provenance.json").write_text(json.dumps({
        "source": TRANSLATION_CT2_REPO,
        "revision": TRANSLATION_CT2_REVISION,
        "original_source": TRANSLATION_REPO,
        "license": "MIT",
        "quantization": "int8",
        "sha256": TRANSLATION_SHA256,
        "verified": True,
    }, indent=2), encoding="utf-8")


def prepare_translation(convert_official: bool = False) -> None:
    """Download verified int8 weights, or explicitly convert Meta's originals."""
    if convert_official:
        _convert_official_translation()
        return
    destination = translation_path()
    if not missing_files(destination, tuple(TRANSLATION_SHA256)):
        announce("校验本地中文翻译模型 SHA256 …")
        if not hash_mismatches(destination):
            write_verified_provenance(destination)
            announce("中文翻译模型校验通过，无需下载。")
            return
    from huggingface_hub import snapshot_download

    # Keep this fixed sibling directory across interrupted runs so HF can
    # resume incomplete downloads; never expose it as a ready runtime model.
    staging = destination.with_name(destination.name + ".download")
    if staging.resolve().parent != destination.parent.resolve():
        raise ValueError("模型暂存目录不能指向模型文件夹之外")
    staging.mkdir(parents=True, exist_ok=True)
    corrupted = hash_mismatches(staging, existing_only=True)
    announce("下载 M2M100 418M INT8 翻译模型（约 500 MB，无需 PyTorch）…")
    snapshot_download(
        TRANSLATION_CT2_REPO,
        revision=TRANSLATION_CT2_REVISION,
        local_dir=str(staging),
        allow_patterns=[*TRANSLATION_SHA256, "README.md", "LICENSE*"],
        max_workers=2,
        force_download=bool(corrupted),
    )
    announce("校验翻译模型与分词器 SHA256 …")
    mismatches = hash_mismatches(staging)
    if mismatches:
        raise RuntimeError(
            "翻译模型 SHA256 校验失败：" + "、".join(mismatches)
            + "。原有模型未更改；重新运行将重新下载损坏文件。"
        )
    write_verified_provenance(staging)
    publish_directory(staging, destination)
    announce(f"翻译模型准备完成：{destination}")


def _convert_official_translation() -> None:

    destination = translation_path()
    provenance = destination / "provenance.json"
    try:
        previous_source = json.loads(provenance.read_text(encoding="utf-8")).get("source")
    except (OSError, ValueError):
        previous_source = None
    if not missing_files(destination, TRANSLATION_REQUIRED) and previous_source == TRANSLATION_REPO:
        announce("中文翻译模型已就绪，跳过转换。")
        return
    try:
        import torch  # noqa: F401 -- needed only for this one-time conversion
    except ImportError as exc:
        raise RuntimeError("官方模型转换需要 CPU PyTorch；请用当前 Python 安装 torch>=2.6，或省略 --convert-official 下载已转换模型。") from exc
    from huggingface_hub import snapshot_download
    from ctranslate2.converters import TransformersConverter
    from transformers import M2M100Tokenizer
    announce("下载 M2M100 418M 官方翻译模型（首次约 2 GB）…")
    source = snapshot_download(
        TRANSLATION_REPO,
        allow_patterns=["config.json", "generation_config.json", "pytorch_model.bin", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "sentencepiece.bpe.model", "README.md", "LICENSE*"],
        max_workers=2,
    )
    announce("在 CPU 上转换为 INT8；此步骤可能需要几分钟 …")
    # A complete destination is only published after conversion succeeds.
    staging = destination.with_name(destination.name + ".building-" + uuid.uuid4().hex[:8])
    converter = TransformersConverter(source)
    converter.convert(str(staging), quantization="int8")
    M2M100Tokenizer.from_pretrained(source, local_files_only=True).save_pretrained(staging)
    if missing_files(staging, TRANSLATION_REQUIRED):
        raise RuntimeError(f"转换输出不完整：{staging}")
    (staging / "provenance.json").write_text(json.dumps({"source": TRANSLATION_REPO, "snapshot": Path(source).name, "quantization": "int8"}), encoding="utf-8")
    publish_directory(staging, destination)
    announce(f"翻译模型准备完成：{destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr", choices=list(ASR_REPOS), default="large-v3-turbo")
    parser.add_argument("--check", action="store_true", help="Only check local model files")
    parser.add_argument("--asr-only", action="store_true", help="Prepare/check only speech recognition for source-only subtitles")
    parser.add_argument("--convert-official", action="store_true", help="Convert original Meta weights locally (requires CPU torch>=2.6 and an extra ~2 GB download)")
    args = parser.parse_args()
    if args.asr_only and args.convert_official:
        parser.error("--asr-only 不能与 --convert-official 同时使用")
    if args.check:
        ready, message = (model_status(args.asr, translate_enabled=False) if args.asr_only
                          else model_status(args.asr))
        announce(message)
        return 0 if ready else 1
    # This process is the explicit exception to runtime's offline-only policy.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from .runtime import bootstrap
    bootstrap(offline=False)
    try:
        prepare_asr(args.asr)
        if not args.asr_only:
            prepare_translation(convert_official=args.convert_official)
        ready, message = (model_status(args.asr, translate_enabled=False) if args.asr_only
                          else model_status(args.asr))
        announce(message)
        return 0 if ready else 1
    except Exception as exc:
        announce(f"准备失败：{exc}\n可重新执行以续传；请检查网络、磁盘空间和 setup.ps1 安装日志。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
