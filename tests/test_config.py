"""Local readiness checks require complete, nonempty model assets."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from simulinterp import config


ASR_FILES = ("model.bin", "config.json", "tokenizer.json")
TRANSLATION_FILES = (
    "model.bin", "config.json", "shared_vocabulary.json", "sentencepiece.bpe.model",
    "vocab.json", "tokenizer_config.json", "special_tokens_map.json",
)


@pytest.fixture
def model_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_ROOT", tmp_path / "models")
    return config.MODEL_ROOT


def populate(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"nonempty-model-asset")


def test_empty_install_reports_both_models_missing(model_root):
    ready, message = config.model_status("large-v3-turbo")
    assert ready is False
    assert "语音识别模型" in message
    assert "中文翻译模型" in message


@pytest.mark.parametrize("model_size", ["large-v3-turbo", "small"])
def test_complete_local_models_are_ready_without_network(model_root, model_size):
    populate(config.asr_path(model_size), ASR_FILES)
    populate(config.translation_path(), TRANSLATION_FILES)
    ready, message = config.model_status(model_size)
    assert ready is True
    assert "离线" in message


def test_missing_translation_does_not_report_asr_missing(model_root):
    populate(config.asr_path("small"), ASR_FILES)
    ready, message = config.model_status("small")
    assert ready is False
    assert "中文翻译模型" in message
    assert "语音识别模型" not in message


def test_source_only_is_ready_with_asr_assets_and_no_translation(model_root):
    populate(config.asr_path("small"), ASR_FILES)
    assert config.model_status("small", translate_enabled=False)[0]
    assert not config.model_status("small")[0]
    (config.asr_path("small") / "model.bin").unlink()
    assert not config.model_status("small", translate_enabled=False)[0]


def test_prepare_source_only_never_downloads_translation(monkeypatch, model_root):
    from simulinterp import prepare_models
    calls = []
    def speech(model_size):
        calls.append(model_size)
        populate(config.asr_path(model_size), ASR_FILES)
    monkeypatch.setattr(prepare_models, "prepare_asr", speech)
    monkeypatch.setattr(prepare_models, "prepare_translation", lambda **_: pytest.fail("source-only must not prepare MT"))
    monkeypatch.setattr(sys, "argv", ["prepare_models", "--asr", "small", "--asr-only"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    # Avoid changing the calling test process's offline environment.
    monkeypatch.setattr("simulinterp.runtime.bootstrap", lambda **_: None)
    assert prepare_models.main() == 0
    assert calls == ["small"]
    monkeypatch.setattr(sys, "argv", ["prepare_models", "--asr", "small", "--asr-only", "--check"])
    assert prepare_models.main() == 0


def test_selected_asr_model_must_be_present(model_root):
    populate(config.asr_path("large-v3-turbo"), ASR_FILES)
    populate(config.translation_path(), TRANSLATION_FILES)
    assert config.model_status("large-v3-turbo")[0] is True
    ready, message = config.model_status("small")
    assert ready is False
    assert "语音识别模型" in message
    assert "中文翻译模型" not in message


@pytest.mark.parametrize("missing_name", ASR_FILES)
def test_each_required_asr_asset_is_checked(model_root, missing_name):
    populate(config.asr_path("small"), ASR_FILES)
    populate(config.translation_path(), TRANSLATION_FILES)
    (config.asr_path("small") / missing_name).unlink()
    assert config.model_status("small")[0] is False


@pytest.mark.parametrize("missing_name", TRANSLATION_FILES)
def test_each_required_translation_asset_is_checked(model_root, missing_name):
    populate(config.asr_path("small"), ASR_FILES)
    populate(config.translation_path(), TRANSLATION_FILES)
    (config.translation_path() / missing_name).unlink()
    assert config.model_status("small")[0] is False


def test_zero_byte_interrupted_download_is_not_ready(model_root):
    populate(config.asr_path("small"), ASR_FILES)
    populate(config.translation_path(), TRANSLATION_FILES)
    (config.translation_path() / "model.bin").write_bytes(b"")
    assert config.model_status("small")[0] is False


def test_missing_files_treats_directories_as_missing(model_root):
    model_root.mkdir(parents=True)
    (model_root / "model.bin").mkdir()
    (model_root / "config.json").write_bytes(b"{}")
    assert config.missing_files(model_root, ("model.bin", "config.json", "tokenizer.json")) == [
        "model.bin", "tokenizer.json",
    ]


def test_unknown_model_is_rejected(model_root):
    with pytest.raises(ValueError, match="Unsupported"):
        config.asr_path("unknown-model")


@pytest.fixture
def tiny_manifest(monkeypatch):
    from simulinterp import prepare_models

    payloads = {
        name: ("verified-data:" + name).encode("utf-8")
        for name in (*TRANSLATION_FILES, "generation_config.json")
    }
    monkeypatch.setattr(prepare_models, "TRANSLATION_SHA256", {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    })
    return payloads


def fill_payloads(directory, payloads):
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)


def test_checksum_detects_missing_and_corrupted_assets(model_root, tiny_manifest):
    from simulinterp import prepare_models

    directory = config.translation_path()
    fill_payloads(directory, tiny_manifest)
    assert prepare_models.hash_mismatches(directory) == []
    (directory / "model.bin").write_bytes(b"corrupt")
    (directory / "shared_vocabulary.json").unlink()
    assert set(prepare_models.hash_mismatches(directory)) == {"model.bin", "shared_vocabulary.json"}
    assert prepare_models.hash_mismatches(directory, existing_only=True) == ["model.bin"]


def test_valid_pinned_install_skips_network_and_records_provenance(monkeypatch, model_root, tiny_manifest):
    from simulinterp import prepare_models

    def unexpected_download(*args, **kwargs):
        pytest.fail("A complete hash-verified model must not use the network")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=unexpected_download))
    destination = config.translation_path()
    fill_payloads(destination, tiny_manifest)
    prepare_models.prepare_translation()
    provenance = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["verified"] is True
    assert provenance["revision"] == config.TRANSLATION_CT2_REVISION
    assert provenance["source"] == config.TRANSLATION_CT2_REPO


def test_download_uses_pinned_revision_and_verifies_before_publication(monkeypatch, model_root, tiny_manifest):
    from simulinterp import prepare_models

    calls = []

    def download(repo, **options):
        calls.append((repo, options))
        assert not config.translation_path().exists()
        fill_payloads(Path(options["local_dir"]), tiny_manifest)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    prepare_models.prepare_translation()
    assert calls[0][0] == config.TRANSLATION_CT2_REPO
    assert calls[0][1]["revision"] == config.TRANSLATION_CT2_REVISION
    assert set(tiny_manifest).issubset(calls[0][1]["allow_patterns"])
    assert prepare_models.hash_mismatches(config.translation_path()) == []
    assert not config.translation_path().with_name("m2m100-418m-int8.download").exists()


def test_bad_download_does_not_replace_existing_model(monkeypatch, model_root, tiny_manifest):
    from simulinterp import prepare_models

    destination = config.translation_path()
    destination.mkdir(parents=True)
    (destination / "user-note.txt").write_text("preserve this", encoding="utf-8")

    def download(repo, **options):
        staging = Path(options["local_dir"])
        fill_payloads(staging, tiny_manifest)
        (staging / "model.bin").write_bytes(b"damaged download")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.raises(RuntimeError, match="SHA256"):
        prepare_models.prepare_translation()
    assert (destination / "user-note.txt").read_text(encoding="utf-8") == "preserve this"
    assert not (destination / "model.bin").exists()


def test_corrupt_staging_is_redownloaded_and_previous_install_is_preserved(monkeypatch, model_root, tiny_manifest):
    from simulinterp import prepare_models

    destination = config.translation_path()
    destination.mkdir(parents=True)
    (destination / "user-note.txt").write_text("keep", encoding="utf-8")
    staging = destination.with_name(destination.name + ".download")
    fill_payloads(staging, tiny_manifest)
    (staging / "model.bin").write_bytes(b"damaged")

    def download(repo, **options):
        assert options["force_download"] is True
        fill_payloads(Path(options["local_dir"]), tiny_manifest)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    prepare_models.prepare_translation()
    assert prepare_models.hash_mismatches(destination) == []
    backups = list(model_root.glob("m2m100-418m-int8.previous-*"))
    assert len(backups) == 1
    assert (backups[0] / "user-note.txt").read_text(encoding="utf-8") == "keep"


def test_publication_failure_restores_previous_install(monkeypatch, model_root):
    from simulinterp import prepare_models

    destination = config.translation_path()
    staging = destination.with_name(destination.name + ".download")
    destination.mkdir(parents=True)
    staging.mkdir()
    (destination / "user-note.txt").write_text("keep", encoding="utf-8")
    original_rename = Path.rename

    def fail_staging_rename(path, target):
        if path == staging:
            raise PermissionError("injected publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_rename)
    with pytest.raises(PermissionError, match="injected"):
        prepare_models.publish_directory(staging, destination)
    assert (destination / "user-note.txt").read_text(encoding="utf-8") == "keep"
    assert staging.exists()
