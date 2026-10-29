from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simulinterp import config, transcribe
from simulinterp.segmenter import SpeechChunk
from simulinterp.subtitles import Subtitle
from simulinterp.transcribe import main, transcribe_file


@pytest.mark.parametrize("arguments", [[], ["--language", "auto"], ["--language", "fr"]])
def test_cli_requires_explicit_supported_language_before_accessing_audio(arguments):
    with pytest.raises(SystemExit) as exc:
        main(["--audio", "missing.wav", *arguments])
    assert exc.value.code == 2


@pytest.mark.parametrize("language", [None, "auto", "fr", ""])
def test_api_rejects_missing_or_automatic_language_before_model_load(language):
    with pytest.raises(ValueError, match="指定课程语言"):
        transcribe_file(Path("missing.wav"), language=language)


def test_api_rejects_unknown_thread_choice_before_model_load():
    with pytest.raises(ValueError, match="线程数"):
        transcribe_file(Path("missing.wav"), language="de", translation_threads=32)


def test_api_rejects_unknown_phrase_profile_before_model_load():
    with pytest.raises(ValueError, match="字幕节奏"):
        transcribe_file(Path("missing.wav"), language="en", asr_profile="unknown")


def test_cli_rejects_unknown_phrase_profile_before_accessing_audio():
    with pytest.raises(SystemExit) as exc:
        main(["--audio", "missing.wav", "--language", "en", "--asr-profile", "unknown"])
    assert exc.value.code == 2


@pytest.mark.parametrize("enabled", [None, "false", 0, 1])
def test_api_rejects_non_boolean_translation_choice_before_model_load(enabled):
    with pytest.raises(ValueError, match="布尔值"):
        transcribe_file(Path("missing.wav"), language="en", translate_enabled=enabled)


@pytest.fixture
def local_pipeline(monkeypatch):
    state = SimpleNamespace(events=[], texts=iter(["The lecture starts.", "The lecture ends."]))

    class Recognizer:
        def __init__(self, *args, **kwargs):
            state.events.append("asr.create")

        def load(self):
            state.events.append("asr.load")

        def transcribe(self, samples, language):
            state.events.append("asr.transcribe")
            return SimpleNamespace(text=next(state.texts), language=language, start=.1, end=.8)

        def close(self):
            state.events.append("asr.close")

    class Translator:
        def __init__(self, *args, **kwargs):
            state.events.append("mt.create")

        def load(self):
            state.events.append("mt.load")

        def translate(self, text, language):
            state.events.append("mt.translate")
            return "中文：" + text

        def close(self):
            state.events.append("mt.close")

    class Segmenter:
        def __init__(self, **kwargs):
            pass

        def feed(self, samples):
            return [SpeechChunk(samples, 10, 11)]

        def flush(self):
            return [SpeechChunk(np.zeros(320, np.float32), 11, 12)]

    monkeypatch.setattr("simulinterp.runtime.bootstrap", lambda **kwargs: None)
    monkeypatch.setattr("simulinterp.recognition.LocalRecognizer", Recognizer)
    monkeypatch.setattr("simulinterp.segmenter.SpeechSegmenter", Segmenter)
    monkeypatch.setitem(sys.modules, "simulinterp.translation", SimpleNamespace(LocalTranslator=Translator))
    monkeypatch.setattr(transcribe, "audio_blocks", lambda _: iter([np.zeros(320, np.float32)]))
    return state


@pytest.mark.parametrize("language", ["en", "de"])
def test_source_only_file_pipeline_never_imports_mt_and_keeps_tail(local_pipeline, monkeypatch, language):
    # An accidental import fails immediately; no MT dependency is needed.
    monkeypatch.setitem(sys.modules, "simulinterp.translation", None)
    messages = []
    entries = transcribe_file(Path("synthetic.wav"), language=language,
                              translate_enabled=False, translation_threads=999,
                              progress=messages.append)
    assert local_pipeline.events == ["asr.create", "asr.load", "asr.transcribe", "asr.transcribe", "asr.close"]
    assert entries == [
        Subtitle(10.1, 10.8, "The lecture starts.", "The lecture starts.", language),
        Subtitle(11.1, 11.8, "The lecture ends.", "The lecture ends.", language),
    ]
    assert "仅原文字幕" in messages[0]


def test_default_file_pipeline_still_translates_and_releases_both_models(local_pipeline):
    entries = transcribe_file(Path("synthetic.wav"), language="en", progress=lambda _: None)
    assert "mt.load" in local_pipeline.events
    assert local_pipeline.events.count("mt.translate") == 2
    assert local_pipeline.events[-2:] == ["asr.close", "mt.close"]
    assert entries[0].source == "The lecture starts."
    assert entries[0].text == "中文：The lecture starts."


def test_cli_no_translation_only_requires_asr_assets_and_writes_original_srt(tmp_path, monkeypatch, local_pipeline):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"decoded by the fake streaming source")
    output = tmp_path / "output.srt"
    model_root = tmp_path / "models"
    monkeypatch.setattr(config, "MODEL_ROOT", model_root)
    asr = config.asr_path("large-v3-turbo")
    asr.mkdir(parents=True)
    for filename in config.ASR_REQUIRED:
        (asr / filename).write_bytes(b"local-asr-asset")
    # No translation assets or importable translation module at all.
    monkeypatch.setitem(sys.modules, "simulinterp.translation", None)
    assert not config.translation_path().exists()
    assert main(["--audio", str(audio), "--language", "en", "--no-translation", "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == (
        "1\n00:00:10,100 --> 00:00:10,800\nThe lecture starts.\n\n"
        "2\n00:00:11,100 --> 00:00:11,800\nThe lecture ends.\n"
    )
    assert not any(event.startswith("mt.") for event in local_pipeline.events)
