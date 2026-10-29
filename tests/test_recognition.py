"""ASR contract tests using an in-memory faster-whisper stand-in."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from simulinterp.recognition import LocalRecognizer, Transcript


def segment(text, start=0.0, end=1.0, *, no_speech=0.1, logprob=-0.2):
    return SimpleNamespace(
        text=text, start=start, end=end,
        no_speech_prob=no_speech, avg_logprob=logprob,
    )


@pytest.fixture
def whisper_stub(monkeypatch):
    state = SimpleNamespace(loads=[], calls=[], segments=[], language="en", generator_starts=0, yielded=0)

    class Model:
        def __init__(self, model_path, **options):
            state.loads.append((model_path, options))

        def transcribe(self, samples, **options):
            state.calls.append((samples, options))

            def segments():
                state.generator_starts += 1
                for item in state.segments:
                    state.yielded += 1
                    yield item

            return segments(), SimpleNamespace(language=state.language)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    return state


@pytest.mark.parametrize("device,compute_type", [("cuda", "int8_float16"), ("cpu", "int8")])
def test_load_uses_only_local_weights_and_hardware_precision(whisper_stub, device, compute_type):
    recognizer = LocalRecognizer(Path("models") / "whisper-small", device)
    recognizer.load()
    model_path, options = whisper_stub.loads[0]
    assert model_path == str(Path("models") / "whisper-small")
    assert options["device"] == device
    assert options["compute_type"] == compute_type
    assert options["local_files_only"] is True


@pytest.mark.parametrize("language,forced", [("de", "de"), ("en", "en")])
def test_whisper_transcribes_source_language_without_english_translation(whisper_stub, language, forced):
    whisper_stub.language = language
    whisper_stub.segments = [segment(" Guten Tag. ", 0.25, 1.3)]
    recognizer = LocalRecognizer(Path("local"))
    recognizer.load()
    audio = np.zeros(32000, dtype=np.float32)
    result = recognizer.transcribe(audio, language)
    supplied_audio, options = whisper_stub.calls[0]
    assert supplied_audio is audio
    assert options["task"] == "transcribe"
    assert options["language"] == forced
    assert options["beam_size"] == 3
    assert options["vad_filter"] is True
    assert options["condition_on_previous_text"] is False
    assert result == Transcript("Guten Tag.", whisper_stub.language, 0.25, 1.3)


def test_filter_drops_blank_and_low_confidence_non_speech(whisper_stub):
    whisper_stub.segments = [
        segment("  ", 0, 0.2),
        segment(" hallucination ", 0.2, 0.5, no_speech=0.95, logprob=-1.5),
        segment(" Hello ", 0.7, 1.2),
        segment(" world. ", 1.3, 2.1),
        segment("another hallucination", 2.2, 3, no_speech=0.9, logprob=-2),
    ]
    recognizer = LocalRecognizer(Path("local"))
    recognizer.load()
    assert recognizer.transcribe(np.zeros(16000, dtype=np.float32), "en") == Transcript(
        "Hello world.", "en", 0.7, 2.1,
    )


@pytest.mark.parametrize(
    "no_speech,logprob",
    [(0.9, -0.2), (0.1, -2), (0.6, -1.1), (0.9, -1.0)],
)
def test_filter_requires_both_non_speech_and_low_confidence(whisper_stub, no_speech, logprob):
    whisper_stub.segments = [segment("Keep this.", no_speech=no_speech, logprob=logprob)]
    recognizer = LocalRecognizer(Path("local"))
    recognizer.load()
    assert recognizer.transcribe(np.zeros(16000, dtype=np.float32), "en").text == "Keep this."


@pytest.mark.parametrize("segments", [[], [segment(" ")], [segment("noise", no_speech=0.9, logprob=-2)]])
def test_no_usable_speech_has_no_transcript(whisper_stub, segments):
    whisper_stub.segments = segments
    recognizer = LocalRecognizer(Path("local"))
    recognizer.load()
    assert recognizer.transcribe(np.zeros(16000, dtype=np.float32), "en") is None


@pytest.mark.parametrize("language", ["auto", "fr", None, ""])
def test_language_must_be_explicit_before_whisper_is_called(whisper_stub, language):
    recognizer = LocalRecognizer(Path("local"))
    recognizer.load()
    with pytest.raises(ValueError, match="指定课程语言"):
        recognizer.transcribe(np.zeros(16000, dtype=np.float32), language)
    assert not whisper_stub.calls


def test_inference_requires_a_loaded_model():
    with pytest.raises(RuntimeError, match="加载"):
        LocalRecognizer(Path("local")).transcribe(np.empty(0, dtype=np.float32), "en")


def test_close_releases_model(whisper_stub):
    recognizer = LocalRecognizer(Path("local"))
    recognizer.load()
    recognizer.close()
    assert recognizer._model is None
    with pytest.raises(RuntimeError, match="加载"):
        recognizer.transcribe(np.empty(0, dtype=np.float32), "en")


def test_invalid_device_fails_before_loading_any_model():
    with pytest.raises(ValueError, match="device"):
        LocalRecognizer(Path("local"), device="invalid")


def test_cuda_warmup_consumes_lazy_generator_and_discards_text(whisper_stub):
    whisper_stub.segments = [segment("discard warmup output")]
    recognizer = LocalRecognizer(Path("local"), device="cuda")
    recognizer.load()
    assert recognizer.warmup() is None
    assert whisper_stub.generator_starts == 1
    assert whisper_stub.yielded == 1
    assert len(whisper_stub.calls) == 1
    audio, options = whisper_stub.calls[0]
    assert audio.dtype == np.float32
    assert audio.shape == (16000,)
    assert not np.any(audio)
    assert options == {
        "language": "en", "task": "transcribe", "beam_size": 1,
        "temperature": 0.0, "condition_on_previous_text": False,
        "vad_filter": False, "without_timestamps": True, "max_new_tokens": 1,
        "no_speech_threshold": None, "log_prob_threshold": None,
        "compression_ratio_threshold": None,
    }


@pytest.mark.parametrize("loaded", [False, True])
def test_cpu_warmup_does_not_execute_inference(whisper_stub, loaded):
    recognizer = LocalRecognizer(Path("local"), device="cpu")
    if loaded:
        recognizer.load()
    assert recognizer.warmup() is None
    assert not whisper_stub.calls
    assert whisper_stub.generator_starts == 0


def test_cuda_warmup_requires_loaded_model():
    with pytest.raises(RuntimeError, match="加载"):
        LocalRecognizer(Path("local"), device="cuda").warmup()
