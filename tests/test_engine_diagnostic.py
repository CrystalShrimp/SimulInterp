import json
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from simulinterp.engine import SessionWorker
from simulinterp.segmenter import SpeechChunk


def worker():
    instance = SessionWorker(dict(language="en", device="cuda", device_id=0,
                                  model_size="large-v3-turbo", translation_threads=8))
    instance._capture = SimpleNamespace(statistics=lambda: {})
    return instance


def test_manual_tap_starts_at_current_audio_and_keeps_asr_before_translation(tmp_path):
    engine = worker()
    engine._tap_diagnostic_audio(np.zeros(16000, np.float32))
    assert not list(tmp_path.iterdir()), "no pre-recording or files before opt-in"
    assert engine.request_diagnostic_start(tmp_path, {"audio_source": "test"})
    assert not engine.request_diagnostic_start(tmp_path), "one recorder at a time"
    try:
        engine._tap_diagnostic_audio(np.full(320, .2, np.float32))
        chunk = SpeechChunk(np.zeros(320, np.float32), 1, 1.02)
        engine._record_diagnostic_transcript(chunk, SimpleNamespace(text="source English", start=0, end=.02), .1)
    finally:
        engine.request_diagnostic_stop(wait=True)
    folder = next(tmp_path.iterdir())
    with wave.open(str(folder / "audio.wav")) as audio:
        assert audio.getnframes() == 320
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["course_start_seconds"] == 1
    assert meta["recognition"]["beam_size"] == 3
    assert meta["segmentation"]["profile"] == "adaptive"
    assert meta["segmentation"]["max_seconds"] == 8.0
    assert meta["segmentation"]["silence_seconds"] == .8
    assert meta["segmentation"]["soft_min_seconds"] == 4.5
    assert meta["segmentation"]["soft_silence_seconds"] == .3
    assert meta["audio_source"] == "test"
    record = json.loads((folder / "transcripts.jsonl").read_text(encoding="utf-8"))
    assert record["text"] == "source English"
    assert record["diagnostic_chunk_start"] == 0
    assert record["asr_seconds"] == .1


def test_pause_finalizes_sample_and_recording_never_resumes_implicitly(tmp_path):
    engine = worker()
    assert engine.request_diagnostic_start(tmp_path)
    engine._tap_diagnostic_audio(np.zeros(320, np.float32))
    engine._pause_event.set()
    engine.paused.connect(engine.request_stop)
    engine._wait_while_paused()
    assert not engine._diagnostic_recorder.active
    engine._tap_diagnostic_audio(np.zeros(320, np.float32))
    with wave.open(str(next(tmp_path.iterdir()) / "audio.wav")) as audio:
        assert audio.getnframes() == 320


def test_no_diagnostic_before_device_ready_or_during_pause_and_end(tmp_path):
    engine = worker()
    engine._capture = None
    assert not engine.request_diagnostic_start(tmp_path)
    engine._capture = SimpleNamespace(request_stop=lambda: None)
    engine.request_pause()
    assert not engine.request_diagnostic_start(tmp_path)
    engine.request_stop()
    assert not engine.request_diagnostic_start(tmp_path)
    assert not list(tmp_path.iterdir())


def test_phase_timing_records_only_opted_in_chunks_with_matching_audio_origin(tmp_path):
    engine = worker()
    chunk = SpeechChunk(np.zeros(320, np.float32), 1, 1.02, available_at=1.7)
    timing = dict(endpoint_wait_seconds=.68, queue_wait_seconds=.2, asr_seconds=.3,
                  translation_seconds=.7, processing_seconds=1.0,
                  enqueue_to_emit_seconds=1.2, result="subtitle_emitted")
    engine._record_diagnostic_latency(chunk, timing)
    assert not list(tmp_path.iterdir())
    engine._tap_diagnostic_audio(np.zeros(16000, np.float32))
    assert engine.request_diagnostic_start(tmp_path)
    try:
        engine._tap_diagnostic_audio(np.ones(320, np.float32) * .1)
        engine._record_diagnostic_latency(SpeechChunk(np.zeros(320), .1, .2), timing)
        engine._record_diagnostic_transcript(chunk, SimpleNamespace(text="English", start=0, end=.02), .3)
        engine._record_diagnostic_latency(chunk, timing)
    finally:
        engine.request_diagnostic_stop(wait=True)
    folder = next(tmp_path.iterdir())
    rows = [json.loads(s) for s in (folder / "latency.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["diagnostic_chunk_start"] == 0
    assert rows[0]["diagnostic_chunk_end"] == pytest.approx(.02)
    assert rows[0]["queue_wait_seconds"] == .2
    assert rows[0]["translation_seconds"] == .7
    assert len((folder / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["latency_format"]["version"] == 1
    assert "GUI paint" in metadata["latency_format"]["limitations"]
