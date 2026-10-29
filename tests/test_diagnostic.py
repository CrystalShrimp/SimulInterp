import json
from pathlib import Path
from threading import Event
import time
from unittest.mock import patch
import wave

import numpy as np

from simulinterp.diagnostic import DiagnosticRecorder


def test_recording_is_opt_in_and_does_not_open_device(tmp_path):
    recorder = DiagnosticRecorder(tmp_path, {"language": "en"})
    assert not recorder.submit_audio(np.ones(320, np.float32))
    assert not recorder.record_transcript({"text": "before"})
    assert not recorder.record_latency({"asr_seconds": .1})
    assert not list(tmp_path.iterdir())
    recorder.stop()
    recorder.start()
    assert not recorder.active


def test_exact_five_minutes_bounded_audio_and_final_wav_header(tmp_path):
    recorder = DiagnosticRecorder(tmp_path, {"language": "de"}, queue_size=100).start()
    block = np.full(16000, .2, np.float32)
    for index in range(300):
        while recorder._queue.qsize() >= 80:
            time.sleep(.001)
        assert recorder.submit_audio(block)
    assert not recorder.submit_audio(block)
    assert recorder.stop(timeout=5)
    assert not recorder.active
    with wave.open(str(recorder.path / "audio.wav")) as audio:
        assert audio.getnframes() == 300 * 16000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
    report = json.loads((recorder.path / "report.json").read_text(encoding="utf-8"))
    assert report["completed"] and not report["truncated"]
    assert report["recorded_seconds"] == 300
    assert not list(recorder._queue.queue)


def test_limit_truncates_last_packet_and_stop_retains_text(tmp_path):
    updates = []
    recorder = DiagnosticRecorder(tmp_path, {"language": "en"}, updates.append, max_seconds=.031).start()
    assert recorder.record_transcript({"course_start": 50, "diagnostic_start": 0, "source": "Hello", "text": "你好"})
    assert recorder.submit_audio(np.ones(1000, np.float32))
    assert recorder.stop()
    assert recorder.stop()
    with wave.open(str(recorder.path / "audio.wav")) as audio:
        assert audio.getnframes() == 496
    record = json.loads((recorder.path / "transcripts.jsonl").read_text(encoding="utf-8"))
    assert record["course_start"] == 50 and record["source"] == "Hello"
    assert updates[-1]["state"] == "completed"
    assert not recorder.record_transcript({"text": "late"})
    assert not recorder.record_latency({"asr_seconds": .1})


def test_latency_records_are_separate_and_manual_stop_drains_both_files(tmp_path):
    transcript = {"chunk_start": 20, "chunk_end": 23, "text": "source before translation"}
    latency = {"chunk_start": 20, "chunk_end": 23, "diagnostic_chunk_start": 0,
               "endpoint_wait_seconds": .68, "queue_wait_seconds": .2,
               "asr_seconds": .3, "translation_seconds": .7,
               "processing_seconds": 1.0, "enqueue_to_emit_seconds": 1.2}
    recorder = DiagnosticRecorder(tmp_path, {}).start()
    assert recorder.submit_audio(np.ones(320, np.float32))
    assert recorder.record_transcript(transcript)
    assert recorder.record_latency(latency)
    assert recorder.stop()
    assert json.loads((recorder.path / "transcripts.jsonl").read_text(encoding="utf-8")) == transcript
    assert json.loads((recorder.path / "latency.jsonl").read_text(encoding="utf-8")) == latency
    snapshot = recorder.snapshot()
    report = json.loads((recorder.path / "report.json").read_text(encoding="utf-8"))
    assert snapshot["transcript_count"] == report["transcript_count"] == 1
    assert snapshot["latency_count"] == report["latency_count"] == 1
    assert report["stop_reason"] == "manual_or_course_end"
    assert "not full end-to-end" in report["timing_note"]
    assert not recorder.record_transcript({"text": "late"})
    assert not recorder.record_latency({"translation_seconds": .8})


def test_latency_queue_overflow_invalidates_diagnostic_and_retains_accepted_asr(tmp_path):
    entered, unblock = Event(), Event()
    original = DiagnosticRecorder._run

    def stalled(recorder):
        entered.set()
        unblock.wait(2)
        original(recorder)

    with patch.object(DiagnosticRecorder, "_run", stalled):
        recorder = DiagnosticRecorder(tmp_path, {}, queue_size=1).start()
        assert entered.wait(1)
        assert recorder.record_transcript({"text": "already recognized"})
        assert not recorder.record_latency({"translation_seconds": .2})
        assert "队列已满" in recorder.snapshot()["error"]
        assert not recorder.active
        unblock.set()
        assert recorder.stop()
    assert json.loads((recorder.path / "transcripts.jsonl").read_text(encoding="utf-8"))["text"] == "already recognized"
    assert (recorder.path / "latency.jsonl").read_text(encoding="utf-8") == ""
    assert recorder.snapshot()["transcript_count"] == 1
    assert recorder.snapshot()["latency_count"] == 0


def test_latency_write_failure_is_contained_and_does_not_accept_more_events(tmp_path):
    original_open = Path.open

    class FailingLatencyFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, payload):
            raise OSError("latency disk full")

    def open_file(path, *args, **kwargs):
        if path.name == "latency.jsonl":
            return FailingLatencyFile()
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", open_file):
        recorder = DiagnosticRecorder(tmp_path, {}).start()
        assert recorder.record_latency({"asr_seconds": .2})
        assert recorder.stop()
    assert recorder.snapshot()["state"] == "error"
    assert "latency disk full" in recorder.snapshot()["error"]
    assert recorder.snapshot()["latency_count"] == 0
    assert not recorder.record_latency({"asr_seconds": .1})
    assert not recorder.submit_audio(np.ones(320, np.float32))


def test_queue_overflow_is_visible_and_does_not_raise_into_audio_callback(tmp_path):
    entered, unblock = Event(), Event()
    original = DiagnosticRecorder._run
    def stalled(recorder):
        entered.set()
        unblock.wait(2)
        original(recorder)
    with patch.object(DiagnosticRecorder, "_run", stalled):
        recorder = DiagnosticRecorder(tmp_path, {}, queue_size=1).start()
        assert entered.wait(1)
        assert recorder.submit_audio(np.ones(320, np.float32))
        assert not recorder.submit_audio(np.ones(320, np.float32))
        assert "队列已满" in recorder.snapshot()["error"]
        assert not recorder.active
        unblock.set()
        assert recorder.stop()
    report = json.loads((recorder.path / "report.json").read_text(encoding="utf-8"))
    assert not report["completed"] and report["truncated"]
    assert report["recorded_seconds"] == .02


def test_disk_failure_and_observer_failure_never_escape_inference(tmp_path):
    updates = []
    def listener(snapshot):
        updates.append(snapshot)
        raise RuntimeError("UI observer failed")
    with patch("simulinterp.diagnostic.wave.open", side_effect=OSError("disk full")):
        recorder = DiagnosticRecorder(tmp_path, {}, listener).start()
        recorder.submit_audio(np.zeros(320, np.float32))
        assert recorder.stop()
    assert recorder.snapshot()["state"] == "error"
    assert "disk full" in recorder.snapshot()["error"]
    assert updates
    assert not recorder.submit_audio(np.zeros(320, np.float32))


def test_nonfinite_audio_stops_only_the_diagnostic(tmp_path):
    recorder = DiagnosticRecorder(tmp_path, {}).start()
    assert not recorder.submit_audio(np.array([np.nan], np.float32))
    assert recorder.stop()
    assert "无效采样值" in recorder.snapshot()["error"]


def test_idle_device_still_stops_at_wall_time_limit(tmp_path):
    now, finished = [0.0], Event()
    def update(snapshot):
        if snapshot["state"] == "stopped":
            finished.set()
    with patch("simulinterp.diagnostic.time.monotonic", side_effect=lambda: now[0]):
        recorder = DiagnosticRecorder(tmp_path, {}, update, max_seconds=.1).start()
        now[0] = .2
        assert finished.wait(2), "an idle loopback must not keep recording armed indefinitely"
        assert recorder.stop()
    report = json.loads((recorder.path / "report.json").read_text(encoding="utf-8"))
    assert report["stop_reason"] == "wall_time_limit"
    assert report["recorded_seconds"] == 0
