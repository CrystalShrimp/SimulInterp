"""Explicit, bounded diagnostic recording of the existing 16 kHz mono stream.

This module never opens an audio device. Nothing is recorded before start().
Audio callbacks enqueue short copies; all file operations belong to the writer.
"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
import time
from typing import Callable
from uuid import uuid4
import wave

import numpy as np


logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000


class DiagnosticRecorder:
    """Opt-in writer with at most five minutes of audio and a bounded backlog.

    ``on_update`` can run on the caller or writer thread. Qt users should bridge
    it through a Signal, not mutate widgets directly. ``stop(wait=False)`` only
    requests finalization; ``stop(wait=True)`` joins it without raising into the
    inference pipeline. An audio/event queue overflow makes this diagnostic
    explicitly invalid, rather than silently losing recording data.
    """

    def __init__(self, directory: Path, metadata: dict, on_update: Callable | None = None,
                 *, max_seconds: float = 300, queue_size: int = 100):
        if not 0 < max_seconds <= 300:
            raise ValueError("诊断采样时长必须大于 0 且不超过 300 秒")
        if queue_size < 1:
            raise ValueError("queue_size 必须大于 0")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(directory) / f"{stamp}-{uuid4().hex[:8]}"
        self.metadata = dict(metadata)
        self.on_update = on_update
        self.max_frames = round(max_seconds * SAMPLE_RATE)
        self._queue = Queue(maxsize=queue_size)
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._state = "idle"
        self._error = ""
        self._accepted_frames = 0
        self._written_frames = 0
        self._transcript_count = 0
        self._latency_count = 0
        self._started = False
        self._started_at = None
        self._wall_limit = False
        self._at_limit = False
        self._last_update = float("-inf")

    @property
    def active(self) -> bool:
        with self._lock:
            return self._started and not self._stop.is_set() and self._state == "recording"

    def snapshot(self) -> dict:
        with self._lock:
            return dict(state=self._state, path=str(self.path), error=self._error,
                        recorded_seconds=self._written_frames / SAMPLE_RATE,
                        accepted_seconds=self._accepted_frames / SAMPLE_RATE,
                        max_seconds=self.max_frames / SAMPLE_RATE,
                        transcript_count=self._transcript_count,
                        latency_count=self._latency_count)

    def _notify(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_update < 1:
            return
        self._last_update = now
        if self.on_update is not None:
            try:
                self.on_update(self.snapshot())
            except Exception:
                logger.exception("Diagnostic update observer failed")

    def start(self):
        with self._lock:
            if self._started or self._stop.is_set():
                return self
            self._started = True
            self._started_at = time.monotonic()
            self._state = "recording"
            self._thread = Thread(target=self._run, name="diagnostic-writer", daemon=False)
            self._thread.start()
        self._notify(force=True)
        return self

    def _fail(self, message: str):
        with self._lock:
            if not self._error:
                self._error = message
            self._state = "error"
            self._stop.set()

    def submit_audio(self, samples: np.ndarray) -> bool:
        """Accept only new mono samples; do no disk I/O and never block on queue."""
        try:
            with self._lock:
                if not self._started or self._stop.is_set():
                    return False
                remaining = self.max_frames - self._accepted_frames
                audio = np.asarray(samples)
                if audio.ndim != 1:
                    raise ValueError("诊断音频必须为单声道")
                audio = audio[:remaining]
                if not audio.size:
                    return False
                if not np.all(np.isfinite(audio)):
                    raise ValueError("诊断音频含无效采样值")
                payload = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                try:
                    self._queue.put_nowait(("audio", payload))
                except Full:
                    self._error = "诊断录音写入队列已满，采样已停止；同传继续运行，请重新采样"
                    self._state = "error"
                    self._stop.set()
                    return False
                self._accepted_frames += audio.size
                if self._accepted_frames >= self.max_frames:
                    self._at_limit = True
                    self._state = "saving"
                    self._stop.set()
                return True
        except Exception as exc:
            self._fail(f"诊断音频采样失败：{exc}")
            return False

    def record_transcript(self, payload: dict) -> bool:
        """Preserve caller-supplied course/diagnostic timestamps and ASR text."""
        return self._record_event("transcript", payload, "文本")

    def record_latency(self, payload: dict) -> bool:
        """Queue software-stage timings separately from pre-translation ASR text.

        These timings are supplied by the caller. This method neither measures
        microphone hardware latency nor waits for the GUI to paint a subtitle.
        """
        return self._record_event("latency", payload, "耗时")

    def _record_event(self, kind: str, payload: dict, label: str) -> bool:
        try:
            with self._lock:
                if not self._started or self._stop.is_set():
                    return False
                self._queue.put_nowait((kind, dict(payload)))
            return True
        except Full:
            self._fail(f"诊断{label}写入队列已满，采样已停止；同传继续运行，请重新采样")
        except Exception as exc:
            self._fail(f"诊断{label}采样失败：{exc}")
        return False

    def stop(self, *, wait: bool = True, timeout: float = 3.0) -> bool:
        # Serialize the stop boundary with a producer's final queue insertion.
        # Otherwise a writer could observe stop+empty before that insertion.
        with self._lock:
            if self._state == "recording":
                self._state = "saving"
            self._stop.set()
        worker = self._thread
        if wait and worker is not None and worker is not current_thread():
            worker.join(timeout)
        return worker is None or not worker.is_alive()

    def _run(self):
        try:
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path / "metadata.json").write_text(json.dumps(dict(
                self.metadata,
                audio_format={"sample_rate": SAMPLE_RATE, "channels": 1, "sample_width_bytes": 2,
                              "stage": "16 kHz mono after device conversion, before speech VAD; not native device PCM"},
                max_seconds=self.max_frames / SAMPLE_RATE,
            ), ensure_ascii=False, indent=2), encoding="utf-8")
            with wave.open(str(self.path / "audio.wav"), "wb") as recording, \
                    (self.path / "transcripts.jsonl").open("x", encoding="utf-8", newline="\n") as texts, \
                    (self.path / "latency.jsonl").open("x", encoding="utf-8", newline="\n") as timings:
                recording.setnchannels(1)
                recording.setsampwidth(2)
                recording.setframerate(SAMPLE_RATE)
                while not self._stop.is_set() or not self._queue.empty():
                    with self._lock:
                        if (not self._stop.is_set() and self._started_at is not None
                                and time.monotonic() - self._started_at >= self.max_frames / SAMPLE_RATE):
                            self._wall_limit = True
                            self._state = "saving"
                            self._stop.set()
                    try:
                        kind, payload = self._queue.get(timeout=.1)
                    except Empty:
                        continue
                    if kind == "audio":
                        recording.writeframes(payload)
                        with self._lock:
                            self._written_frames += len(payload) // 2
                    elif kind == "transcript":
                        texts.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        texts.flush()
                        with self._lock:
                            self._transcript_count += 1
                    elif kind == "latency":
                        timings.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        timings.flush()
                        with self._lock:
                            self._latency_count += 1
                    self._notify()
            with self._lock:
                if not self._error:
                    self._state = "completed" if self._at_limit else "stopped"
        except Exception as exc:
            self._fail(f"诊断文件保存失败：{exc}")
        finally:
            # Release queued copies even if writing failed before normal drain.
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            try:
                report = dict(self.snapshot(), completed=self._at_limit and not self._error,
                              truncated=not self._at_limit or bool(self._error),
                              stop_reason="error" if self._error else "sample_limit" if self._at_limit
                                          else "wall_time_limit" if self._wall_limit else "manual_or_course_end",
                              timing_note=("Transcript and latency timestamps are supplied by the caller; "
                                           "final in-flight ASR or latency events may arrive after recording has ended and be omitted. "
                                           "Software-stage durations and media-clock endpoint waiting do not measure "
                                           "microphone hardware delay or GUI paint latency; enqueue-to-emit is not full end-to-end latency."))
                (self.path / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                self._fail(f"诊断报告保存失败：{exc}")
            self._notify(force=True)
