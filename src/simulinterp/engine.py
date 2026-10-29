"""Single inference worker with bounded backlog and graceful capture shutdown."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock
import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from .config import (DEFAULT_ASR_PROFILE, DEFAULT_TRANSLATION_THREADS,
                     TRANSLATION_THREAD_OPTIONS, asr_path, segmentation_options,
                     translation_path)
from .subtitles import Subtitle


@dataclass(frozen=True)
class _QueuedSpeech:
    chunk: object
    enqueued_at: float


class SessionWorker(QThread):
    subtitle = Signal(dict)
    status = Signal(str)
    error = Signal(str)
    level = Signal(float)
    metrics = Signal(int, float)
    listening = Signal()
    paused = Signal()
    diagnostic = Signal(dict)

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._stop_event = Event()
        self._abort_event = Event()
        self._pause_event = Event()
        self._paused_event = Event()
        self._wake_event = Event()
        self._control_lock = Lock()
        self._capture = None
        self._queue: Queue = Queue(maxsize=3)
        self._dropped = 0
        self._last_level = 0.0
        self._diagnostic_lock = Lock()
        self._diagnostic_recorder = None
        self._audio_cursor_samples = 0
        self._diagnostic_start_sample = 0
        self._stats_lock = Lock()
        self._stats = {
            "processing_seconds": None, "audio_seconds": None,
            "realtime_factor": None, "processed_segments": 0,
            "driver_overflows": 0, "queue_dropped_seconds": 0.0,
            "endpoint_wait_seconds": None, "queue_wait_seconds": None,
            "asr_seconds": None, "translation_seconds": None,
            "enqueue_to_emit_seconds": None,
            "translate_enabled": None,
        }

    def session_metrics(self) -> dict:
        # Queue's mutex protects this bounded (at most three item) view. No
        # model call or audio array copy runs on the GUI/monitoring thread.
        with self._queue.mutex:
            depth = len(self._queue.queue)
            seconds = sum(item.chunk.duration for item in self._queue.queue)
        with self._stats_lock:
            result = dict(self._stats)
            capture = self._capture
            if capture is not None and hasattr(capture, "statistics"):
                current = capture.statistics()
                for key in ("driver_overflows", "queue_dropped_seconds"):
                    result[key] += current.get(key, 0)
        result.update(queue_depth=depth, queue_audio_seconds=seconds,
                      dropped_segments=self._dropped)
        return result

    def _record_processing(self, seconds: float, audio_seconds: float, *, timing: dict | None = None) -> None:
        with self._stats_lock:
            self._stats.update(
                processing_seconds=seconds, audio_seconds=audio_seconds,
                realtime_factor=seconds / audio_seconds if audio_seconds > 0 else None,
                processed_segments=self._stats["processed_segments"] + 1,
            )
            if timing is not None:
                for key in ("endpoint_wait_seconds", "queue_wait_seconds", "asr_seconds",
                            "translation_seconds", "enqueue_to_emit_seconds", "translate_enabled"):
                    self._stats[key] = timing.get(key)

    def request_diagnostic_start(self, directory: Path, extra_metadata: dict | None = None) -> bool:
        """Manually tap future input only, before either speech filter."""
        from .diagnostic import DiagnosticRecorder

        with self._diagnostic_lock:
            if (self.is_stopping or self._pause_event.is_set() or self._capture is None
                    or self._abort_event.is_set()):
                return False
            previous = self._diagnostic_recorder
            if previous is not None:
                if previous.snapshot()["state"] in ("recording", "saving"):
                    return False
                if not previous.stop(wait=False):
                    return False
            self._diagnostic_start_sample = self._audio_cursor_samples
            metadata = dict(
                extra_metadata or {}, settings=dict(self.settings),
                course_start_seconds=self._diagnostic_start_sample / 16000,
                recognition={"beam_size": 3, "compute_type": "int8_float16" if self.settings["device"] == "cuda" else "int8",
                             "vad_filter": True, "vad_min_silence_ms": 300,
                             "condition_on_previous_text": False, "temperature": 0.0},
                translation={"enabled": self.settings.get("translate_enabled", True)},
                segmentation=dict(segmentation_options(self.settings.get("asr_profile", DEFAULT_ASR_PROFILE)),
                                  profile=self.settings.get("asr_profile", DEFAULT_ASR_PROFILE),
                                  webrtc_mode=2),
                latency_format={"version": 1, "file": "latency.jsonl",
                                "endpoint_wait": "capture-stream endpoint position minus retained chunk end",
                                "processing": ("monotonic inference durations; original and Chinese emitted together after translation"
                                               if self.settings.get("translate_enabled", True) else
                                               "monotonic inference durations; source-only subtitle emitted after ASR; translation disabled"),
                                "limitations": "Excludes driver/capture queue, device conversion and GUI paint latency; final in-flight results may be absent."},
            )
            recorder = DiagnosticRecorder(directory, metadata, self.diagnostic.emit)
            self._diagnostic_recorder = recorder
            recorder.start()
            return True

    def request_diagnostic_stop(self, *, wait: bool = False) -> None:
        with self._diagnostic_lock:
            recorder = self._diagnostic_recorder
        if recorder is not None:
            recorder.stop(wait=wait)

    def _tap_diagnostic_audio(self, samples) -> None:
        with self._diagnostic_lock:
            self._audio_cursor_samples += samples.size
            recorder = self._diagnostic_recorder
            if recorder is not None:
                recorder.submit_audio(samples)

    def _record_diagnostic_transcript(self, chunk, transcript, seconds: float) -> None:
        with self._diagnostic_lock:
            recorder = self._diagnostic_recorder
            origin = self._diagnostic_start_sample / 16000
        if recorder is not None and recorder.active and chunk.end > origin:
            stats = self.session_metrics()
            recorder.record_transcript(dict(
                chunk_start=chunk.start, chunk_end=chunk.end,
                diagnostic_chunk_start=chunk.start - origin,
                diagnostic_chunk_end=chunk.end - origin,
                text=transcript.text if transcript else None,
                source_start=chunk.start + transcript.start if transcript else None,
                source_end=min(chunk.end, chunk.start + transcript.end) if transcript else None,
                language=self.settings["language"], asr_seconds=seconds,
                queue_depth=stats["queue_depth"], dropped_segments=stats["dropped_segments"],
                driver_overflows=stats["driver_overflows"],
                queue_dropped_seconds=stats["queue_dropped_seconds"],
            ))

    def _record_diagnostic_latency(self, chunk, timing: dict) -> None:
        with self._diagnostic_lock:
            recorder = self._diagnostic_recorder
            origin = self._diagnostic_start_sample / 16000
        if recorder is not None and recorder.active and chunk.end > origin:
            recorder.record_latency(dict(
                chunk_start=chunk.start, chunk_end=chunk.end,
                diagnostic_chunk_start=chunk.start - origin,
                diagnostic_chunk_end=chunk.end - origin,
                audio_seconds=chunk.duration, **timing,
            ))

    @property
    def is_stopping(self) -> bool:
        return self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._paused_event.is_set()

    def request_stop(self) -> None:
        """Stop recording immediately; drain finalized speech on the worker."""
        self._stop_event.set()
        self._wake_event.set()
        capture = self._capture
        if capture is not None:
            capture.request_stop()

    def request_pause(self) -> None:
        """Stop accepting audio now; acknowledge after tail/queue are drained."""
        with self._control_lock:
            if self.is_stopping:
                return
            self._pause_event.set()
        capture = self._capture
        if capture is not None:
            capture.request_stop()

    def request_resume(self) -> bool:
        """Resume only an acknowledged pause, without reloading either model."""
        with self._control_lock:
            if self.is_stopping or not self.is_paused:
                return False
            self._pause_event.clear()
            self._paused_event.clear()
            self._wake_event.set()
            return True

    def _wait_while_paused(self) -> None:
        self.request_diagnostic_stop(wait=True)
        # Clear before publishing paused so an immediate Resume cannot lose its
        # wakeup. Control methods serialize the acknowledgement/resume change.
        with self._control_lock:
            if self.is_stopping or not self._pause_event.is_set():
                return
            self._wake_event.clear()
            self._paused_event.set()
        self.level.emit(0.0)
        self.status.emit("已暂停 · 模型保留，可继续本次课程")
        self.paused.emit()
        while self._pause_event.is_set() and not self.is_stopping:
            self._wake_event.wait()
            self._wake_event.clear()
        self._paused_event.clear()

    def _retire_capture(self, capture) -> None:
        """Accumulate a stopped device exactly once across capture restarts."""
        current = capture.statistics() if hasattr(capture, "statistics") else {}
        with self._stats_lock:
            if self._capture is capture:
                for key in ("driver_overflows", "queue_dropped_seconds"):
                    self._stats[key] += current.get(key, 0)
                self._capture = None

    def _enqueue(self, chunk) -> None:
        pending = _QueuedSpeech(chunk, time.monotonic())
        try:
            self._queue.put_nowait(pending)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._dropped += 1
            self._queue.put_nowait(pending)
            self.status.emit(f"处理速度暂时不足，已跳过 {self._dropped} 段积压音频；可改用 Small 模型")

    def _report_error(self, exc: Exception) -> None:
        message = str(exc)
        if any(word in message.lower() for word in ("cuda", "cudnn", "cublas", "out of memory")):
            message += "\n请关闭占用显存的程序，运行 setup.ps1 安装 CUDA 运行库，或切换到 CPU / Small。"
        self.error.emit(message)

    def run(self) -> None:
        capture = recognizer = translator = None
        try:
            # Protect imports as well so missing dependencies reach the UI.
            from .audio import AudioCapture
            from .recognition import LocalRecognizer
            from .segmenter import SpeechSegmenter

            if self.is_stopping:
                return
            if self.settings.get("language") not in ("de", "en"):
                raise ValueError("请先在设置中选择课程语言：德语或英语")
            translate_enabled = self.settings.get("translate_enabled", True)
            if type(translate_enabled) is not bool:
                raise ValueError("翻译开关必须为布尔值")
            threads = self.settings.get("translation_threads", DEFAULT_TRANSLATION_THREADS)
            if translate_enabled and threads not in TRANSLATION_THREAD_OPTIONS:
                raise ValueError("翻译 CPU 线程数必须为 4、6 或 8")
            phrase_options = segmentation_options(self.settings.get("asr_profile", DEFAULT_ASR_PROFILE))
            recognizer = LocalRecognizer(asr_path(self.settings["model_size"]), self.settings["device"])
            if translate_enabled:
                from .translation import LocalTranslator
                translator = LocalTranslator(translation_path(), cpu_threads=threads)
            self.status.emit("正在加载本地语音识别模型 …")
            recognizer.load()
            if self.is_stopping:
                return
            if self.settings["device"] == "cuda":
                self.status.emit("正在预热 GPU …")
                recognizer.warmup()
                if self.is_stopping:
                    return
            if translator is not None:
                self.status.emit("正在加载本地中文翻译模型 …")
                translator.load()
            if self.is_stopping:
                return
            segmenter = SpeechSegmenter(**phrase_options)

            def on_audio(samples):
                # Stop still accepts already captured packets and final tail.
                if self._abort_event.is_set():
                    return
                self._tap_diagnostic_audio(samples)
                now = time.monotonic()
                if now - self._last_level > .1:
                    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                    self.level.emit(min(1.0, rms * 8))
                    self._last_level = now
                for chunk in segmenter.feed(samples):
                    self._enqueue(chunk)

            def on_error(message):
                self._abort_event.set()
                self._stop_event.set()
                self._wake_event.set()
                self.error.emit(message)

            def on_ready():
                if not self.is_stopping and not self._pause_event.is_set() and not self._abort_event.is_set():
                    self.listening.emit()
                    language_name = "德语" if self.settings["language"] == "de" else "英语"
                    self.status.emit(f"正在聆听 · {language_name} → 中文字幕" if translate_enabled else
                                     f"正在聆听 · {language_name}原文字幕（不翻译）")

            while not self.is_stopping and not self._abort_event.is_set():
                if self._pause_event.is_set():
                    self._wait_while_paused()
                    if self.is_stopping:
                        break
                self.status.emit("正在初始化音频设备 …")
                capture = AudioCapture(
                    int(self.settings["device_id"]), on_audio, on_error,
                    on_warning=self.status.emit, on_ready=on_ready,
                )
                # Publish before Start and recheck afterward: Stop/Pause may
                # arrive while a native device is being initialized.
                with self._stats_lock:
                    self._capture = capture
                capture.start()
                if self.is_stopping or self._pause_event.is_set():
                    capture.request_stop()
                capture_stopped = False
                while not self._abort_event.is_set():
                    if (self.is_stopping or self._pause_event.is_set()) and not capture_stopped:
                        self.status.emit("正在结束采集并处理最后几段语音 …" if self.is_stopping
                                         else "正在暂停，处理已采集的最后几段语音 …")
                        capture.stop()
                        capture_stopped = True
                        if self._abort_event.is_set():
                            break
                        # Joining capture protects the shared segmenter. Flush
                        # resets phrase state but preserves its sample cursor.
                        for chunk in segmenter.flush():
                            self._enqueue(chunk)
                    if capture_stopped and self._queue.empty():
                        break
                    try:
                        pending = self._queue.get(timeout=.15)
                    except Empty:
                        continue
                    started = time.monotonic()
                    chunk = pending.chunk
                    queue_wait = max(0.0, started - pending.enqueued_at)
                    transcript = recognizer.transcribe(chunk.samples, self.settings["language"])
                    asr_done = time.monotonic()
                    self._record_diagnostic_transcript(chunk, transcript, asr_done - started)
                    if self._abort_event.is_set():
                        break
                    translation_started = time.monotonic()
                    translated = (translator.translate(transcript.text, transcript.language) if translator is not None else
                                  transcript.text) if transcript else None
                    translation_done = time.monotonic()
                    if self._abort_event.is_set():
                        break
                    if translated:
                        self.subtitle.emit(asdict(Subtitle(
                            chunk.start + transcript.start,
                            min(chunk.end, chunk.start + transcript.end),
                            transcript.text, translated, transcript.language,
                        )))
                    emitted_at = time.monotonic()
                    elapsed = emitted_at - started
                    endpoint = getattr(chunk, "available_at", None)
                    timing = dict(
                        endpoint_wait_seconds=max(0.0, endpoint - chunk.end) if endpoint is not None else None,
                        queue_wait_seconds=queue_wait,
                        asr_seconds=asr_done - started,
                        translation_seconds=translation_done - translation_started if transcript and translator is not None else 0.0,
                        translate_enabled=translate_enabled,
                        processing_seconds=elapsed,
                        enqueue_to_emit_seconds=max(0.0, emitted_at - pending.enqueued_at),
                        result="subtitle_emitted" if translated else "no_translation" if transcript else "no_speech",
                    )
                    self._record_processing(elapsed, chunk.duration, timing=timing)
                    self._record_diagnostic_latency(chunk, timing)
                    self.metrics.emit(self._queue.qsize(), elapsed)
                if not capture_stopped:
                    capture.stop()
                self._retire_capture(capture)
                capture = None
                if self._abort_event.is_set() or self.is_stopping:
                    break
        except Exception as exc:
            self._abort_event.set()
            self._report_error(exc)
        finally:
            self._abort_event.set()
            self.request_diagnostic_stop(wait=True)
            self._paused_event.clear()
            # Device cleanup failure must not prevent either model releasing.
            for resource in (capture, recognizer, translator):
                if resource is None:
                    continue
                try:
                    resource.stop() if resource is capture else resource.close()
                except Exception as exc:
                    self.error.emit(f"释放资源时发生错误：{exc}")
            if capture is not None and hasattr(capture, "statistics"):
                self._retire_capture(capture)
            else:
                with self._stats_lock:
                    self._capture = None
            self.level.emit(0.0)
