"""Lifecycle tests use fake devices/models and never open an audio device."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from simulinterp.engine import SessionWorker
from simulinterp.segmenter import SpeechChunk


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.worker = SessionWorker(dict(model_size="small", device="cpu", device_id=1, language="en"))
        self.errors = []
        self.subtitles = []
        self.lifecycle = []
        self.worker.error.connect(self.errors.append)
        self.worker.subtitle.connect(self.subtitles.append)
        test = self

        class Recognizer:
            instance = None
            instances = []

            def __init__(self, *args, **kwargs):
                Recognizer.instance = self
                Recognizer.instances.append(self)
                self.closed = False
                self.calls = 0

            def load(self):
                test.lifecycle.append("recognizer.load")

            def warmup(self):
                test.lifecycle.append("recognizer.warmup")

            def transcribe(self, samples, language):
                self.calls += 1
                return SimpleNamespace(text=f"phrase {self.calls}", language="en", start=0, end=0.1)

            def close(self):
                self.closed = True

        class Translator:
            instance = None

            def __init__(self, *args, **kwargs):
                Translator.instance = self
                self.cpu_threads = kwargs.get("cpu_threads")
                self.closed = False

            def load(self):
                test.lifecycle.append("translator.load")

            def translate(self, text, language):
                return "中文：" + text

            def close(self):
                self.closed = True

        class Capture:
            instance = None
            instances = []

            def __init__(self, index, on_audio, on_error, *, on_warning=None, on_ready=None):
                Capture.instance = self
                Capture.instances.append(self)
                self.on_audio, self.on_error = on_audio, on_error
                self.on_warning, self.on_ready = on_warning, on_ready
                self.stopped = False
                self.stop_requested = False

            def start(self):
                test.lifecycle.append("capture.start")
                self.on_audio(np.ones(320, dtype=np.float32))
                # Exercise Stop arriving inside capture.start(), before the
                # capture is published to the GUI thread.
                test.worker.request_stop()

            def request_stop(self):
                self.stop_requested = True

            def stop(self):
                if not self.stopped:
                    self.stopped = True
                    self.on_audio(np.ones(160, dtype=np.float32))

        class Segmenter:
            options = None

            def __init__(self, **options):
                Segmenter.options = options
                self.feeds = 0

            def feed(self, samples):
                self.feeds += 1
                return [SpeechChunk(samples, 0.0, 0.1)] if self.feeds == 1 else []

            def flush(self):
                test.assertTrue(Capture.instance.stopped)
                test.assertEqual(self.feeds, 2, "resampler tail was discarded during Stop")
                return [SpeechChunk(np.ones(160), 0.1, 0.2)]

        self.Recognizer, self.Translator = Recognizer, Translator
        self.Capture, self.Segmenter = Capture, Segmenter

    def run_mocked(self):
        with ExitStack() as stack:
            for name, replacement in (
                ("simulinterp.audio.AudioCapture", self.Capture),
                ("simulinterp.recognition.LocalRecognizer", self.Recognizer),
                ("simulinterp.translation.LocalTranslator", self.Translator),
                ("simulinterp.segmenter.SpeechSegmenter", self.Segmenter),
            ):
                stack.enter_context(patch(name, replacement))
            self.worker.run()

    def test_stop_drains_queued_speech_and_flushes_final_phrase(self):
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual([s["source"] for s in self.subtitles], ["phrase 1", "phrase 2"])
        self.assertTrue(self.Capture.instance.stop_requested)
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)
        self.assertTrue(self.worker.is_stopping)

    def test_source_only_skips_translation_model_and_emits_original_to_primary_caption(self):
        self.worker.settings.update(translate_enabled=False, translation_threads=123)
        self.Translator.__init__ = lambda *args, **kwargs: self.fail("Must never construct disabled MT")
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertIsNone(self.Translator.instance)
        self.assertNotIn("translator.load", self.lifecycle)
        self.assertEqual([s["text"] for s in self.subtitles], ["phrase 1", "phrase 2"])
        self.assertTrue(all(s["text"] == s["source"] for s in self.subtitles))
        stats = self.worker.session_metrics()
        self.assertIs(stats["translate_enabled"], False)
        self.assertEqual(stats["translation_seconds"], 0.0)
        self.assertTrue(self.Recognizer.instance.closed)

    def test_source_only_works_when_translation_module_is_unavailable(self):
        self.worker.settings["translate_enabled"] = False
        with patch.dict("sys.modules", {"simulinterp.translation": None}), ExitStack() as stack:
            for name, replacement in (("simulinterp.audio.AudioCapture", self.Capture),
                                      ("simulinterp.recognition.LocalRecognizer", self.Recognizer),
                                      ("simulinterp.segmenter.SpeechSegmenter", self.Segmenter)):
                stack.enter_context(patch(name, replacement))
            self.worker.run()
        self.assertFalse(self.errors)
        self.assertEqual(len(self.subtitles), 2)

    def test_source_only_pause_resume_reuses_asr_and_never_loads_translation(self):
        self.worker.settings["translate_enabled"] = False
        self._install_pause_fakes(on_paused=self.worker.request_resume)
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(self.lifecycle.count("recognizer.load"), 1)
        self.assertNotIn("translator.load", self.lifecycle)
        self.assertEqual(len(self.Capture.instances), 2)
        self.assertEqual([s["text"] for s in self.subtitles], ["phrase 1", "phrase 2"])

    def test_invalid_translation_switch_fails_before_models(self):
        self.worker.settings["translate_enabled"] = "false"
        self.run_mocked()
        self.assertIn("翻译开关", self.errors[0])
        self.assertFalse(self.lifecycle)

    def test_audio_error_aborts_pending_speech(self):
        def fail(capture):
            capture.on_audio(np.ones(320, dtype=np.float32))
            capture.on_error("device disconnected")

        self.Capture.start = fail
        self.run_mocked()
        self.assertEqual(self.errors, ["device disconnected"])
        self.assertFalse(self.subtitles)
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)

    def test_listening_waits_for_ready_and_recovery_warning_is_not_fatal(self):
        ready, statuses = [], []
        self.worker.listening.connect(lambda: ready.append(True))
        self.worker.status.connect(statuses.append)

        def start(capture):
            capture.on_audio(np.ones(320, dtype=np.float32))
            self.assertFalse(ready, "stream initialization must precede Listening")
            capture.on_ready()
            self.assertEqual(ready, [True])
            capture.on_warning("音频采集已自动恢复")
            self.worker.request_stop()

        self.Capture.start = start
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertIn("音频采集已自动恢复", statuses)
        self.assertEqual(len(self.subtitles), 2)

    def test_model_load_failure_still_releases_both_models(self):
        def fail(recognizer):
            raise RuntimeError("CUDA unavailable")

        self.Recognizer.load = fail
        self.run_mocked()
        self.assertIn("CUDA unavailable", self.errors[0])
        self.assertIn("CPU / Small", self.errors[0])
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)
        self.assertIsNone(self.Capture.instance)

    def test_driver_cleanup_failure_does_not_skip_model_cleanup(self):
        def fail(capture):
            raise RuntimeError("driver stuck")

        self.Capture.stop = fail
        self.run_mocked()
        self.assertTrue(any("driver stuck" in error for error in self.errors))
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)

    def test_stop_during_model_load_never_opens_device(self):
        self.Recognizer.load = lambda recognizer: self.worker.request_stop()
        self.run_mocked()
        self.assertIsNone(self.Capture.instance)
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)
        self.assertFalse(self.errors)

    def test_dependency_import_error_reaches_error_signal(self):
        with patch.dict("sys.modules", {"simulinterp.audio": None}):
            self.worker.run()
        self.assertEqual(len(self.errors), 1)
        self.assertIn("simulinterp.audio", self.errors[0])

    def test_backlog_discards_oldest_audio_and_stays_bounded(self):
        for index in range(5):
            self.worker._enqueue(SpeechChunk(np.zeros(320), index, index + 0.02))
        self.assertEqual(self.worker._dropped, 2)
        self.assertEqual([self.worker._queue.get_nowait().chunk.start for _ in range(3)], [2, 3, 4])

    def test_phase_timing_distinguishes_queue_wait_from_asr_and_translation(self):
        clock, timings = [0.0], []
        recognize, translate = self.Recognizer.transcribe, self.Translator.translate

        def transcribe(recognizer, samples, language):
            clock[0] += .25
            return recognize(recognizer, samples, language)

        def translate_slow(translator, text, language):
            clock[0] += .75
            return translate(translator, text, language)

        self.Recognizer.transcribe = transcribe
        self.Translator.translate = translate_slow
        self.worker._record_diagnostic_latency = lambda chunk, timing: timings.append(timing)
        with patch("simulinterp.engine.time.monotonic", side_effect=lambda: clock[0]):
            self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(len(timings), 2)
        self.assertEqual([t["queue_wait_seconds"] for t in timings], [0.0, 1.0])
        self.assertEqual([t["enqueue_to_emit_seconds"] for t in timings], [1.0, 2.0])
        for timing in timings:
            self.assertEqual(timing["asr_seconds"], .25)
            self.assertEqual(timing["translation_seconds"], .75)
            self.assertEqual(timing["processing_seconds"], 1.0)
            self.assertEqual(timing["result"], "subtitle_emitted")
            self.assertIsNone(timing["endpoint_wait_seconds"], "missing media-time endpoint must stay unknown")
        stats = self.worker.session_metrics()
        self.assertEqual(stats["asr_seconds"], .25)
        self.assertEqual(stats["translation_seconds"], .75)
        self.assertEqual(stats["queue_wait_seconds"], 1.0)

    def test_empty_asr_does_not_invoke_translation_or_fake_a_subtitle(self):
        timings = []
        self.Recognizer.transcribe = lambda *args: None
        self.Translator.translate = lambda *args: self.fail("silent input must not invoke MT")
        self.worker._record_diagnostic_latency = lambda chunk, timing: timings.append(timing)
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertFalse(self.subtitles)
        self.assertEqual(len(timings), 2)
        self.assertTrue(all(t["translation_seconds"] == 0 and t["result"] == "no_speech" for t in timings))

    def test_telemetry_reports_backlog_duration_and_known_capture_loss(self):
        for index in range(5):
            self.worker._enqueue(SpeechChunk(np.zeros(320), index, index + .02))
        self.worker._capture = SimpleNamespace(statistics=lambda: {
            "driver_overflows": 2, "queue_dropped_seconds": .3,
        })
        stats = self.worker.session_metrics()
        self.assertEqual(stats["queue_depth"], 3)
        self.assertAlmostEqual(stats["queue_audio_seconds"], .06)
        self.assertEqual(stats["dropped_segments"], 2)
        self.assertEqual(stats["driver_overflows"], 2)
        self.assertEqual(stats["queue_dropped_seconds"], .3)
        self.assertIsNone(stats["realtime_factor"])

    def test_real_time_factor_compares_processing_to_actual_chunk_duration(self):
        self.worker._record_processing(.8, 4)
        stats = self.worker.session_metrics()
        self.assertEqual(stats["processing_seconds"], .8)
        self.assertEqual(stats["audio_seconds"], 4)
        self.assertEqual(stats["realtime_factor"], .2)
        self.assertEqual(stats["processed_segments"], 1)

    def test_cuda_warmup_finishes_before_audio_capture(self):
        self.worker.settings["device"] = "cuda"
        statuses = []
        self.worker.status.connect(statuses.append)
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(self.lifecycle[:4], [
            "recognizer.load", "recognizer.warmup", "translator.load", "capture.start",
        ])
        self.assertIn("正在预热 GPU …", statuses)

    def test_cpu_session_skips_gpu_warmup(self):
        self.run_mocked()
        self.assertNotIn("recognizer.warmup", self.lifecycle)
        self.assertIn("capture.start", self.lifecycle)

    def test_translation_thread_choice_reaches_cpu_model(self):
        self.worker.settings["translation_threads"] = 4
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(self.Translator.instance.cpu_threads, 4)

    def test_default_classroom_profile_reaches_segmenter(self):
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(self.Segmenter.options["max_seconds"], 8.0)
        self.assertEqual(self.Segmenter.options["silence_seconds"], .8)
        self.assertEqual(self.Segmenter.options["soft_min_seconds"], 4.5)
        self.assertEqual(self.Segmenter.options["soft_silence_seconds"], .3)

    def test_original_coherent_profile_retains_fixed_endpoint(self):
        self.worker.settings["asr_profile"] = "coherent"
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(self.Segmenter.options["max_seconds"], 8.0)
        self.assertNotIn("soft_min_seconds", self.Segmenter.options)

    def test_fast_profile_keeps_original_phrase_boundaries(self):
        self.worker.settings["asr_profile"] = "fast"
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(self.Segmenter.options["max_seconds"], 5.0)
        self.assertEqual(self.Segmenter.options["silence_seconds"], .5)

    def test_invalid_profile_fails_before_model_load(self):
        self.worker.settings["asr_profile"] = "unknown"
        self.run_mocked()
        self.assertIn("字幕节奏", self.errors[0])
        self.assertFalse(self.lifecycle)

    def test_auto_language_is_rejected_before_loading_models(self):
        self.worker.settings["language"] = "auto"
        self.run_mocked()
        self.assertIn("选择课程语言", self.errors[0])
        self.assertFalse(self.lifecycle)

    def test_invalid_threads_are_rejected_before_loading_models(self):
        self.worker.settings["translation_threads"] = 100
        self.run_mocked()
        self.assertIn("CPU 线程数", self.errors[0])
        self.assertFalse(self.lifecycle)

    def test_stop_during_warmup_never_starts_audio_capture(self):
        self.worker.settings["device"] = "cuda"
        self.Recognizer.warmup = lambda recognizer: self.worker.request_stop()
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertIsNone(self.Capture.instance)
        self.assertNotIn("translator.load", self.lifecycle)
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)

    def _install_pause_fakes(self, on_paused):
        events = []

        class PauseCapture:
            instances = []

            def __init__(capture, index, on_audio, on_error, *, on_warning=None, on_ready=None):
                PauseCapture.instances.append(capture)
                events.append("capture.open")
                capture.on_audio, capture.on_error = on_audio, on_error
                capture.on_ready = on_ready
                capture.stopped = False
                capture.stop_requested = False

            def start(capture):
                capture.on_audio(np.ones(320, dtype=np.float32))
                capture.on_ready()
                if events.count("capture.open") == 1:
                    test.worker.request_pause()
                else:
                    test.worker.request_stop()

            def request_stop(capture):
                capture.stop_requested = True

            def stop(capture):
                events.append("capture.stop")
                capture.stopped = True

        class PauseSegmenter:
            def __init__(self, **options):
                self.cursor = 0.0

            def feed(self, samples):
                self.cursor += .1
                return [SpeechChunk(samples, self.cursor - .1, self.cursor)]

            def flush(self):
                return []

        test = self
        self.Capture = PauseCapture
        self.Segmenter = PauseSegmenter
        self.worker.paused.connect(on_paused)
        return events

    def test_pause_drains_tail_then_resume_reuses_loaded_models(self):
        resumed = []
        self.worker.listening.connect(lambda: resumed.append(True))
        self._install_pause_fakes(on_paused=self.worker.request_resume)
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(len(self.Capture.instances), 2, "Resume must restart capture")
        self.assertEqual(self.lifecycle.count("recognizer.load"), 1,
                         "Resume must not reload either model")
        self.assertEqual(self.lifecycle.count("translator.load"), 1)
        self.assertEqual([s["source"] for s in self.subtitles], ["phrase 1", "phrase 2"],
                         "Speech before pause and after resume both become subtitles")
        self.assertEqual(resumed, [True, True])
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)

    def test_stop_while_paused_ends_the_session_without_resume(self):
        statuses = []
        self.worker.status.connect(statuses.append)
        self._install_pause_fakes(on_paused=self.worker.request_stop)
        self.run_mocked()
        self.assertFalse(self.errors)
        self.assertEqual(len(self.Capture.instances), 1, "No capture restart after Stop")
        self.assertIn("已暂停", "".join(statuses))
        self.assertEqual(self.lifecycle.count("recognizer.load"), 1)
        self.assertTrue(self.Recognizer.instance.closed)
        self.assertTrue(self.Translator.instance.closed)


if __name__ == "__main__":
    unittest.main()
