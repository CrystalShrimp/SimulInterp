"""Desktop interaction checks; models and recording are never started here."""

import os
from datetime import datetime
from pathlib import Path
import unittest
from unittest.mock import patch

# Use Qt's headless platform before its first application is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from simulinterp.ui import MainWindow, SessionCleanupDialog
from simulinterp.config import DEFAULT_TRANSLATION_THREADS


DEVICES = [
    {"id": 3, "name": "Speakers [Loopback]", "is_loopback": True},
    {"id": 4, "name": "Microphone", "is_loopback": False},
]


class DesktopInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)
        # The Windows offscreen Qt plugin does not enumerate system fonts.
        # Load real fonts so wrapping tests exercise actual Chinese glyph sizes.
        if os.name == "nt" and not QFontDatabase.families():
            for name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf"):
                path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name
                if path.is_file():
                    QFontDatabase.addApplicationFont(str(path))

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def ready(self):
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData("de"))
        self.window.set_devices(DEVICES, default_id=3)
        self.window.set_model_status("模型已就绪", True)

    def test_capture_requires_ready_models_and_an_audio_device(self):
        started = QSignalSpy(self.window.start_requested)
        self.window.start_button.click()
        self.assertEqual(started.count(), 0)
        self.window.set_model_status("模型已就绪", True)
        self.assertFalse(self.window.start_button.isEnabled())
        self.window.set_devices(DEVICES, default_id=3)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertIn("指定", self.window.start_button.toolTip())
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData("de"))
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertEqual(started.count(), 0)  # Readiness never starts recording.
        self.window.set_model_status("模型未下载", False)
        self.assertFalse(self.window.start_button.isEnabled())

    def test_start_and_prepare_emit_the_selected_configuration(self):
        self.ready()
        started = QSignalSpy(self.window.start_requested)
        prepared = QSignalSpy(self.window.prepare_requested)
        changed = QSignalSpy(self.window.model_changed)
        self.window.device_combo.setCurrentIndex(1)
        self.window.language_combo.setCurrentIndex(1)
        self.window.model_combo.setCurrentIndex(1)
        self.window.compute_combo.setCurrentIndex(1)
        self.window.asr_profile_combo.setCurrentIndex(self.window.asr_profile_combo.findData("fast"))
        self.window.start_button.click()
        self.window.prepare_button.click()
        expected = {"device_id": 4, "language": "de", "model_size": "small", "device": "cpu",
                    "asr_profile": "fast", "translate_enabled": True,
                    "translation_threads": DEFAULT_TRANSLATION_THREADS}
        self.assertEqual(started.at(0)[0], expected)
        self.assertEqual(prepared.at(0)[0], expected)
        self.assertEqual(changed.at(0)[0], "small")
        self.assertEqual(self.window.selected_model(), "small")

    def test_busy_and_running_keep_stop_available_and_lock_configuration(self):
        self.ready()
        stopped = QSignalSpy(self.window.stop_requested)
        self.window.set_busy(True, "正在加载模型")
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.model_combo.isEnabled())
        self.assertFalse(self.window.translation_threads_combo.isEnabled())
        self.assertFalse(self.window.asr_profile_combo.isEnabled())
        self.assertFalse(self.window.translate_check.isEnabled())
        self.assertFalse(self.window.prepare_button.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.window.stop_button.click()
        self.assertEqual(stopped.count(), 1)
        self.window.set_busy(False)
        self.window.set_running(True)
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertFalse(self.window.device_combo.isEnabled())
        self.window.set_busy(True, "正在停止，等待当前推理结束")
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertEqual(self.window.state_badge.text(), "停止中…")
        self.window.set_busy(False)
        self.window.set_running(False)
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertTrue(self.window.device_combo.isEnabled())
        self.assertTrue(self.window.asr_profile_combo.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

    def test_pause_button_tracks_paused_state_and_keeps_end_available(self):
        self.ready()
        resumed, paused = QSignalSpy(self.window.resume_requested), QSignalSpy(self.window.pause_requested)
        self.assertFalse(self.window.pause_button.isEnabled())
        self.window.set_running(True)
        self.assertTrue(self.window.pause_button.isEnabled())
        self.assertEqual(self.window.pause_button.text(), "暂停")
        self.window.set_pausing()
        self.assertFalse(self.window.pause_button.isEnabled())
        self.window.set_paused(True)
        self.assertEqual(self.window.pause_button.text(), "继续")
        self.assertEqual(self.window.state_badge.text(), "已暂停")
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertFalse(self.window.language_combo.isEnabled())
        self.assertFalse(self.window.asr_profile_combo.isEnabled())
        self.assertFalse(self.window.translate_check.isEnabled())
        self.assertIn("占用内存与显存", self.window.course_hint.text())
        self.window.pause_button.click()
        self.assertEqual(resumed.count(), 1)
        self.assertEqual(paused.count(), 0)
        self.window.set_paused(False)
        self.window.pause_button.click()
        self.assertEqual(paused.count(), 1)
        self.window.set_running(False)
        self.assertFalse(self.window.pause_button.isEnabled())

    def test_diagnostic_capture_requires_manual_start_during_active_listening(self):
        requested = QSignalSpy(self.window.diagnostic_requested)
        stopped = QSignalSpy(self.window.diagnostic_stop_requested)
        self.ready()
        self.assertFalse(self.window.diagnostic_button.isEnabled())
        self.assertFalse(self.window.diagnostic_stop_button.isEnabled())
        self.window.set_running(True)
        self.assertTrue(self.window.diagnostic_button.isEnabled())
        self.assertEqual(requested.count(), 0, "starting a class must not start recording")
        self.window.diagnostic_button.click()
        self.assertEqual(requested.count(), 1)
        self.window.set_diagnostic_state({"state": "recording", "recorded_seconds": 73,
                                          "path": "D:/diagnostics/class-sample"})
        self.assertFalse(self.window.diagnostic_button.isEnabled())
        self.assertTrue(self.window.diagnostic_stop_button.isEnabled())
        self.assertIn("01:13 / 05:00", self.window.diagnostic_status_label.text())
        self.assertFalse(self.window.diagnostic_badge.isHidden())
        self.window.set_busy(True, "正在暂停")
        self.assertTrue(self.window.diagnostic_stop_button.isEnabled())
        self.window.diagnostic_stop_button.click()
        self.assertEqual(stopped.count(), 1)
        self.window.set_diagnostic_state({"state": "saving"})
        self.assertFalse(self.window.diagnostic_stop_button.isEnabled())
        self.window.set_busy(False)
        self.window.set_paused(True)
        self.window.set_diagnostic_state({"state": "completed", "recorded_seconds": 75})
        self.assertFalse(self.window.diagnostic_button.isEnabled())
        self.window.set_paused(False)
        self.assertTrue(self.window.diagnostic_button.isEnabled())

    def test_diagnostic_completion_and_errors_preserve_output_path_and_course_controls(self):
        self.ready()
        self.window.set_running(True)
        self.window.set_diagnostic_state({"state": "completed", "recorded_seconds": 300,
                                          "path": "D:/diagnostics/class-sample"})
        self.assertIn("05:00", self.window.diagnostic_status_label.text())
        self.assertIn("D:/diagnostics/class-sample", self.window.diagnostic_status_label.text())
        self.assertTrue(self.window.diagnostic_status_label.textInteractionFlags()
                        & Qt.TextInteractionFlag.TextSelectableByMouse)
        self.window.set_diagnostic_state({"state": "error", "error": "disk full"})
        self.assertIn("disk full", self.window.diagnostic_status_label.text())
        self.assertIn("同传仍可继续", self.window.diagnostic_status_label.text())
        self.assertTrue(self.window.pause_button.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertTrue(self.window.diagnostic_button.isEnabled())
        self.window.resize(900, 680)
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.width(), 900)
        self.assertGreaterEqual(self.window.diagnostic_badge.width(), 180)
        self.assertTrue(self.window.rect().contains(
            self.window.diagnostic_badge.mapTo(self.window, self.window.diagnostic_badge.rect().bottomRight())))

    def test_refresh_preserves_selection_or_uses_default_if_removed(self):
        self.ready()
        self.window.device_combo.setCurrentIndex(1)
        self.window.set_devices(list(reversed(DEVICES)), default_id=3)
        self.assertEqual(self.window.configuration()["device_id"], 4)
        self.window.set_devices([DEVICES[0]], default_id=3)
        self.assertEqual(self.window.configuration()["device_id"], 3)
        self.window.set_devices([])
        self.assertFalse(self.window.start_button.isEnabled())

    def test_subtitles_overlay_source_toggle_and_clear_stay_in_sync(self):
        cleared = QSignalSpy(self.window.clear_requested)
        record = {"start": 1.2, "end": 3.6, "source": "Guten Morgen.", "text": "早上好。", "language": "de"}
        self.window.add_subtitle(record)
        self.assertEqual(self.window.subtitle_label.text(), "早上好。")
        self.assertEqual(self.window.overlay.chinese.text(), "早上好。")
        self.window.source_check.setChecked(False)
        self.assertTrue(self.window.source_label.isHidden())
        self.assertTrue(self.window.overlay.original.isHidden())
        self.assertNotIn("Guten Morgen", self.window.history_browser.toPlainText())
        self.window.source_check.setChecked(True)
        self.assertIn("Guten Morgen", self.window.history_browser.toPlainText())
        self.window.overlay_button.click()
        self.app.processEvents()
        self.assertTrue(self.window.overlay.isVisible())
        self.window.overlay.close()
        self.assertFalse(self.window.overlay_button.isChecked())
        self.window.clear_button.click()
        self.assertEqual(cleared.count(), 1)
        self.assertFalse(self.window.export_button.isEnabled())
        self.assertNotIn("Guten Morgen", self.window.history_browser.toPlainText())
        self.assertEqual(self.window.overlay.original.text(), "")

    def test_source_only_choice_restores_without_emitting_and_locks_during_course(self):
        self.assertTrue(self.window.configuration()["translate_enabled"])
        changed, models = QSignalSpy(self.window.preferences_changed), QSignalSpy(self.window.model_changed)
        self.window.apply_preferences({"translate_enabled": False})
        self.assertFalse(self.window.configuration()["translate_enabled"])
        self.assertEqual(changed.count(), 0)
        self.assertEqual(models.count(), 0)
        self.assertFalse(self.window.translation_threads_combo.isEnabled())
        self.assertFalse(self.window.source_check.isEnabled())
        self.assertIn("原文字幕", self.window.output_hint.text())
        self.window.translate_check.setChecked(True)
        self.assertEqual(changed.count(), 1)
        self.assertEqual(changed.at(0)[0]["translate_enabled"], True)
        self.assertEqual(models.count(), 1, "Translation choice changes which local models are required")
        self.assertTrue(self.window.translation_threads_combo.isEnabled())
        self.window.set_running(True)
        self.assertFalse(self.window.translate_check.isEnabled())
        self.window.set_paused(True)
        self.assertFalse(self.window.translate_check.isEnabled())
        self.window.set_running(False)
        self.assertTrue(self.window.translate_check.isEnabled())

    def test_source_only_primary_caption_never_disappears_or_duplicates_in_overlay_and_history(self):
        self.window.apply_preferences({"translate_enabled": False})
        self.window.source_check.setChecked(False)
        text = "A source-only English sentence."
        self.window.add_subtitle(dict(start=1, end=3, source=text, text=text, language="en"))
        self.assertEqual(self.window.subtitle_label.text(), text)
        self.assertEqual(self.window.overlay.chinese.text(), text)
        self.assertEqual(self.window.overlay.original.text(), "")
        self.assertEqual(self.window.source_label.text(), "")
        self.assertEqual(self.window.history_browser.toPlainText().count(text), 1)
        self.window.source_check.setChecked(True)
        self.assertEqual(self.window.subtitle_label.text(), text)
        self.assertEqual(self.window.history_browser.toPlainText().count(text), 1)

    def test_source_only_performance_labels_actual_completed_session_mode(self):
        self.window.begin_session()
        self.window.apply_preferences({"translate_enabled": False})
        self.assertEqual(self.window.performance_labels["inference"].text(), "等待语音")
        self.window.set_session_metrics({"translate_enabled": False, "asr_seconds": .25,
                                         "translation_seconds": 0.0})
        self.assertEqual(self.window.performance_labels["inference"].text(), "0.25 秒 / 关闭")
        self.window.apply_preferences({"translate_enabled": True})
        self.assertEqual(self.window.performance_labels["inference"].text(), "0.25 秒 / 关闭")
        self.window.begin_session()
        self.assertEqual(self.window.performance_labels["inference"].text(), "等待语音")

    def test_history_is_bounded_and_export_returns_a_file_without_writing(self):
        for index in range(205):
            self.window.add_subtitle({"start": index, "end": index + 1, "source": f"source {index}",
                                      "text": f"字幕 {index}", "language": "en"})
        self.assertEqual(len(self.window._history), 200)
        self.assertEqual(self.window._history[0]["start"], 5)
        self.assertIn("字幕 204", self.window.history_browser.toPlainText())
        exported = QSignalSpy(self.window.export_requested)
        with patch.object(QFileDialog, "getSaveFileName", return_value=("D:/captions", "SRT 字幕 (*.srt)")):
            self.window.export_button.click()
        self.assertEqual(exported.at(0)[0], "D:/captions.srt")
        with patch.object(QFileDialog, "getSaveFileName", return_value=("", "")):
            self.window.export_button.click()
        self.assertEqual(exported.count(), 1)

    def test_large_subtitles_wrap_inside_a_narrow_main_window(self):
        self.window.resize(900, 820)
        self.window.show()
        self.window.font_slider.setValue(46)
        self.window.add_subtitle({
            "start": 0, "end": 6, "language": "en",
            "text": "您可以在听到此演示文稿时阅读中国字幕。今天我们讨论语言学习和技术。",
            "source": "You can read Chinese subtitles while listening to this presentation. "
                      "Today we are talking about language learning and technology.",
        })
        self.app.processEvents()
        self.assertEqual(self.window.width(), 900)
        for label in (self.window.subtitle_label, self.window.source_label):
            with self.subTest(label=label.text()):
                bounds = label.fontMetrics().boundingRect(
                    QRect(0, 0, label.contentsRect().width(), 9999),
                    Qt.TextFlag.TextWordWrap, label.text(),
                )
                self.assertLessEqual(bounds.height(), label.contentsRect().height())
                self.assertLessEqual(label.geometry().right(), label.parentWidget().width())

    def test_overlay_preserves_its_width_when_captions_and_font_change(self):
        self.window.overlay.show()
        self.app.processEvents()
        overlay = self.window.overlay
        width = overlay.width()
        overlay.set_font_size(46)
        overlay.set_subtitle(
            "您可以在听到此演示文稿时阅读中国字幕。今天我们讨论语言学习和技术。",
            "You can read Chinese subtitles while listening to this presentation. "
            "Today we are talking about language learning and technology.",
        )
        self.app.processEvents()
        self.assertEqual(overlay.width(), width)
        for label in (overlay.chinese, overlay.original):
            bounds = label.fontMetrics().boundingRect(
                QRect(0, 0, label.contentsRect().width(), 9999),
                Qt.TextFlag.TextWordWrap, label.text(),
            )
            self.assertLessEqual(bounds.height(), label.contentsRect().height())
        screen = overlay.screen().availableGeometry()
        self.assertTrue(screen.contains(overlay.geometry()))

    def test_settings_live_in_a_separate_dialog_and_remain_viewable_while_running(self):
        self.ready()
        self.window.show()
        self.window.settings_button.click()
        self.app.processEvents()
        self.assertTrue(self.window.settings_dialog.isVisible())
        for field in (self.window.language_combo, self.window.model_combo,
                      self.window.compute_combo, self.window.translation_threads_combo,
                      self.window.prepare_button):
            self.assertTrue(self.window.settings_dialog.isAncestorOf(field))
        self.assertFalse(self.window.settings_dialog.isAncestorOf(self.window.device_combo))
        self.window.set_running(True)
        self.assertTrue(self.window.settings_button.isEnabled())
        self.assertFalse(self.window.language_combo.isEnabled())
        self.assertIn("结束并释放", self.window.settings_hint.text())
        self.window.close()
        self.assertFalse(self.window.settings_dialog.isVisible())

    def test_missing_resource_values_are_unavailable_instead_of_false_zero(self):
        self.window.set_performance({"system_cpu_percent": None, "gpu_used_bytes": None,
                                     "gpu_util_percent": float("nan"), "gpu_temperature_c": float("inf")})
        for key in ("cpu", "ram", "process", "vram", "gpu"):
            self.assertEqual(self.window.performance_labels[key].text(), "暂不可用")
        self.assertIn("监测暂不可用", self.window.performance_warning.text())
        self.assertNotIn("未触发", self.window.performance_warning.text())

    def test_resource_pressure_warns_without_changing_model_or_stopping_capture(self):
        self.ready()
        stopped = QSignalSpy(self.window.stop_requested)
        self.window.begin_session()
        self.window.set_performance({"system_cpu_percent": 14, "ram_total_bytes": 16 * 1024**3,
                                     "ram_available_bytes": 700 * 1024**2,
                                     "process_rss_bytes": 2 * 1024**3,
                                     "gpu_total_bytes": 8 * 1024**3,
                                     "gpu_used_bytes": 7.8 * 1024**3, "gpu_util_percent": 80,
                                     "gpu_temperature_c": 70})
        self.assertIn("可用内存偏低", self.window.performance_warning.text())
        self.assertIn("可用显存不足", self.window.performance_warning.text())
        self.assertEqual(self.window.selected_model(), "large-v3-turbo")
        self.assertEqual(stopped.count(), 0)
        self.window.compute_combo.setCurrentIndex(1)
        self.assertNotIn("可用显存不足", self.window.performance_warning.text())
        self.window.set_performance({"ram_total_bytes": 16 * 1024**3,
                                     "ram_available_bytes": 8 * 1024**3})
        self.assertNotIn("可用内存偏低", self.window.performance_warning.text())

    def test_session_slow_warning_is_sustained_and_elapsed_passes_ninety_minutes(self):
        self.window.begin_session()
        sample = {"elapsed_seconds": 5401, "queue_depth": 2, "queue_audio_seconds": 7.5,
                  "realtime_factor": 1.3, "subtitle_count": 1080}
        self.window.set_session_metrics(sample)
        self.window.set_session_metrics(sample)
        self.assertNotIn("持续偏慢", self.window.performance_warning.text())
        self.window.set_session_metrics(sample)
        self.assertIn("持续偏慢", self.window.performance_warning.text())
        self.assertEqual(self.window.performance_labels["elapsed"].text(), "01:30:01")
        self.assertIn("1080", self.window.performance_labels["elapsed"].toolTip())
        self.window.set_session_metrics({"queue_depth": 0, "queue_audio_seconds": 0, "realtime_factor": .4})
        self.assertNotIn("持续偏慢", self.window.performance_warning.text())
        self.window.set_session_metrics({"driver_overflows": 1})
        self.assertIn("丢帧或跳段", self.window.performance_warning.text())
        self.window.end_session()
        self.assertNotIn("丢帧或跳段", self.window.performance_warning.text())

    def test_archive_failure_is_visible_and_a_new_session_clears_metrics(self):
        self.window.begin_session()
        self.window.set_session_metrics({"elapsed_seconds": 5400, "driver_overflows": 2})
        self.window.set_archive_status("D:/captions/session.srt", "disk full")
        self.assertEqual(self.window.archive_status_label.text(), "保存异常")
        self.assertIn("手动导出", self.window.performance_warning.text())
        self.window.set_archive_status("D:/captions/session.srt")
        self.window.end_session()
        self.assertEqual(self.window.archive_status_label.text(), "已保存")
        self.window.begin_session()
        self.assertEqual(self.window.performance_labels["elapsed"].text(), "00:00")
        self.assertEqual(self.window.performance_labels["drops"].text(), "0 次 / 0 段")

    def test_phase_times_are_visible_and_reset_on_new_course(self):
        self.window.begin_session()
        self.assertEqual(self.window.performance_labels["inference"].text(), "等待语音")
        self.window.set_session_metrics({"audio_seconds": 8.0, "asr_seconds": .25,
                                         "translation_seconds": .75, "queue_wait_seconds": .1,
                                         "endpoint_wait_seconds": .68, "realtime_factor": .125})
        self.assertEqual(self.window.performance_labels["inference"].text(), "0.25 / 0.75 秒")
        self.assertIn("8.00 秒", self.window.performance_labels["queue"].toolTip())
        self.assertIn("排队 0.10 秒", self.window.performance_labels["queue"].toolTip())
        self.assertIn("0.68 秒", self.window.performance_labels["queue"].toolTip())
        self.window.begin_session()
        self.assertEqual(self.window.performance_labels["inference"].text(), "等待语音")
        self.assertNotIn("排队 0.10 秒", self.window.performance_labels["queue"].toolTip())

    def test_compact_main_window_keeps_controls_inside_the_window(self):
        self.window.resize(900, 680)
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.size().width(), 900)
        self.assertEqual(self.window.size().height(), 680)
        for control in (self.window.start_button, self.window.stop_button, self.window.settings_button):
            position = control.mapTo(self.window, control.rect().bottomRight())
            self.assertTrue(self.window.rect().contains(position))

    def test_explicit_language_and_threads_preferences_restore_without_autodetection(self):
        languages = [self.window.language_combo.itemData(index)
                     for index in range(self.window.language_combo.count())]
        self.assertEqual(languages, [None, "de", "en"])
        self.assertIsNone(self.window.configuration()["language"])
        changed = QSignalSpy(self.window.preferences_changed)
        self.window.apply_preferences(dict(language="en", translation_threads=4,
                                           model_size="large-v3-turbo", device="cuda"))
        self.assertEqual(changed.count(), 0, "Restoring settings must not rewrite a file")
        self.assertEqual(self.window.configuration()["language"], "en")
        self.assertEqual(self.window.configuration()["translation_threads"], 4)
        self.window.translation_threads_combo.setCurrentIndex(
            self.window.translation_threads_combo.findData(8))
        self.assertEqual(changed.count(), 1)
        self.assertEqual(changed.at(0)[0]["translation_threads"], 8)

    def test_caption_timing_defaults_restores_and_emits_configuration(self):
        self.assertEqual(self.window.configuration()["asr_profile"], "adaptive")
        self.assertEqual(self.window.asr_profile_combo.currentText(), "清晰收音（实时优先）")
        self.assertEqual([self.window.asr_profile_combo.itemData(i)
                          for i in range(self.window.asr_profile_combo.count())],
                         ["adaptive", "coherent", "fast"])
        changed = QSignalSpy(self.window.preferences_changed)
        self.window.apply_preferences({"asr_profile": "fast"})
        self.assertEqual(self.window.configuration()["asr_profile"], "fast")
        self.assertEqual(changed.count(), 0)
        self.window.apply_preferences({"asr_profile": "coherent"})
        self.assertEqual(self.window.configuration()["asr_profile"], "coherent")
        self.assertEqual(changed.count(), 0, "An explicit existing mode must remain selected")
        self.window.apply_preferences({"asr_profile": "invalid"})
        self.assertEqual(self.window.configuration()["asr_profile"], "adaptive")
        for count, profile in enumerate(("coherent", "fast", "adaptive"), start=1):
            self.window.asr_profile_combo.setCurrentIndex(self.window.asr_profile_combo.findData(profile))
            self.assertEqual(changed.count(), count)
            self.assertEqual(changed.at(count - 1)[0]["asr_profile"], profile)
        self.window.apply_preferences({})
        self.assertEqual(self.window.configuration()["asr_profile"], "adaptive")
        self.assertEqual(changed.count(), 3, "Restoring settings must not rewrite a file")

    def test_cleanup_dialog_selects_confirms_and_reports_what_will_go(self):
        dialog = SessionCleanupDialog(self.window)
        sessions = [
            {"path": "D:/exports/sessions/20261028-093000-bbbb2222.srt",
             "name": "20261028-093000-bbbb2222", "size_bytes": 2048,
             "modified": datetime(2026, 10, 28, 9, 30)},
            {"path": "D:/exports/sessions/20261027-080000-aaaa1111.srt",
             "name": "20261027-080000-aaaa1111", "size_bytes": 5 * 1024 ** 2,
             "modified": datetime(2026, 10, 27, 8, 0)},
        ]
        dialog.set_sessions(sessions)
        self.assertFalse(dialog.delete_button.isEnabled())
        self.assertIn("2 KB", dialog.list_widget.item(0).text())
        self.assertIn("2026-10-28 09:30", dialog.list_widget.item(0).text())
        self.assertIn("5.0 MB", dialog.list_widget.item(1).text())
        dialog.list_widget.item(0).setCheckState(Qt.CheckState.Checked)
        self.assertEqual(dialog.delete_button.text(), "删除所选（1）")
        dialog.select_all.setChecked(True)
        self.assertEqual(dialog.delete_button.text(), "删除所选（2）")
        dialog.select_all.setChecked(False)
        dialog.list_widget.item(1).setCheckState(Qt.CheckState.Checked)
        deletions = QSignalSpy(dialog.deletion_requested)
        with patch.object(QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.No):
            dialog.delete_button.click()
        self.assertEqual(deletions.count(), 0, "cancellation must delete nothing")
        with patch.object(QMessageBox, "question",
                          return_value=QMessageBox.StandardButton.Yes):
            dialog.delete_button.click()
        self.assertEqual(deletions.count(), 1)
        self.assertEqual(deletions.at(0)[0], ["D:/exports/sessions/20261027-080000-aaaa1111.srt"])
        dialog.set_sessions([])
        self.assertEqual(dialog.list_widget.count(), 1)
        self.assertFalse(dialog.delete_button.isEnabled())
        self.assertIn("暂无", dialog.list_widget.item(0).text())

    def test_cleanup_entry_lives_in_settings_and_locks_while_class_runs(self):
        self.ready()
        self.assertTrue(self.window.settings_dialog.isAncestorOf(self.window.cleanup_button))
        opened = QSignalSpy(self.window.cleanup_requested)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window.cleanup_button.click()
        self.assertEqual(opened.count(), 1)
        self.window.set_running(True)
        self.assertFalse(self.window.cleanup_button.isEnabled())
        self.window.set_running(False)
        self.assertTrue(self.window.cleanup_button.isEnabled())

    def test_open_cleanup_relays_deletions_and_refreshes_the_visible_list(self):
        sessions = [{"path": "D:/a.srt", "name": "20261028-090000-a", "size_bytes": 10,
                     "modified": datetime(2026, 10, 28, 9)}]
        relayed = QSignalSpy(self.window.delete_sessions_requested)
        self.window.open_cleanup(sessions)
        self.assertTrue(self.window.cleanup_dialog.isVisible())
        self.window.cleanup_dialog.list_widget.item(0).setCheckState(Qt.CheckState.Checked)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window.cleanup_dialog.delete_button.click()
        self.assertEqual(relayed.count(), 1)
        self.assertEqual(relayed.at(0)[0], ["D:/a.srt"])
        self.window.refresh_cleanup([])
        self.assertIn("暂无", self.window.cleanup_dialog.list_widget.item(0).text())
        self.window.close()


if __name__ == "__main__":
    unittest.main()
