from __future__ import annotations

from pathlib import Path
import sys
import time

from .runtime import bootstrap


def _redirect_detached_stdio(log_dir: Path) -> None:
    # pythonw.exe runs without a console and sys.stderr is None there, which
    # silently discards tracebacks. Keep them on disk so startup stays fixable.
    if sys.stderr is not None:
        return
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        sys.stderr = open(log_dir / "app.stderr.log", "a", buffering=1,
                          encoding="utf-8", errors="replace")
    except OSError:
        pass


def main() -> int:
    bootstrap()
    from .config import DATA_ROOT
    _redirect_detached_stdio(DATA_ROOT / "logs")
    from PySide6.QtCore import QObject, QEvent, QProcess, QProcessEnvironment, QTimer
    from PySide6.QtWidgets import QApplication
    from .audio import list_audio_devices
    from .archive import SessionArchive, list_session_archives
    from .config import model_status
    from .engine import SessionWorker
    from .performance import ResourceMonitor
    from .preferences import load_preferences, save_preferences
    from .subtitles import Subtitle, render_srt, render_text
    from .tray import TrayController, application_icon
    from .ui import MainWindow

    class Controller(QObject):
        def __init__(self, window):
            super().__init__(window)
            self.window = window
            self.worker = None
            self.preparation = None
            self.entries: list[Subtitle] = []
            self.closing = False
            self.last_error = ""
            self.cancelled_preparation = False
            self.archive = None
            self._listening_since = None
            self._elapsed_seconds = 0.0
            self._last_session_metrics = {}
            self._resuming = False
            self._quitting = False
            window.apply_preferences(load_preferences())
            window.installEventFilter(self)
            window.start_requested.connect(self.start)
            window.stop_requested.connect(self.stop)
            window.pause_requested.connect(self.pause)
            window.resume_requested.connect(self.resume)
            window.minimize_to_tray_requested.connect(self.hide_to_tray)
            window.refresh_devices_requested.connect(self.refresh_devices)
            window.prepare_requested.connect(self.prepare)
            window.model_changed.connect(self.check_models)
            window.export_requested.connect(self.export)
            window.clear_requested.connect(self.clear)
            window.cleanup_requested.connect(self.show_cleanup)
            window.delete_sessions_requested.connect(self.delete_sessions)
            window.preferences_changed.connect(self.save_settings)
            window.diagnostic_requested.connect(self.start_diagnostic)
            window.diagnostic_stop_requested.connect(self.stop_diagnostic)
            self.check_models(window.selected_model())
            self.tray = TrayController(self)
            self.tray.show_requested.connect(self.show_from_tray)
            self.tray.pause_requested.connect(self.pause)
            self.tray.resume_requested.connect(self.resume)
            self.tray.end_requested.connect(self.stop)
            self.tray.quit_requested.connect(self.quit_app)
            self.tray.overlay_requested.connect(window.overlay_button.setChecked)
            window.overlay_button.toggled.connect(self.tray.set_overlay_visible)
            window.overlay.visibility_changed.connect(self.tray.set_overlay_visible)
            self.tray_available = self.tray.show()
            if not self.tray_available:
                window.tray_button.hide()
            self._set_tray_state("idle")
            self.monitor = ResourceMonitor(self)
            self.monitor.snapshot.connect(window.set_performance)
            self.monitor.finished.connect(self.monitor_finished)
            self.monitor.start()
            self.session_timer = QTimer(self)
            self.session_timer.setInterval(2000)
            self.session_timer.timeout.connect(self.update_session_metrics)
            self.session_timer.start()
            QTimer.singleShot(0, self.refresh_devices)
            if window.language_combo.currentData() not in ("de", "en"):
                self.window.set_status("请先在设置中指定课程语言：德语或英语；下次会记住选择。")
                QTimer.singleShot(0, self.window.settings_button.click)

        def save_settings(self, settings):
            try:
                save_preferences(settings)
            except OSError as exc:
                self.window.set_status(f"设置保存失败，本次选择仍可使用：{exc}")

        def start_diagnostic(self):
            if self.worker is None or not self.window._running or self.window._busy or self.window._paused:
                self.window.set_status("请先开始同传，待正在聆听后再开启诊断采样。")
                return
            started = self.worker.request_diagnostic_start(
                DATA_ROOT / "diagnostics", {"audio_source": self.window.device_combo.currentText()},
            )
            if not started:
                self.window.set_status("暂不能开始诊断采样，请等上次采样完成或恢复同传后重试。")

        def stop_diagnostic(self):
            if self.worker is not None:
                self.worker.request_diagnostic_stop()
                self.window.set_diagnostic_state({"state": "saving"})

        def diagnostic_updated(self, data):
            if data.get("state") == "stopped":
                data = dict(data, state="completed")
            self.window.set_diagnostic_state(data)

        def update_session_metrics(self):
            if self._listening_since is not None:
                now = time.monotonic()
                self._elapsed_seconds += now - self._listening_since
                self._listening_since = now
            if self.worker is not None:
                self._last_session_metrics = self.worker.session_metrics()
            self.window.set_session_metrics(dict(
                self._last_session_metrics, elapsed_seconds=self._elapsed_seconds,
                subtitle_count=len(self.entries),
            ))

        def _freeze_elapsed(self):
            if self._listening_since is not None:
                self._elapsed_seconds += time.monotonic() - self._listening_since
                self._listening_since = None

        def monitor_finished(self):
            if self.closing and not self.worker and not self.preparation:
                self.window.close()
                # The tray icon is not a window, so a hidden-window close does
                # not trigger quitOnLastWindowClosed; quit explicitly.
                QApplication.instance().quit()

        def close_archive(self):
            if self.archive is not None:
                archive, self.archive = self.archive, None
                try:
                    archive.close()
                except OSError as exc:
                    self.window.set_archive_status(str(archive.path), str(exc))

        def refresh_devices(self):
            try:
                devices = list_audio_devices()
                # Classroom use starts on a microphone. Device indices are
                # session-local; the window still preserves an explicit choice
                # when refreshing a list that contains the same device.
                default = next((d.index for d in devices if d.is_default and d.kind == "microphone"), None)
                if default is None:
                    default = next((d.index for d in devices if d.kind == "microphone"), None)
                if default is None:
                    default = next((d.index for d in devices if d.is_default and d.kind == "loopback"), None)
                if default is None and devices:
                    default = devices[0].index
                self.window.set_devices([
                    {"id": d.index, "name": d.name, "is_loopback": d.kind == "loopback"} for d in devices
                ], default)
                if not devices:
                    self.window.set_status("没有可用音频设备，请检查 Windows 声音设置后刷新")
            except Exception as exc:
                self.window.set_status(str(exc))

        def check_models(self, model_size):
            ready, message = (model_status(model_size) if self.window.configuration().get("translate_enabled", True)
                              else model_status(model_size, translate_enabled=False))
            self.window.set_model_status(message, ready)

        def start(self, settings):
            if self.worker is not None or self.preparation is not None:
                return
            if settings.get("language") not in ("de", "en"):
                self.window.set_status("请先在设置中指定课程语言：德语或英语。")
                self.window.settings_button.click()
                return
            ready, message = (model_status(settings["model_size"]) if settings.get("translate_enabled", True)
                              else model_status(settings["model_size"], translate_enabled=False))
            if not ready:
                self.window.show_error(message)
                return
            self.clear()
            self.last_error = ""
            self._resuming = False
            self._listening_since = None
            self._elapsed_seconds = 0.0
            self._last_session_metrics = {}
            self.close_archive()
            try:
                self.archive = SessionArchive(DATA_ROOT / "exports" / "sessions")
                self.window.set_archive_status(str(self.archive.path))
            except OSError as exc:
                self.window.set_archive_status(None, str(exc))
            self.update_session_metrics()
            self.window.set_busy(True, "正在加载模型 …")
            self._set_tray_state("loading")
            self.worker = SessionWorker(settings, self)
            if hasattr(self.worker, "diagnostic"):
                self.worker.diagnostic.connect(self.diagnostic_updated)
            self.worker.status.connect(self.window.set_status)
            self.worker.error.connect(self.report_error)
            self.worker.level.connect(self.window.set_level)
            self.worker.metrics.connect(self.window.set_metrics)
            self.worker.subtitle.connect(self.receive_subtitle)
            self.worker.listening.connect(self.started)
            self.worker.paused.connect(self.paused)
            self.worker.finished.connect(self.finished)
            self.worker.start()

        def started(self):
            if self.worker and not self.worker.is_stopping:
                resuming = self._resuming
                self._resuming = False
                self.window.set_busy(False)
                self.window.set_running(True)
                self._listening_since = time.monotonic()
                if not resuming:
                    self.window.begin_session()
                self._set_tray_state("listening")
                self.update_session_metrics()

        def receive_subtitle(self, data):
            item = Subtitle(**data)
            self.entries.append(item)
            if self.archive is not None:
                try:
                    self.archive.append(item)
                except OSError as exc:
                    path = str(self.archive.path)
                    self.close_archive()
                    self.window.set_archive_status(path, str(exc))
            self.window.add_subtitle(data)

        def pause(self):
            worker = self.worker
            if worker is None or worker.is_stopping or worker.is_paused:
                return
            self.window.set_pausing()
            self._set_tray_state("pausing")
            worker.request_pause()

        def paused(self):
            self._freeze_elapsed()
            self.window.set_paused(True)
            self._set_tray_state("paused")

        def resume(self):
            worker = self.worker
            if worker is None or worker.is_stopping or not worker.is_paused:
                return
            self.window.set_busy(True, "正在继续，恢复音频采集 …")
            self._set_tray_state("resuming")
            if worker.request_resume():
                self._resuming = True
            else:
                self.window.set_paused(True)
                self._set_tray_state("paused")

        def stop(self):
            if self.preparation:
                self.cancelled_preparation = True
                self.window.set_busy(True, "正在取消模型准备 …")
                self.preparation.kill()
            if self.worker:
                self._resuming = False
                self.window.set_busy(True, "正在结束并释放模型，等待当前推理结束 …")
                self._set_tray_state("ending")
                self.worker.request_stop()

        def finished(self):
            self.update_session_metrics()
            self._freeze_elapsed()
            self._resuming = False
            self.window.end_session()
            self.close_archive()
            if self.worker:
                self.worker.deleteLater()
                self.worker = None
            self.window.set_busy(False)
            self.window.set_running(False)
            self.window.set_status(f"已停止 · {self.last_error.splitlines()[0]}" if self.last_error else "已停止 · 可以导出本次字幕")
            self._set_tray_state("idle")
            if self.closing:
                self.window.close()

        def prepare(self, settings):
            if self.worker or self.preparation:
                return
            self.window.set_busy(True, "正在准备模型，请保持网络连接 …")
            self.cancelled_preparation = False
            process = QProcess(self)
            self.preparation = process
            environment = QProcessEnvironment.systemEnvironment()
            environment.remove("HF_HUB_OFFLINE")
            environment.remove("TRANSFORMERS_OFFLINE")
            environment.insert("PYTHONIOENCODING", "utf-8")
            environment.insert("PYTHONUTF8", "1")
            process.setProcessEnvironment(environment)
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(self.read_progress)
            process.finished.connect(self.prepared)
            process.errorOccurred.connect(self.prepare_error)
            arguments = ["-u", "-m", "simulinterp.prepare_models", "--asr", settings["model_size"]]
            if not settings.get("translate_enabled", True):
                arguments.append("--asr-only")
            process.start(sys.executable, arguments)

        def read_progress(self):
            if self.preparation:
                output = bytes(self.preparation.readAllStandardOutput()).decode("utf-8", errors="replace")
                lines = [s.strip() for s in output.replace("\r", "\n").splitlines() if s.strip()]
                if lines:
                    self.window.set_status(lines[-1][-240:])

        def prepare_error(self, error):
            if error == QProcess.ProcessError.FailedToStart:
                self.prepared(1)

        def prepared(self, exit_code, *_):
            if self.preparation:
                self.preparation.deleteLater()
                self.preparation = None
            self.window.set_busy(False)
            self.check_models(self.window.selected_model())
            message = "模型准备完成，可开始同传" if exit_code == 0 else "模型准备未完成，可重试续传；详情可运行 download-models.cmd"
            if self.cancelled_preparation:
                message = "已取消模型准备 · 可重新下载以续传"
            self.window.set_status(message)
            if self.closing:
                self.window.close()

        def clear(self):
            self.entries.clear()
            self.window.clear_subtitles()

        def _session_archives(self):
            return list_session_archives(DATA_ROOT / "exports" / "sessions")

        def show_cleanup(self):
            if self.worker is not None or self.preparation is not None:
                self.window.set_status("课程进行中，请先结束并释放，再清理历史课程文件。")
                return
            self.window.open_cleanup(self._session_archives())

        def delete_sessions(self, paths):
            if self.worker is not None or self.preparation is not None:
                self.window.set_status("课程进行中，未删除任何文件。")
                return
            deleted, failures = 0, []
            for name in paths:
                try:
                    Path(name).unlink()
                    deleted += 1
                except OSError as exc:
                    failures.append(f"{Path(name).name}：{exc}")
            if failures:
                self.window.show_error("部分课程文件未能删除：\n" + "\n".join(failures))
            self.window.set_status(f"已删除 {deleted} 个课程文件" if deleted else "未删除课程文件")
            self.window.refresh_cleanup(self._session_archives())

        def report_error(self, message):
            self.last_error = message
            self.window.show_error(message)

        def export(self, filename):
            if not self.entries:
                self.window.set_status("暂无可导出的字幕")
                return
            path = Path(filename)
            try:
                path.write_text(render_srt(self.entries) if path.suffix.lower() == ".srt" else render_text(self.entries), encoding="utf-8-sig")
                self.window.set_status(f"字幕已导出：{path.name}")
            except OSError as exc:
                self.window.show_error(f"导出失败：{exc}")

        def _set_tray_state(self, state):
            self.tray.update_state(state)

        def hide_to_tray(self):
            if not self.tray_available or self._quitting or self.closing:
                return
            show_captions = self.worker is not None or self.window.overlay_button.isChecked()
            self.window.settings_dialog.hide()
            if self.window.cleanup_dialog is not None:
                self.window.cleanup_dialog.hide()
            self.window.hide()
            if show_captions:
                self.window.overlay_button.setChecked(True)
                self.window.overlay.show_on_top()
            self.tray.notify_hidden_once()

        def hide_if_minimized(self):
            if self.window.isMinimized():
                self.hide_to_tray()

        def show_from_tray(self):
            if self.window.isMinimized():
                self.window.showNormal()
            else:
                self.window.show()
            self.window.raise_()
            self.window.activateWindow()

        def quit_app(self):
            self._quitting = True
            if self.worker or self.preparation:
                self.closing = True
                self.stop()
                return
            self.closing = True
            self.close_archive()
            if self.monitor.isRunning():
                self.monitor.request_stop()
            else:
                self.window.close()
                QApplication.instance().quit()

        def eventFilter(self, obj, event):
            if (obj is self.window and event.type() == QEvent.Type.WindowStateChange
                    and self.tray_available and self.window.isMinimized()
                    and not self._quitting and not self.closing):
                QTimer.singleShot(0, self.hide_if_minimized)
            if obj is self.window and event.type() == QEvent.Type.Close:
                if self.tray_available and not self._quitting and not self.closing:
                    self.hide_to_tray()
                    event.ignore()
                    return True  # Close keeps the session; restore from the tray.
                if self.worker or self.preparation:
                    self.closing = True
                    self.stop()
                    event.ignore()
                    return True  # Keep QThread alive until inference releases models.
                self.close_archive()
                if self.monitor.isRunning():
                    self.closing = True
                    self.monitor.request_stop()
                    event.ignore()
                    return True  # Defer window deletion until monitor releases NVML.
            return super().eventFilter(obj, event)

    app = QApplication(sys.argv)
    app.setApplicationName("本地同传")
    app.setWindowIcon(application_icon())
    window = MainWindow()
    controller = Controller(window)
    window.show()
    try:
        return app.exec()
    finally:
        # The overlay has no native owner, so it needs explicit final cleanup.
        window.overlay.close()
        controller.tray.hide()
        controller.monitor.request_stop()
        controller.monitor.wait(3000)
        controller.close_archive()


if __name__ == "__main__":
    raise SystemExit(main())
