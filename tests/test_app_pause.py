"""Pause/resume and tray wiring through the real Controller in a subprocess."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROBE = r'''
import json, sys
from pathlib import Path
from threading import Event
from unittest.mock import patch
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QSystemTrayIcon as RealTray
from simulinterp.app import main
from simulinterp.audio import AudioDevice

root = Path(sys.argv[1])
report = {"errors": [], "stages": []}
original_exec = QApplication.exec

class Worker(QThread):
    subtitle, status, error = Signal(dict), Signal(str), Signal(str)
    level, metrics, listening, paused = Signal(float), Signal(int, float), Signal(), Signal()
    def __init__(self, settings, parent):
        super().__init__(parent)
        self._stop = Event()
        self._pause = Event()
        self._paused = Event()
        self._wake = Event()
    @property
    def is_stopping(self):
        return self._stop.is_set()
    @property
    def is_paused(self):
        return self._paused.is_set()
    def request_stop(self):
        self._stop.set()
        self._wake.set()
    def request_pause(self):
        self._pause.set()
    def request_resume(self):
        if self.is_stopping or not self._paused.is_set():
            return False
        self._pause.clear()
        self._paused.clear()
        self._wake.set()
        return True
    def session_metrics(self):
        return dict(queue_depth=0, queue_audio_seconds=0)
    def run(self):
        self.listening.emit()
        self.subtitle.emit(dict(start=12.0, end=13.0, source="vor der Pause",
                                text="暂停前的字幕", language="de"))
        while not self._stop.is_set():
            if self._pause.is_set():
                self._paused.set()
                self.paused.emit()
                while self._pause.is_set() and not self._stop.is_set():
                    self._wake.wait(.05)
                    self._wake.clear()
                self._paused.clear()
                if self._stop.is_set():
                    break
                self.listening.emit()
            self._wake.wait(.05)

class FakeSystemTrayIcon(QObject):
    # The offscreen platform has no tray; record what the app would display.
    activated = Signal(object)
    MessageIcon = RealTray.MessageIcon
    ActivationReason = RealTray.ActivationReason

    def __init__(self, icon, parent=None):
        super().__init__(parent)
        self.messages = []
        self.visible = False
        self.tooltip = ""

    @staticmethod
    def isSystemTrayAvailable():
        return True

    def setContextMenu(self, menu):
        pass

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False

    def setToolTip(self, text):
        self.tooltip = text

    def showMessage(self, title, message, icon, duration):
        self.messages.append((title, message))

def run(app):
    window = next(w for w in app.topLevelWidgets() if w.__class__.__name__ == "MainWindow")
    control = next(c for c in window.children() if c.__class__.__name__ == "Controller")
    assert control.tray_available, "fake tray must register as available"

    def fail(exc):
        report["errors"].append(repr(exc))
        control._quitting = True
        window.close()
        QApplication.instance().quit()

    stage = {"n": 0}

    def step():
        if report["errors"]:
            return
        try:
            n = stage["n"]
            if n == 0:
                window.language_combo.setCurrentIndex(window.language_combo.findData("de"))
                window.start_button.click()
                stage["n"] += 1
            elif n == 1 and window._running and control.entries:
                assert "正在聆听" in control.tray.tray.tooltip
                stage["n"] += 1
                report["stages"].append("listening")
                window.pause_button.click()
            elif n == 2 and window._paused:
                assert window.pause_button.text() == "继续"
                assert window.state_badge.text() == "已暂停"
                assert len(control.entries) == 1, "pause must keep subtitle records"
                assert control.archive is not None and control.archive.count == 1
                report["archive_kept"] = str(control.archive.path)
                assert "已暂停" in control.tray.tray.tooltip
                stage["n"] += 1
                report["stages"].append("paused")
                window.pause_button.click()  # 继续
            elif n == 3 and window._running and not window._paused and not window._busy:
                assert str(control.archive.path) == report["archive_kept"], "resume must not restart the archive"
                assert control.archive.count == 1
                assert len(control.entries) == 1, "resume must not clear records"
                stage["n"] += 1
                report["stages"].append("resumed")
                window.stop_button.click()  # 结束并释放
            elif n == 4 and control.worker is None:
                assert control.archive is None
                assert window.state_badge.text() == "已停止"
                assert "待开始" in control.tray.tray.tooltip
                stage["n"] += 1
                report["stages"].append("ended")
                window.close()  # X button must hide to the tray, not quit
            elif n == 5 and not window.isVisible():
                assert control.tray.tray.messages, "hidden notification expected once"
                stage["n"] += 1
                report["stages"].append("hidden")
                control.tray.show_requested.emit()  # tray double click
            elif n == 6 and window.isVisible():
                stage["n"] += 1
                report["stages"].append("restored")
                control.tray.quit_requested.emit()
                stepper.stop()
        except Exception as exc:
            fail(exc)

    stepper = QTimer()
    stepper.setInterval(40)
    stepper.timeout.connect(step)
    stepper.start()
    QTimer.singleShot(15000, lambda: fail(TimeoutError("probe stages stalled at %d" % stage["n"])))
    original_exec()
    stepper.stop()
    report["monitor_released"] = not control.monitor.isRunning()
    report["window_closed"] = not window.isVisible()
    return 0

QApplication.exec = run
from contextlib import ExitStack
with ExitStack() as stack:
    stack.enter_context(patch("simulinterp.engine.SessionWorker", Worker))
    stack.enter_context(patch("simulinterp.config.DATA_ROOT", root))
    stack.enter_context(patch("simulinterp.config.model_status", lambda _: (True, "ready")))
    stack.enter_context(patch("simulinterp.audio.list_audio_devices", lambda: [AudioDevice(0, "fake", "loopback", 16000, 1, True)]))
    stack.enter_context(patch("simulinterp.tray.QSystemTrayIcon", FakeSystemTrayIcon))
    main()
assert not report["errors"], report
assert report["stages"] == ["listening", "paused", "resumed", "ended", "hidden", "restored"], report
assert report.get("monitor_released") and report.get("window_closed"), report
print(json.dumps(report))
'''


def test_pause_resume_end_and_tray_keep_records_and_quit_cleanly(tmp_path):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    sessions = list((tmp_path / "exports" / "sessions").glob("*.srt"))
    assert len(sessions) == 1, "one class session archive across pause/resume"
    assert "暂停前的字幕" in sessions[0].read_text(encoding="utf-8-sig")
