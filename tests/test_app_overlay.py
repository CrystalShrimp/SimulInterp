"""Floating subtitles remain independent of the real Controller's tray state."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROBE = r'''
import json, sys
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QSystemTrayIcon as RealTray
from simulinterp.app import main
from simulinterp.audio import AudioDevice

root, mode = Path(sys.argv[1]), sys.argv[2]
report = {"errors": [], "mode": mode}
original_exec = QApplication.exec

class Worker(QObject):
    subtitle, status, error = Signal(dict), Signal(str), Signal(str)
    level, metrics = Signal(float), Signal(int, float)
    listening, paused, finished = Signal(), Signal(), Signal()
    is_stopping, is_paused = False, False
    def __init__(self, settings, parent):
        super().__init__(parent)
    def start(self):
        self.listening.emit()
        self.send(0, "收起前")
    def send(self, start, text):
        self.subtitle.emit(dict(start=start, end=start+1, source="source",
                                text=text, language="en"))
    def request_stop(self):
        self.is_stopping = True
        QTimer.singleShot(0, self.finished.emit)
    def session_metrics(self):
        return dict(queue_depth=0, queue_audio_seconds=0)

class Icon(QObject):
    activated = Signal(object)
    MessageIcon, ActivationReason = RealTray.MessageIcon, RealTray.ActivationReason
    def __init__(self, icon, parent):
        super().__init__(parent)
    @staticmethod
    def isSystemTrayAvailable():
        return True
    def setContextMenu(self, menu): pass
    def show(self): pass
    def hide(self): pass
    def setToolTip(self, text): pass
    def showMessage(self, *args): pass

def run(app):
    window = next(w for w in app.topLevelWidgets() if w.__class__.__name__ == "MainWindow")
    control = next(c for c in window.children() if c.__class__.__name__ == "Controller")
    overlay = window.overlay
    stage = {"n": 0}
    def fail(exc):
        report["errors"].append(repr(exc))
        timer.stop()
        control.quit_app()
    def step():
        try:
            n = stage["n"]
            if n == 0:
                window.language_combo.setCurrentIndex(window.language_combo.findData("en"))
                window.settings_dialog.close()
                window.start_button.click()
                assert control.worker is not None
            elif n == 1:
                assert not overlay.isVisible()
                if mode == "minimize":
                    window.showMinimized()
                elif mode == "close":
                    window.close()
                else:
                    window.tray_button.click()
            elif n == 2:
                assert not window.isVisible(), "main window must be in the tray"
                assert overlay.isVisible(), "active captions must appear automatically"
                assert overlay.parentWidget() is None
                assert overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
                assert overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
                assert not overlay.testAttribute(Qt.WidgetAttribute.WA_QuitOnClose)
                assert window.overlay_button.isChecked()
                assert control.tray.overlay_action.isChecked()
                control.worker.send(1, "托盘期间新字幕")
                assert overlay.chinese.text() == "托盘期间新字幕"
                assert len(control.entries) == 2
                overlay.close()
                assert not window.overlay_button.isChecked()
                assert not control.tray.overlay_action.isChecked()
            elif n == 3:
                # Closing the last visible caption window must not quit the app.
                assert not window.isVisible() and not overlay.isVisible()
                control.worker.send(2, "主动隐藏后")
                assert not overlay.isVisible(), "new captions must respect manual hiding"
                control.tray.show_requested.emit()
            elif n == 4:
                assert window.isVisible() and not window.isMinimized()
                assert not overlay.isVisible()
                window.overlay_button.click()
                window.close()
                assert overlay.isVisible(), "an already visible overlay must stay visible"
                control.tray.overlay_action.trigger()
                assert not overlay.isVisible()
                assert not window.overlay_button.isChecked()
            elif n == 5:
                control.tray.overlay_action.trigger()
                assert overlay.isVisible() and window.overlay_button.isChecked()
                assert not window.isVisible()
                timer.stop()
                control.tray.quit_requested.emit()
            stage["n"] += 1
        except Exception as exc:
            fail(exc)
    timer = QTimer()
    timer.setInterval(40)
    timer.timeout.connect(step)
    timer.start()
    QTimer.singleShot(10000, lambda: fail(TimeoutError(str(stage))))
    original_exec()
    report.update(stage=stage["n"], overlay_closed=not overlay.isVisible(),
                  monitor_stopped=not control.monitor.isRunning())
    return 0

QApplication.exec = run
with patch("simulinterp.config.DATA_ROOT", root), \
     patch("simulinterp.engine.SessionWorker", Worker), \
     patch("simulinterp.config.model_status", lambda _: (True, "ready")), \
     patch("simulinterp.audio.list_audio_devices", lambda: [AudioDevice(0, "fake", "loopback", 16000, 1, True)]), \
     patch("simulinterp.tray.QSystemTrayIcon", Icon):
    main()
assert not report["errors"], report
assert report["stage"] == 6 and report["overlay_closed"] and report["monitor_stopped"], report
print(json.dumps(report))
'''


@pytest.mark.parametrize("mode", ["minimize", "button", "close"])
def test_tray_keeps_live_captions_and_manual_visibility_controls(tmp_path, mode):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path), mode],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
