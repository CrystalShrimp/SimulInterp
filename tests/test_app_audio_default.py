"""Actual controller/UI device selection, with enumeration but no capture."""
import os
from pathlib import Path
import subprocess
import sys


PROBE = r'''
import json, sys
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication
from simulinterp.app import main
from simulinterp.audio import AudioDevice

root = Path(sys.argv[1])
loop = AudioDevice(0, "Default speakers", "loopback", 48000, 2, True)
other_loop = AudioDevice(4, "Other speakers", "loopback", 48000, 2)
mic = AudioDevice(2, "Other microphone", "microphone", 48000, 1)
default_mic = AudioDevice(3, "Windows default microphone", "microphone", 48000, 1, True)
devices = [loop, mic, default_mic]
report = {"errors": []}
original_exec = QApplication.exec

class Monitor(QObject):
    snapshot, finished = Signal(dict), Signal()
    def __init__(self, parent):
        super().__init__(parent)
        self.running = False
    def start(self):
        self.running = True
    def isRunning(self):
        return self.running
    def request_stop(self):
        if self.running:
            self.running = False
            self.finished.emit()
    def wait(self, *_):
        return True

def run(app):
    window = next(w for w in app.topLevelWidgets() if w.__class__.__name__ == "MainWindow")
    control = next(c for c in window.children() if c.__class__.__name__ == "Controller")
    def check():
        try:
            # Enumeration order deliberately begins with default loopback.
            assert window.device_combo.currentData() == 3
            assert "麦克风" in window.device_combo.currentText()
            assert control.worker is None
            window.device_combo.setCurrentIndex(window.device_combo.findData(0))
            control.refresh_devices()
            assert window.device_combo.currentData() == 0, "Keep manual loopback choice on refresh"
            window.device_combo.setCurrentIndex(window.device_combo.findData(2))
            control.refresh_devices()
            assert window.device_combo.currentData() == 2, "Keep manual microphone choice on refresh"
            devices[:] = [loop, default_mic]
            control.refresh_devices()
            assert window.device_combo.currentData() == 3, "Removed selection falls back to default microphone"
            devices[:] = [loop, AudioDevice(6, "Nondefault microphone", "microphone", 48000, 1)]
            control.refresh_devices()
            assert window.device_combo.currentData() == 6, "Any microphone beats default loopback"
            devices[:] = [other_loop, loop]
            control.refresh_devices()
            assert window.device_combo.currentData() == 0, "Use default loopback only without microphones"
            devices[:] = [other_loop]
            control.refresh_devices()
            assert window.device_combo.currentData() == 4
            devices.clear()
            control.refresh_devices()
            assert window.device_combo.currentData() is None
            assert not window.start_button.isEnabled()
            devices[:] = [loop, default_mic]
            control.refresh_devices()
            assert window.device_combo.currentData() == 3
            assert control.worker is None, "Device selection must never start recording"
            report["complete"] = True
        except Exception as exc:
            report["errors"].append(repr(exc))
        finally:
            control.quit_app()
    QTimer.singleShot(20, check)
    QTimer.singleShot(10000, app.quit)
    return original_exec()

QApplication.exec = run
with patch("simulinterp.config.DATA_ROOT", root), \
     patch("simulinterp.audio.list_audio_devices", lambda: list(devices)), \
     patch("simulinterp.audio.AudioCapture.start", side_effect=AssertionError("No capture before Start")), \
     patch("simulinterp.engine.SessionWorker", side_effect=AssertionError("No model worker before Start")), \
     patch("simulinterp.performance.ResourceMonitor", Monitor), \
     patch("simulinterp.tray.TrayController.show", return_value=False), \
     patch("simulinterp.config.model_status", return_value=(True, "ready")):
    main()
assert report.get("complete") and not report["errors"], report
print(json.dumps(report))
'''


def test_default_microphone_and_refresh_fallback_do_not_start_capture(tmp_path):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
