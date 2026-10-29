"""Exercise actual QApplication close/worker/archive integration in isolation."""
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
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication
from simulinterp.app import main
from simulinterp.audio import AudioDevice

root, disk_failure = Path(sys.argv[1]), sys.argv[2] == "fail"
report = {"errors": []}
original_exec = QApplication.exec

class Worker(QThread):
    subtitle, status, error = Signal(dict), Signal(str), Signal(str)
    level, metrics, listening, paused = Signal(float), Signal(int, float), Signal(), Signal()
    def __init__(self, settings, parent):
        super().__init__(parent)
        self.stop_event = Event()
    @property
    def is_stopping(self):
        return self.stop_event.is_set()
    @property
    def is_paused(self):
        return False
    def request_stop(self):
        self.stop_event.set()
    def request_pause(self):
        pass
    def request_resume(self):
        return False
    def session_metrics(self):
        return dict(queue_depth=0, queue_audio_seconds=0)
    def run(self):
        self.listening.emit()
        self.subtitle.emit(dict(start=5398, end=5400, source="final phrase",
                                text="最后一句", language="de"))
        self.stop_event.wait(5)

def run(app):
    window = next(w for w in app.topLevelWidgets() if w.__class__.__name__ == "MainWindow")
    control = next(c for c in window.children() if c.__class__.__name__ == "Controller")
    def guard(fn):
        def call():
            try:
                fn()
            except Exception as exc:
                report["errors"].append(repr(exc))
                window.close()
        return call
    def begin():
        window.language_combo.setCurrentIndex(window.language_combo.findData("de"))
        window.start_button.click()
        control.worker.finished.connect(guard(finished))
    def stop():
        assert window._running and len(control.entries) == 1
        if disk_failure:
            assert control.archive is None
            assert window.archive_status_label.text() == "保存异常"
            assert "手动导出" in window.performance_warning.text()
        else:
            assert control.archive.count == 1
            assert "最后一句" in control.archive.path.read_text(encoding="utf-8-sig")
        control.export(str(root / "manual.srt"))
        window.close()
        assert window.isVisible(), "QCloseEvent must be ignored until worker has drained"
    def finished():
        assert control.worker is None and control.archive is None
        report["worker_released"] = True
    QTimer.singleShot(25, guard(begin))
    QTimer.singleShot(250, guard(stop))
    QTimer.singleShot(6000, app.quit)
    result = original_exec()
    report["monitor_released"] = not control.monitor.isRunning()
    return result

QApplication.exec = run
failure = patch("simulinterp.archive.SessionArchive.append", side_effect=OSError("disk full"))
from contextlib import ExitStack
with ExitStack() as stack:
    stack.enter_context(patch("simulinterp.engine.SessionWorker", Worker))
    stack.enter_context(patch("simulinterp.config.DATA_ROOT", root))
    stack.enter_context(patch("simulinterp.config.model_status", lambda _: (True, "ready")))
    stack.enter_context(patch("simulinterp.audio.list_audio_devices", lambda: [AudioDevice(0, "fake", "loopback", 16000, 1, True)]))
    if disk_failure:
        stack.enter_context(failure)
    main()
assert not report["errors"], report
assert report.get("worker_released") and report.get("monitor_released"), report
print(json.dumps(report))
'''


@pytest.mark.parametrize("mode", ["normal", "fail"])
def test_close_during_inference_drains_and_preserves_export_on_archive_failure(tmp_path, mode):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path), mode],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "01:29:58,000 --> 01:30:00,000\n最后一句" in (
        tmp_path / "manual.srt").read_text(encoding="utf-8-sig")
