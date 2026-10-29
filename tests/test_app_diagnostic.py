"""Actual Qt/controller/diagnostic writer, with simulated input and inference."""
import os
from pathlib import Path
import subprocess
import sys


PROBE = r'''
import json, sys, wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from simulinterp.app import main
from simulinterp.audio import AudioDevice
from simulinterp.engine import SessionWorker
from simulinterp.segmenter import SpeechChunk

root = Path(sys.argv[1])
report, original_exec = {"errors": []}, QApplication.exec
class Worker(SessionWorker):
    def run(self):
        self._capture = SimpleNamespace(statistics=lambda: {}, request_stop=lambda: None)
        self.listening.emit()
        block, tick = np.full(320, .1, np.float32), 0
        try:
            while not self.is_stopping:
                self._tap_diagnostic_audio(block)
                tick += 1
                if tick % 10 == 0:
                    chunk = SpeechChunk(block, (tick-10)/50, tick/50)
                    self._record_diagnostic_transcript(chunk, SimpleNamespace(text="example phrase", start=0, end=.2), .01)
                self._stop_event.wait(.01)
        finally:
            self.request_diagnostic_stop(wait=True)
            self._capture = None

def run(app):
    window = next(w for w in app.topLevelWidgets() if w.__class__.__name__ == "MainWindow")
    control = next(c for c in window.children() if c.__class__.__name__ == "Controller")
    stage = [0]
    def step():
        try:
            if stage[0] == 0:
                window.language_combo.setCurrentIndex(window.language_combo.findData("en"))
                window.start_button.click()
                stage[0] = 1
            elif stage[0] == 1 and window._running:
                assert window.diagnostic_button.isEnabled()
                window.diagnostic_button.click()
                stage[0] = 2
            elif stage[0] == 2 and window._diagnostic.get("recorded_seconds", 0) >= .3:
                window.diagnostic_stop_button.click()
                stage[0] = 3
            elif stage[0] == 3 and window._diagnostic["state"] == "completed":
                assert window._running and control.worker is not None, "diagnostic Stop must not end the class"
                folder = Path(window._diagnostic["path"])
                with wave.open(str(folder / "audio.wav")) as recording:
                    assert recording.getnframes() > 0
                    assert recording.getframerate() == 16000
                metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
                assert metadata["settings"]["language"] == "en"
                assert metadata["recognition"]["beam_size"] == 3
                assert "example phrase" in (folder / "transcripts.jsonl").read_text(encoding="utf-8")
                report["complete"] = True
                timer.stop()
                control.quit_app()
        except Exception as exc:
            report["errors"].append(repr(exc))
            timer.stop()
            control.quit_app()
    timer = QTimer()
    timer.setInterval(40)
    timer.timeout.connect(step)
    timer.start()
    QTimer.singleShot(10000, app.quit)
    original_exec()
    assert not control.monitor.isRunning()
    return 0

QApplication.exec = run
with patch("simulinterp.config.DATA_ROOT", root), \
     patch("simulinterp.engine.SessionWorker", Worker), \
     patch("simulinterp.config.model_status", lambda _: (True, "ready")), \
     patch("simulinterp.audio.list_audio_devices", lambda: [AudioDevice(0, "fake", "loopback", 16000, 1, True)]):
    main()
assert not report["errors"] and report.get("complete"), report
print(json.dumps(report))
'''


def test_diagnostic_button_records_existing_stream_and_leaves_class_running(tmp_path):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
