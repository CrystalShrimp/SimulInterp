"""Cleanup wiring through the real Controller in a subprocess."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROBE = r'''
import json, sys
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox
from simulinterp.app import main
from simulinterp.audio import AudioDevice

root = Path(sys.argv[1])
sessions_dir = root / "exports" / "sessions"
sessions_dir.mkdir(parents=True)
keep = sessions_dir / "20261027-080000-keepme111.srt"
gone = sessions_dir / "20261028-093000-goneme222.srt"
keep.write_text("1\n00:00:01,000 --> 00:00:02,000\n保留\n\n", encoding="utf-8")
gone.write_text("1\n00:00:01,000 --> 00:00:02,000\n删除\n\n", encoding="utf-8")

report = {"errors": [], "stages": []}
original_exec = QApplication.exec

def run(app):
    window = next(w for w in app.topLevelWidgets() if w.__class__.__name__ == "MainWindow")
    control = next(c for c in window.children() if c.__class__.__name__ == "Controller")

    def fail(exc):
        report["errors"].append(repr(exc))
        window.close()
        QApplication.instance().quit()

    stage = {"n": 0}

    def step():
        if report["errors"]:
            return
        try:
            n = stage["n"]
            if n == 0:
                window.cleanup_button.click()
                stage["n"] += 1
            elif n == 1 and window.cleanup_dialog is not None and window.cleanup_dialog.isVisible():
                dialog = window.cleanup_dialog
                assert dialog.list_widget.count() == 2
                assert "goneme222" in dialog.list_widget.item(0).text(), "newest first"
                dialog.list_widget.item(0).setCheckState(Qt.CheckState.Checked)
                stage["n"] += 1
                report["stages"].append("listed")
                dialog.delete_button.click()
            elif n == 2 and not gone.exists():
                assert keep.read_text(encoding="utf-8-sig").startswith("1\n")
                assert window.cleanup_dialog.list_widget.count() == 1
                assert "keepme111" in window.cleanup_dialog.list_widget.item(0).text()
                assert "已删除 1 个课程文件" in window.status_label.text()
                stage["n"] += 1
                report["stages"].append("deleted-one")
                window.cleanup_dialog.select_all.setChecked(True)
                window.cleanup_dialog.delete_button.click()
            elif n == 3 and not keep.exists():
                assert list(sessions_dir.glob("*.srt")) == []
                assert window.cleanup_dialog.list_widget.count() == 1
                assert "暂无" in window.cleanup_dialog.list_widget.item(0).text()
                stage["n"] += 1
                report["stages"].append("deleted-all")
                stepper.stop()
                window.close()
        except Exception as exc:
            fail(exc)

    stepper = QTimer()
    stepper.setInterval(40)
    stepper.timeout.connect(step)
    stepper.start()
    QTimer.singleShot(15000, lambda: fail(TimeoutError("cleanup probe stalled at %d" % stage["n"])))
    original_exec()
    stepper.stop()
    report["monitor_released"] = not control.monitor.isRunning()
    return 0

QApplication.exec = run
from contextlib import ExitStack
with ExitStack() as stack:
    stack.enter_context(patch("simulinterp.config.DATA_ROOT", root))
    stack.enter_context(patch("simulinterp.config.model_status", lambda _: (True, "ready")))
    stack.enter_context(patch("simulinterp.audio.list_audio_devices", lambda: [AudioDevice(0, "fake", "loopback", 16000, 1, True)]))
    stack.enter_context(patch("simulinterp.ui.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes))
    main()
assert not report["errors"], report
assert report["stages"] == ["listed", "deleted-one", "deleted-all"], report
assert report.get("monitor_released"), report
print(json.dumps(report))
'''


def test_settings_cleanup_deletes_one_then_all_session_archives(tmp_path):
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, encoding="utf-8", timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((tmp_path / "exports" / "sessions").glob("*.srt")) == []
