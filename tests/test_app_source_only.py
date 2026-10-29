"""Actual controller readiness, source-only display/archive/export, no models."""
import os
from pathlib import Path
import subprocess
import sys


PROBE = r'''
import json, sys
from pathlib import Path
from threading import Event
from unittest.mock import patch
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication
from simulinterp.app import main
from simulinterp.audio import AudioDevice
from simulinterp import config

root = Path(sys.argv[1])
model_root = root / 'models'
asr = model_root / 'whisper-large-v3-turbo'
asr.mkdir(parents=True)
for name in config.ASR_REQUIRED:
    (asr / name).write_bytes(b'asset')
original_exec = QApplication.exec
report = {'errors': []}

class Monitor(QObject):
    snapshot, finished = Signal(dict), Signal()
    def __init__(self, parent):
        super().__init__(parent); self.running = False
    def start(self): self.running = True
    def isRunning(self): return self.running
    def request_stop(self):
        if self.running:
            self.running = False; self.finished.emit()
    def wait(self, *_): return True

class Worker(QThread):
    subtitle, status, error = Signal(dict), Signal(str), Signal(str)
    level, metrics, listening, paused = Signal(float), Signal(int, float), Signal(), Signal()
    def __init__(self, settings, parent):
        super().__init__(parent); self.settings = settings; self.stop_event = Event()
    @property
    def is_stopping(self): return self.stop_event.is_set()
    @property
    def is_paused(self): return False
    def request_stop(self): self.stop_event.set()
    def request_pause(self): pass
    def request_resume(self): return False
    def session_metrics(self): return {'translate_enabled': False, 'asr_seconds': .2, 'translation_seconds': 0}
    def run(self):
        assert self.settings['translate_enabled'] is False
        self.listening.emit()
        self.subtitle.emit(dict(start=1, end=2, source='This is the original sentence.',
                                text='This is the original sentence.', language='en'))
        self.stop_event.wait(5)

def run(app):
    window = next(w for w in app.topLevelWidgets() if w.__class__.__name__ == 'MainWindow')
    control = next(c for c in window.children() if c.__class__.__name__ == 'Controller')
    stage = [0]
    def step():
        try:
            if stage[0] == 0:
                window.language_combo.setCurrentIndex(window.language_combo.findData('en'))
                assert not window._ready, 'Translation mode must still require MT assets'
                window.translate_check.setChecked(False)
                assert window._ready and window.start_button.isEnabled(), 'ASR-only assets should suffice'
                window.translate_check.setChecked(True)
                assert not window._ready
                window.translate_check.setChecked(False)
                window.source_check.setChecked(False)
                window.start_button.click()
                stage[0] = 1
            elif stage[0] == 1 and control.entries:
                expected = 'This is the original sentence.'
                assert window.subtitle_label.text() == expected
                assert window.overlay.chinese.text() == expected
                assert not window.overlay.original.text()
                assert window.history_browser.toPlainText().count(expected) == 1
                assert control.archive.count == 1
                assert expected in control.archive.path.read_text(encoding='utf-8-sig')
                control.export(str(root / 'source.srt'))
                control.export(str(root / 'source.txt'))
                assert (root / 'source.srt').read_text(encoding='utf-8-sig').count(expected) == 1
                assert (root / 'source.txt').read_text(encoding='utf-8-sig').count(expected) == 1
                assert json.loads((root / 'settings.json').read_text(encoding='utf-8'))['translate_enabled'] is False
                report['complete'] = True
                timer.stop(); control.quit_app()
        except Exception as exc:
            report['errors'].append(repr(exc)); timer.stop(); control.quit_app()
    timer = QTimer(); timer.setInterval(25); timer.timeout.connect(step); timer.start()
    QTimer.singleShot(8000, app.quit)
    return original_exec()

QApplication.exec = run
with patch('simulinterp.config.DATA_ROOT', root), patch('simulinterp.config.MODEL_ROOT', model_root), \
     patch('simulinterp.engine.SessionWorker', Worker), patch('simulinterp.performance.ResourceMonitor', Monitor), \
     patch('simulinterp.tray.TrayController.show', return_value=False), \
     patch('simulinterp.audio.list_audio_devices', lambda: [AudioDevice(1, 'fake mic', 'microphone', 16000, 1, True)]):
    main()
assert report.get('complete') and not report['errors'], report
print(json.dumps(report))
'''


def test_source_only_controller_accepts_missing_mt_and_exports_original(tmp_path):
    environment = os.environ.copy()
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    result = subprocess.run([sys.executable, '-c', PROBE, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
