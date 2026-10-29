"""Tray menu behavior with a fake icon; no system tray is required."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

from simulinterp.tray import TrayController


class FakeTrayIcon(QObject):
    activated = Signal(object)

    def __init__(self, icon, parent=None):
        super().__init__(parent)
        self.shown = False
        self.messages = []
        self.tooltip = ""
        self.context_menu = None

    def setContextMenu(self, menu):
        self.context_menu = menu

    def show(self):
        self.shown = True

    def hide(self):
        self.shown = False

    def setToolTip(self, text):
        self.tooltip = text

    def showMessage(self, title, message, icon, duration):
        self.messages.append((title, message))


class TrayControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.controller = TrayController(tray_factory=FakeTrayIcon, available=True)
        self.icon = self.controller.tray

    def test_show_reports_availability_and_only_shows_when_present(self):
        hidden = TrayController(tray_factory=FakeTrayIcon, available=False)
        self.assertFalse(hidden.show())
        self.assertFalse(hidden.tray.shown)
        self.assertTrue(self.controller.show())
        self.assertTrue(self.icon.shown)
        self.controller.hide()
        self.assertFalse(self.icon.shown)

    def test_menu_actions_follow_the_session_state(self):
        self.controller.update_state("idle")
        self.assertFalse(self.controller.pause_action.isEnabled())
        self.assertFalse(self.controller.end_action.isEnabled())
        self.controller.update_state("listening")
        self.assertEqual(self.controller.pause_action.text(), "暂停")
        self.assertTrue(self.controller.pause_action.isEnabled())
        self.assertTrue(self.controller.end_action.isEnabled())
        self.assertIn("正在聆听", self.icon.tooltip)
        self.controller.update_state("paused")
        self.assertEqual(self.controller.pause_action.text(), "继续")
        self.assertIn("已暂停", self.icon.tooltip)
        for state in ("pausing", "resuming", "loading"):
            self.controller.update_state(state)
            self.assertFalse(self.controller.pause_action.isEnabled())
            self.assertTrue(self.controller.end_action.isEnabled())
        self.controller.update_state("ending")
        self.assertFalse(self.controller.pause_action.isEnabled())
        self.assertFalse(self.controller.end_action.isEnabled())
        self.controller.update_state("unknown-state")
        self.assertIn("待开始", self.icon.tooltip)

    def test_pause_action_emits_pause_or_resume_from_the_current_state(self):
        paused, resumed = QSignalSpy(self.controller.pause_requested), QSignalSpy(self.controller.resume_requested)
        self.controller.update_state("listening")
        self.controller.pause_action.trigger()
        self.assertEqual(paused.count(), 1)
        self.controller.update_state("paused")
        self.controller.pause_action.trigger()
        self.assertEqual(resumed.count(), 1)
        self.assertEqual(paused.count(), 1)

    def test_hidden_notification_is_shown_once_and_only_when_available(self):
        unavailable = TrayController(tray_factory=FakeTrayIcon, available=False)
        unavailable.notify_hidden_once()
        self.assertEqual(self.icon.messages, [])
        self.controller.notify_hidden_once()
        self.controller.notify_hidden_once()
        self.assertEqual(len(self.icon.messages), 1)
        self.assertIn("托盘", self.icon.messages[0][1])

    def test_double_click_requests_the_window(self):
        shown = QSignalSpy(self.controller.show_requested)
        self.controller.tray.activated.emit(QSystemTrayIcon.ActivationReason.DoubleClick)
        self.assertEqual(shown.count(), 1)
        self.controller.tray.activated.emit(QSystemTrayIcon.ActivationReason.Context)
        self.assertEqual(shown.count(), 1)

    def test_overlay_menu_follows_visibility_without_recursive_commands(self):
        changes = QSignalSpy(self.controller.overlay_requested)
        self.controller.set_overlay_visible(True)
        self.assertTrue(self.controller.overlay_action.isChecked())
        self.assertEqual(changes.count(), 0)
        self.controller.overlay_action.trigger()
        self.assertEqual(changes.at(0)[0], False)
        self.controller.overlay_action.trigger()
        self.assertEqual(changes.at(1)[0], True)


if __name__ == "__main__":
    unittest.main()
