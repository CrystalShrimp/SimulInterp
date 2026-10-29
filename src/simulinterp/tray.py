"""System tray commands; the controller owns capture and shutdown decisions."""
from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon


def application_icon() -> QIcon:
    """Render a recognizable icon without external assets or font dependency."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(size / 128, size / 128)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#111e28"))
        painter.drawRoundedRect(QRectF(0, 0, 128, 128), 29, 29)
        painter.setBrush(QColor("#63d7bc"))
        painter.drawRoundedRect(QRectF(19, 19, 90, 78), 20, 20)
        tail = QPainterPath()
        tail.moveTo(38, 91)
        tail.lineTo(34, 111)
        tail.lineTo(61, 94)
        tail.closeSubpath()
        painter.drawPath(tail)
        pen = QPen(QColor("#12352f"), 7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        for index, height in enumerate((12, 26, 38, 22, 12)):
            x = 37 + index * 13.5
            painter.drawLine(QPointF(x, 48 - height / 2), QPointF(x, 48 + height / 2))
        painter.setPen(QPen(QColor("#f1fff9"), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(37, 78), QPointF(62, 78))
        painter.drawLine(QPointF(73, 78), QPointF(91, 78))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class TrayController(QObject):
    show_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    end_requested = Signal()
    quit_requested = Signal()
    overlay_requested = Signal(bool)

    _LABELS = {
        "idle": "待开始", "loading": "正在准备", "listening": "正在聆听",
        "pausing": "正在暂停", "paused": "已暂停 · 模型保留",
        "resuming": "正在继续", "ending": "正在结束并释放",
    }

    def __init__(self, parent=None, *, tray_factory=None, available: bool | None = None):
        super().__init__(parent)
        factory = tray_factory or QSystemTrayIcon
        self._available_override = available
        self._notified_hidden = False
        self._state = "idle"
        self.tray = factory(application_icon(), self)
        self.menu = QMenu()
        self.show_action = self.menu.addAction("恢复主窗口")
        self.show_action.triggered.connect(lambda _checked=False: self.show_requested.emit())
        self.overlay_action = self.menu.addAction("悬浮字幕")
        self.overlay_action.setCheckable(True)
        self.overlay_action.toggled.connect(self.overlay_requested.emit)
        self.menu.addSeparator()
        self.pause_action = self.menu.addAction("暂停")
        self.pause_action.triggered.connect(self._pause_or_resume)
        self.end_action = self.menu.addAction("结束并释放")
        self.end_action.triggered.connect(lambda _checked=False: self.end_requested.emit())
        self.menu.addSeparator()
        self.quit_action = self.menu.addAction("退出")
        self.quit_action.triggered.connect(lambda _checked=False: self.quit_requested.emit())
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._activated)
        self.update_state("idle")

    @property
    def tray_available(self) -> bool:
        if self._available_override is not None:
            return self._available_override
        return bool(QSystemTrayIcon.isSystemTrayAvailable())

    def show(self) -> bool:
        if not self.tray_available:
            return False
        self.tray.show()
        return True

    def hide(self) -> None:
        self.tray.hide()

    def set_overlay_visible(self, visible: bool) -> None:
        blocked = self.overlay_action.blockSignals(True)
        self.overlay_action.setChecked(visible)
        self.overlay_action.blockSignals(blocked)

    def update_state(self, state: str) -> None:
        self._state = state if state in self._LABELS else "idle"
        paused = self._state == "paused"
        self.pause_action.setText("继续" if paused else "暂停")
        self.pause_action.setEnabled(self._state in ("listening", "paused"))
        self.end_action.setEnabled(self._state not in ("idle", "ending"))
        self.tray.setToolTip("本地同传 · " + self._LABELS[self._state])

    def notify_hidden_once(self) -> None:
        if not self.tray_available or self._notified_hidden:
            return
        self._notified_hidden = True
        self.tray.showMessage("本地同传仍在运行", "双击托盘图标恢复窗口；右键可暂停、结束或退出。",
                              QSystemTrayIcon.MessageIcon.Information, 4000)

    def _pause_or_resume(self, _checked=False) -> None:
        if self._state == "paused":
            self.resume_requested.emit()
        elif self._state == "listening":
            self.pause_requested.emit()

    def _activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_requested.emit()
