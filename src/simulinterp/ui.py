"""Chinese desktop interface. All audio and model work belongs to the controller."""

from __future__ import annotations

import html
import math
from collections import deque
from typing import Any

from .config import DEFAULT_ASR_PROFILE, DEFAULT_TRANSLATION_THREADS, TRANSLATION_THREAD_OPTIONS

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont, QMouseEvent, QShowEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QScrollArea, QSlider, QStyle, QTextBrowser,
    QVBoxLayout, QWidget,
)


STYLE = """
QWidget { background: #11171d; color: #e7edf2; font-family: 'Microsoft YaHei UI', 'Segoe UI'; font-size: 13px; }
QMainWindow { background: #11171d; }
QFrame#card { background: #19222b; border: 1px solid #293540; border-radius: 14px; }
QFrame#card QLabel, QFrame#card QCheckBox { background: transparent; }
QLabel#title { font-size: 25px; font-weight: 700; }
QLabel#muted { color: #91a1af; }
QLabel#section { font-size: 15px; font-weight: 600; }
QLabel#eyebrow { color: #6dddc8; font-size: 11px; font-weight: 600; }
QLabel#badge { background: #163a35; color: #88e2ce; border: 1px solid #28544b; border-radius: 10px; padding: 5px 11px; }
QLabel#state { background: #23313d; color: #afc1cf; border-radius: 9px; padding: 5px 10px; }
QComboBox { background: #111a22; border: 1px solid #364552; border-radius: 8px; padding: 9px 10px; min-height: 18px; }
QComboBox:hover { border-color: #648c94; }
QComboBox:disabled { color: #6d7e8b; border-color: #2b3741; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #1e2b36; selection-background-color: #28594f; padding: 5px; }
QPushButton { background: #24323d; border: 1px solid #3a4a57; border-radius: 8px; padding: 8px 12px; font-weight: 500; }
QPushButton:hover { background: #30434f; border-color: #678390; }
QPushButton:pressed { background: #192c35; }
QPushButton:disabled { color: #60717f; background: #1b2630; border-color: #2c3944; }
QPushButton#primary { background: #63d7bc; color: #102820; border-color: #63d7bc; font-weight: 700; font-size: 15px; padding: 12px; }
QPushButton#primary:hover { background: #8ce8d3; }
QPushButton#primary:disabled { background: #2c514b; color: #779c92; border-color: #2c514b; }
QPushButton#stop { background: #3c2b32; color: #edb1b7; border-color: #67414a; }
QPushButton#stop:disabled { background: #1b2630; color: #60717f; border-color: #2c3944; }
QPushButton#ghost { background: transparent; padding: 4px 9px; border-color: #354551; font-size: 12px; }
QPushButton:checked { background: #214b43; color: #9af0db; border-color: #438575; }
QCheckBox { spacing: 7px; color: #afbfcb; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #516575; border-radius: 4px; background: #111a22; }
QCheckBox::indicator:checked { background: #63d7bc; border-color: #63d7bc; }
QProgressBar { background: #101920; border: none; border-radius: 3px; height: 5px; }
QProgressBar::chunk { background: #63d7bc; border-radius: 3px; }
QTextBrowser { background: transparent; border: none; color: #d9e3eb; padding: 0px; }
QScrollBar:vertical { background: #141e26; width: 7px; margin: 2px; border-radius: 3px; }
QScrollBar::handle:vertical { background: #40525e; min-height: 25px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollArea { background: transparent; border: none; }
QSlider::groove:horizontal { background: #33434f; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #79dbc4; width: 12px; margin: -4px 0; border-radius: 6px; }
QToolTip { color: #e7edf2; background: #263642; border: 1px solid #4a6572; padding: 6px; }
"""


def _label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    if wrap:
        # A long subtitle must wrap inside its column, never set the column width.
        policy = widget.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        policy.setHeightForWidth(True)
        widget.setSizePolicy(policy)
        widget.setMinimumWidth(0)
    return widget


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(13)
    return frame, layout


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


class SubtitleOverlay(QWidget):
    """A movable subtitle window that stays above ordinary desktop windows."""

    visibility_changed = Signal(bool)

    def __init__(self) -> None:
        # A native owned Tool window follows its owner's minimize state on
        # Windows. Keep captions independent of the main window's visibility.
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setWindowTitle("本地同传 · 悬浮字幕")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self._drag: QPoint | None = None
        self._placed = False
        self.setMinimumWidth(400)
        self.resize(900, 190)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QFrame()
        panel.setStyleSheet("QFrame { background: rgba(12, 19, 25, 232); border: 1px solid #52656b; border-radius: 14px; } QLabel { background: transparent; border: none; color: white; }")
        outer.addWidget(panel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(22, 9, 22, 21)
        layout.setSpacing(6)
        toolbar = QHBoxLayout()
        toolbar.addWidget(_label("拖动空白处移动", "muted"))
        toolbar.addStretch()
        close = QPushButton("×")
        close.setToolTip("关闭悬浮字幕")
        close.setFixedSize(30, 27)
        close.setStyleSheet("QPushButton { background: transparent; color: #aab9c5; border: none; font-size: 21px; padding: 0px; } QPushButton:hover { color: white; }")
        close.clicked.connect(self.close)
        toolbar.addWidget(close)
        layout.addLayout(toolbar)
        self.chinese = _label("等待语音…", wrap=True)
        self.chinese.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chinese.setStyleSheet("font-size: 30px; font-weight: 600;")
        self.original = _label("", wrap=True)
        self.original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original.setStyleSheet("font-size: 16px; color: #b9c8d2;")
        layout.addWidget(self.chinese)
        layout.addWidget(self.original)

    def show_on_top(self) -> None:
        """Show captions without activating the main window or stealing focus."""
        self.show()
        self.raise_()

    def set_subtitle(self, text: str, source: str = "") -> None:
        self.chinese.setText(text or "等待语音…")
        self.original.setText(source)
        self._fit_height()

    def set_source_visible(self, visible: bool) -> None:
        self.original.setVisible(visible)
        self._fit_height()

    def set_font_size(self, size: int) -> None:
        self.chinese.setStyleSheet(f"font-size: {size}px; font-weight: 600;")
        self._fit_height()

    def _fit_height(self) -> None:
        screen = self.screen().availableGeometry()
        # Keep a stable line width as captions arrive; adjustSize() instead uses
        # the text's preferred width, which can grow past the edge of a screen.
        available_width = max(240, screen.width() - 32)
        self.setMinimumWidth(min(400, available_width))
        width = min(max(self.minimumWidth(), self.width()), available_width)
        self.ensurePolished()
        panel = self.chinese.parentWidget()
        panel.layout().invalidate()
        panel.updateGeometry()
        self.layout().invalidate()
        height = self.layout().totalHeightForWidth(width)
        if height < 0:
            height = self.layout().totalSizeHint().height()
        self.resize(width, min(max(96, height), max(96, screen.height() - 32)))
        self.layout().activate()
        if self._placed:
            self.move(max(screen.left(), min(self.x(), screen.right() - self.width() + 1)),
                      max(screen.top(), min(self.y(), screen.bottom() - self.height() + 1)))

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._placed:
            screen = self.screen().availableGeometry()
            width = min(1000, max(min(400, screen.width() - 32), int(screen.width() * 0.72)))
            self.resize(width, self.height())
            self._fit_height()
            self.move(screen.x() + (screen.width() - width) // 2,
                      max(screen.y() + 16, screen.y() + screen.height() - self.height() - 65))
            self._placed = True
        self.visibility_changed.emit(True)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.visibility_changed.emit(False)
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag = None
        super().mouseReleaseEvent(event)


class SessionCleanupDialog(QDialog):
    """Checkable list of past class SRT archives; deletion itself belongs outside."""

    deletion_requested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("本地同传 · 清理课程文件")
        self.setMinimumSize(600, 360)
        self.setStyleSheet(STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)
        layout.addWidget(_label("历史课程文件", "section"))
        self.hint = _label("勾选要删除的课程记录；删除后不可恢复。", "muted", True)
        layout.addWidget(self.hint)
        self.list_widget = QListWidget()
        self.list_widget.setItemAlignment(Qt.AlignmentFlag.AlignLeft)
        self.list_widget.itemChanged.connect(lambda _item: self._sync_footer())
        layout.addWidget(self.list_widget, 1)
        footer = QHBoxLayout()
        self.select_all = QCheckBox("全选")
        self.select_all.toggled.connect(self._toggle_all)
        footer.addWidget(self.select_all)
        footer.addStretch()
        self.delete_button = QPushButton("删除所选")
        self.delete_button.setObjectName("stop")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._confirm_deletion)
        footer.addWidget(self.delete_button)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        footer.addWidget(close)
        layout.addLayout(footer)
        self._sessions: list[dict[str, Any]] = []

    def set_sessions(self, sessions: list[dict[str, Any]]) -> None:
        self._sessions = list(sessions)
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for item in sessions:
            entry = QListWidgetItem(f"{item['name']}.srt · {_file_size(item['size_bytes'])}"
                                    f" · {item['modified']:%Y-%m-%d %H:%M}")
            entry.setData(Qt.ItemDataRole.UserRole, item["path"])
            entry.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            entry.setCheckState(Qt.CheckState.Unchecked)
            self.list_widget.addItem(entry)
        if not sessions:
            empty = QListWidgetItem("暂无历史课程文件")
            empty.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.list_widget.addItem(empty)
        self.list_widget.blockSignals(False)
        self._sync_footer()

    def _checked_paths(self) -> list[str]:
        paths = []
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable and \
                    item.checkState() == Qt.CheckState.Checked:
                paths.append(item.data(Qt.ItemDataRole.UserRole))
        return paths

    def _sync_footer(self) -> None:
        total = len(self._sessions)
        checked = len(self._checked_paths())
        self.delete_button.setEnabled(checked > 0)
        self.delete_button.setText(f"删除所选（{checked}）" if checked else "删除所选")
        self.select_all.blockSignals(True)
        self.select_all.setChecked(0 < checked == total)
        self.select_all.blockSignals(False)

    def _toggle_all(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.list_widget.blockSignals(True)
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)
        self.list_widget.blockSignals(False)
        self._sync_footer()

    def _confirm_deletion(self, _checked: bool = False) -> None:
        paths = self._checked_paths()
        if not paths:
            return
        answer = QMessageBox.question(
            self, "本地同传",
            f"确定删除选中的 {len(paths)} 个课程文件吗？\n删除后不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.deletion_requested.emit(paths)


def _file_size(size_bytes: Any) -> str:
    try:
        size = float(size_bytes)
    except (TypeError, ValueError):
        return "—"
    return f"{size / 1024 ** 2:.1f} MB" if size >= 1024 ** 2 else f"{max(0, size) / 1024:.0f} KB"


class MainWindow(QMainWindow):
    start_requested = Signal(dict)
    stop_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    minimize_to_tray_requested = Signal()
    refresh_devices_requested = Signal()
    export_requested = Signal(str)
    clear_requested = Signal()
    cleanup_requested = Signal()
    delete_sessions_requested = Signal(list)
    model_changed = Signal(str)
    prepare_requested = Signal(dict)
    preferences_changed = Signal(dict)
    diagnostic_requested = Signal()
    diagnostic_stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("本地同传 · 德语 / 英语 → 中文字幕")
        self.resize(1160, 820)
        self.setMinimumSize(900, 680)
        self.setStyleSheet(STYLE)
        self._ready = False
        self._running = False
        self._paused = False
        self._busy = False
        self._busy_caption = "准备中…"
        self._history: deque[dict[str, Any]] = deque(maxlen=200)
        self._latest_source = ""
        self._performance: dict[str, Any] = {}
        self._session_metrics: dict[str, Any] = {}
        self._session_active = False
        self._slow_updates = 0
        self._archive_error = ""
        self._archive_path: str | None = None
        self._subtitle_total = 0
        self._diagnostic: dict[str, Any] = {"state": "idle", "recorded_seconds": 0}
        self.overlay = SubtitleOverlay()
        self.destroyed.connect(self.overlay.deleteLater)
        self.cleanup_dialog: SessionCleanupDialog | None = None
        self._build_settings_dialog()

        central = QWidget()
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(28, 23, 28, 18)
        page.setSpacing(20)
        header = QHBoxLayout()
        mark = _label("译")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(48, 48)
        mark.setStyleSheet("background: #1f5248; color: #a2f2df; border: 1px solid #3f7c6e; border-radius: 14px; font-size: 27px; font-weight: 700;")
        header.addWidget(mark)
        title = QVBoxLayout()
        title.setSpacing(3)
        title.addWidget(_label("本地同传", "title"))
        self.output_hint = _label("德语 / 英语语音  →  简体中文字幕", "muted")
        title.addWidget(self.output_hint)
        header.addSpacing(3)
        header.addLayout(title)
        header.addStretch()
        self.privacy_badge = _label("音频与翻译均在本机处理", "badge")
        header.addWidget(self.privacy_badge)
        self.settings_button = QPushButton("设置")
        self.settings_button.setToolTip("源语言、字幕节奏、识别模型、计算设备与本地模型下载")
        self.settings_button.clicked.connect(self._show_settings)
        header.addWidget(self.settings_button)
        self.tray_button = QPushButton("收起到托盘")
        self.tray_button.setToolTip("保持同传运行；从系统托盘恢复主窗口或退出。")
        self.tray_button.clicked.connect(self.minimize_to_tray_requested.emit)
        header.addWidget(self.tray_button)
        page.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(18)
        settings, options = _card()
        settings.setFixedWidth(320)
        options.setSpacing(10)
        options.addWidget(_label("课堂控制", "section"))
        options.addWidget(_label("音频来源", "muted"))
        input_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.setMinimumContentsLength(12)
        self.device_combo.setToolTip("系统回环：捕获电脑播放的声音；麦克风：捕获现场讲话。")
        input_row.addWidget(self.device_combo, 1)
        self.refresh_button = QPushButton()
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setFixedWidth(35)
        self.refresh_button.setToolTip("刷新音频设备")
        self.refresh_button.clicked.connect(self.refresh_devices_requested.emit)
        input_row.addWidget(self.refresh_button)
        options.addLayout(input_row)
        options.addWidget(_label("听视频或会议选“系统回环”；听现场讲话选麦克风。", "muted", True))
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 1000)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(5)
        self.level_bar.setToolTip("输入音量；开始采集后显示")
        options.addWidget(self.level_bar)
        options.addSpacing(2)
        self._build_dashboard(options)
        self.start_button = QPushButton("开始同传")
        self.start_button.setObjectName("primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(lambda: self.start_requested.emit(self.configuration()))
        options.addWidget(self.start_button)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.setToolTip("停止采集并处理尾句，保留本课字幕和已加载模型。")
        self.pause_button.clicked.connect(self._request_pause_or_resume)
        options.addWidget(self.pause_button)
        self.stop_button = QPushButton("结束并释放")
        self.stop_button.setObjectName("stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        options.addWidget(self.stop_button)
        self.course_hint = _label("暂停保留模型；结束释放资源。", "muted", True)
        options.addWidget(self.course_hint)
        body.addWidget(settings)

        output = QVBoxLayout()
        output.setSpacing(16)
        live, live_layout = _card()
        live.setMinimumHeight(260)
        live_header = QHBoxLayout()
        live_header.addWidget(_label("LIVE  /  实时字幕", "eyebrow"))
        live_header.addStretch()
        self.state_badge = _label("待开始", "state")
        live_header.addWidget(self.state_badge)
        live_layout.addLayout(live_header)
        self.subtitle_label = _label("字幕将在这里出现", wrap=True)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.subtitle_label.setStyleSheet("color: #b9cbd3; font-size: 29px; font-weight: 600;")
        self.subtitle_label.setMinimumHeight(95)
        subtitle_policy = self.subtitle_label.sizePolicy()
        subtitle_policy.setVerticalPolicy(QSizePolicy.Policy.Expanding)
        self.subtitle_label.setSizePolicy(subtitle_policy)
        live_layout.addWidget(self.subtitle_label, 1)
        self.source_label = _label("选择音频来源，点击“开始同传”。", "muted", True)
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        live_layout.addWidget(self.source_label)
        live_layout.addSpacing(5)
        subtitle_options = QHBoxLayout()
        self.source_check = QCheckBox("显示原文")
        self.source_check.setChecked(True)
        self.source_check.toggled.connect(self._toggle_source)
        subtitle_options.addWidget(self.source_check)
        subtitle_options.addStretch()
        subtitle_options.addWidget(_label("字号", "muted"))
        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(22, 46)
        self.font_slider.setValue(30)
        self.font_slider.setFixedWidth(85)
        self.font_slider.valueChanged.connect(self._font_changed)
        subtitle_options.addWidget(self.font_slider)
        self.overlay_button = QPushButton("悬浮字幕 ↗")
        self.overlay_button.setObjectName("ghost")
        self.overlay_button.setCheckable(True)
        self.overlay_button.toggled.connect(self.overlay.setVisible)
        self.overlay.visibility_changed.connect(self._overlay_visibility)
        subtitle_options.addWidget(self.overlay_button)
        live_layout.addLayout(subtitle_options)
        output.addWidget(live, 1)

        history, history_layout = _card()
        history_header = QHBoxLayout()
        history_header.addWidget(_label("字幕记录", "section"))
        self.history_count = _label("0 条", "muted")
        history_header.addWidget(self.history_count)
        history_header.addStretch()
        self.clear_button = QPushButton("清空")
        self.clear_button.setObjectName("ghost")
        self.clear_button.clicked.connect(self._request_clear)
        self.export_button = QPushButton("导出字幕")
        self.export_button.setObjectName("ghost")
        self.export_button.clicked.connect(self._choose_export)
        self.export_button.setEnabled(False)
        history_header.addWidget(self.clear_button)
        history_header.addWidget(self.export_button)
        history_layout.addLayout(history_header)
        self.history_browser = QTextBrowser()
        self.history_browser.setOpenExternalLinks(False)
        self.history_browser.setMinimumHeight(120)
        self.history_browser.document().setDefaultFont(QFont("Microsoft YaHei UI", 10))
        history_layout.addWidget(self.history_browser, 1)
        self._render_history()
        output.addWidget(history, 1)
        body.addLayout(output, 1)
        page.addLayout(body, 1)

        footer = QHBoxLayout()
        self.status_label = _label("就绪后可开始；首次使用请先在设置中准备模型。", "muted")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer.addWidget(self.status_label, 1)
        self.diagnostic_badge = _label("")
        self.diagnostic_badge.setMinimumWidth(180)
        self.diagnostic_badge.setMaximumWidth(190)
        self.diagnostic_badge.hide()
        footer.addWidget(self.diagnostic_badge)
        self.metrics_label = _label("排队 0  ·  处理 —", "muted")
        footer.addWidget(self.metrics_label)
        page.addLayout(footer)
        self.compute_combo.currentIndexChanged.connect(lambda _: self._update_performance_warning())
        for control in (self.language_combo, self.model_combo, self.compute_combo,
                        self.asr_profile_combo, self.translation_threads_combo):
            control.currentIndexChanged.connect(self._preferences_changed)
        self.translate_check.toggled.connect(self._translation_changed)
        self._update_controls()

    def _build_settings_dialog(self) -> None:
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("本地同传 · 设置")
        self.settings_dialog.setMinimumWidth(440)
        self.settings_dialog.setStyleSheet(STYLE)
        self.settings_dialog.resize(480, min(780, self.screen().availableGeometry().height() - 80))
        outer = QVBoxLayout(self.settings_dialog)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(12)
        layout.addWidget(_label("识别与本地模型", "section"))
        self.settings_hint = _label("开始前指定课程语言；设置会自动保存，下次开始时生效。", "muted", True)
        layout.addWidget(self.settings_hint)
        layout.addWidget(_label("源语言", "muted"))
        self.language_combo = QComboBox()
        self.language_combo.addItem("请选择课程语言", None)
        self.language_combo.addItem("德语  Deutsch", "de")
        self.language_combo.addItem("英语  English", "en")
        self.language_combo.setToolTip("始终按指定语言识别，不进行自动语言检测。")
        layout.addWidget(self.language_combo)
        self.translate_check = QCheckBox("翻译为中文字幕")
        self.translate_check.setChecked(True)
        layout.addWidget(self.translate_check)
        layout.addWidget(_label("关闭后仅显示德语 / 英语原文，跳过翻译模型加载，识别结束即可显示；仍需等待断句。", "muted", True))
        layout.addWidget(_label("字幕节奏", "muted"))
        self.asr_profile_combo = QComboBox()
        self.asr_profile_combo.addItem("清晰收音（实时优先）", "adaptive")
        self.asr_profile_combo.addItem("较差收音（长上下文）", "coherent")
        self.asr_profile_combo.addItem("低延迟", "fast")
        self.asr_profile_combo.setCurrentIndex(self.asr_profile_combo.findData(DEFAULT_ASR_PROFILE))
        layout.addWidget(self.asr_profile_combo)
        layout.addWidget(_label("清晰收音：积累约 4.5 秒上下文后，遇 0.3 秒停顿提前处理；短句仍等 0.8 秒，最长 8 秒。较差收音：保留原 0.8 秒停顿 / 最长 8 秒的长上下文。低延迟：0.5 秒 / 5 秒。按环境手动切换，均需加识别和翻译耗时。", "muted", True))
        layout.addWidget(_label("识别模型", "muted"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Turbo · 推荐 RTX 4070", "large-v3-turbo")
        self.model_combo.addItem("Small · 节省资源", "small")
        self.model_combo.setToolTip("两个模型均支持德语和英语。模型越大，首次下载与加载时间越长。")
        self.model_combo.currentIndexChanged.connect(lambda _: self.model_changed.emit(self.selected_model()))
        layout.addWidget(self.model_combo)
        layout.addWidget(_label("识别计算设备", "muted"))
        self.compute_combo = QComboBox()
        self.compute_combo.addItem("NVIDIA GPU · INT8 / FP16", "cuda")
        self.compute_combo.addItem("CPU · INT8", "cpu")
        self.compute_combo.setToolTip("翻译使用 CPU；GPU 模式为语音识别保留显存。")
        layout.addWidget(self.compute_combo)
        layout.addWidget(_label("中文翻译 CPU 线程", "muted"))
        self.translation_threads_combo = QComboBox()
        for threads in TRANSLATION_THREAD_OPTIONS:
            self.translation_threads_combo.addItem(f"{threads} 线程", threads)
        self.translation_threads_combo.setCurrentIndex(
            self.translation_threads_combo.findData(DEFAULT_TRANSLATION_THREADS))
        self.translation_threads_combo.setToolTip("只控制中文翻译；不改变识别模型、识别候选数或译文质量。")
        layout.addWidget(self.translation_threads_combo)
        self.model_status = _label("正在检查本地模型…", "muted", True)
        self.model_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.model_status)
        self.prepare_button = QPushButton("下载 / 准备模型")
        self.prepare_button.setToolTip("首次准备需要网络；准备完成后可离线使用。")
        self.prepare_button.clicked.connect(lambda: self.prepare_requested.emit(self.configuration()))
        layout.addWidget(self.prepare_button)
        layout.addSpacing(6)
        layout.addWidget(_label("历史课程文件", "section"))
        self.cleanup_button = QPushButton("清理课程文件 …")
        self.cleanup_button.setToolTip("按次删除 exports/sessions/ 中的课程字幕归档；课程进行中不可清理。")
        self.cleanup_button.clicked.connect(self.cleanup_requested.emit)
        layout.addWidget(self.cleanup_button)
        layout.addWidget(_label("课程字幕自动保存到 exports/sessions/；正常同传不保存录音，诊断采样需单独手动开启。", "muted", True))
        layout.addSpacing(6)
        layout.addWidget(_label("诊断采样", "section"))
        layout.addWidget(_label("手动开启后，保存最多 5 分钟音频、识别原文与运行参数到本机，不会上传。暂停或结束课程时也会结束采样。", "muted", True))
        self.diagnostic_button = QPushButton("录制 5 分钟诊断素材")
        self.diagnostic_button.setToolTip("先开始同传并等待正在聆听，再手动开启；普通同传不会自动保存录音。")
        self.diagnostic_button.clicked.connect(self.diagnostic_requested.emit)
        layout.addWidget(self.diagnostic_button)
        self.diagnostic_stop_button = QPushButton("结束本次采样")
        self.diagnostic_stop_button.clicked.connect(self.diagnostic_stop_requested.emit)
        layout.addWidget(self.diagnostic_stop_button)
        self.diagnostic_status_label = _label("未开启 · 正常同传不会保存录音。", "muted", True)
        self.diagnostic_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.diagnostic_status_label)
        close = QPushButton("完成")
        close.clicked.connect(self.settings_dialog.close)
        layout.addWidget(close)

    def _show_settings(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def set_diagnostic_state(self, snapshot: dict[str, Any]) -> None:
        """Render recorder notifications without starting any capture in the UI."""
        self._diagnostic.update(snapshot)
        state = self._diagnostic.get("state", "idle")
        seconds = min(300, self._number(self._diagnostic.get("recorded_seconds")) or 0)
        path = str(self._diagnostic.get("path") or "")
        error = str(self._diagnostic.get("error") or "")
        if state == "recording":
            text = f"正在录制 {_clock(seconds)} / 05:00"
            color = "#edb1b7"
        elif state == "saving":
            text, color = f"正在保存 · 已采样 {_clock(seconds)}", "#edc78f"
        elif state == "completed":
            text, color = f"采样已完成 · {_clock(seconds)}", "#a7d9ce"
        elif state == "error":
            text, color = "诊断采样异常 · 同传仍可继续", "#edc78f"
        else:
            text, color = "未开启 · 正常同传不会保存录音。", "#91a1af"
        detail = text + (f"\n{error}" if error else "") + (f"\n文件夹：{path}" if path else "")
        self.diagnostic_status_label.setText(detail)
        self.diagnostic_status_label.setToolTip(detail)
        self.diagnostic_status_label.setStyleSheet(f"color: {color};")
        badge_text = (f"采样中 {_clock(seconds)} / 05:00" if state == "recording" else
                      "正在保存诊断素材" if state == "saving" else
                      "诊断采样已完成" if state == "completed" else "采样异常 · 详见设置")
        self.diagnostic_badge.setText(badge_text)
        self.diagnostic_badge.setToolTip(detail)
        self.diagnostic_badge.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.diagnostic_badge.setVisible(state in ("recording", "saving", "completed", "error"))
        self._update_controls()

    def open_cleanup(self, sessions: list[dict[str, Any]]) -> None:
        if self.cleanup_dialog is None:
            self.cleanup_dialog = SessionCleanupDialog(self)
            self.cleanup_dialog.deletion_requested.connect(self.delete_sessions_requested.emit)
        self.cleanup_dialog.set_sessions(sessions)
        self.cleanup_dialog.show()
        self.cleanup_dialog.raise_()
        self.cleanup_dialog.activateWindow()

    def refresh_cleanup(self, sessions: list[dict[str, Any]]) -> None:
        dialog = getattr(self, "cleanup_dialog", None)
        if dialog is not None and dialog.isVisible():
            dialog.set_sessions(sessions)

    def _build_dashboard(self, parent: QVBoxLayout) -> None:
        heading = QHBoxLayout()
        heading.addWidget(_label("运行监测", "section"))
        heading.addStretch()
        heading.addWidget(_label("每 2 秒更新", "muted"))
        parent.addLayout(heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        rows = QVBoxLayout(content)
        rows.setContentsMargins(0, 2, 6, 2)
        rows.setSpacing(7)
        self.performance_labels: dict[str, QLabel] = {}
        for key, name in (("cpu", "CPU"), ("ram", "内存"), ("process", "本软件内存"),
                          ("vram", "显存"), ("gpu", "GPU"), ("elapsed", "本次时长"),
                          ("queue", "待处理"), ("rtf", "处理速度"),
                          ("inference", "识别 / 翻译"), ("drops", "采集 / 跳过")):
            row = QHBoxLayout()
            row.addWidget(_label(name, "muted"))
            row.addStretch()
            value = _label("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.performance_labels[key] = value
            row.addWidget(value)
            rows.addLayout(row)
        self.performance_labels["rtf"].setToolTip("最近一段的 RTF = 识别与翻译耗时 ÷ 该段音频时长；小于 1 表示处理速度超过实时。它不包含断句等待时间，需结合积压观察。")
        self.performance_labels["inference"].setToolTip("最近完成一段的识别 / 翻译耗时；开启翻译时两者在翻译完成后一起显示，关闭时直接显示原文。不含采集、等断句、排队和界面绘制时间。")
        self.performance_labels["vram"].setToolTip("整张 NVIDIA 显卡的占用，包含其他程序；非本软件独占显存。")
        self.performance_labels["process"].setToolTip("本软件进程的物理内存工作集。")
        archive_row = QHBoxLayout()
        archive_row.addWidget(_label("字幕归档", "muted"))
        archive_row.addStretch()
        self.archive_status_label = _label("待开始")
        archive_row.addWidget(self.archive_status_label)
        rows.addLayout(archive_row)
        self.performance_warning = _label("正在读取本机资源…", wrap=True)
        self.performance_warning.setStyleSheet("color: #a7d9ce; font-size: 12px;")
        rows.addSpacing(5)
        rows.addWidget(self.performance_warning)
        rows.addWidget(_label("持续更新当前值，不累计保存音频。", "muted", True))
        rows.addStretch(1)
        scroll.setWidget(content)
        parent.addWidget(scroll, 1)

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @staticmethod
    def _gib(value: float) -> str:
        return f"{value / 1024 ** 3:.1f} GB"

    def set_performance(self, snapshot: dict[str, Any]) -> None:
        self._performance = dict(snapshot)
        value = self._number
        cpu = value(snapshot.get("system_cpu_percent"))
        self.performance_labels["cpu"].setText(f"{cpu:.0f}%" if cpu is not None else "暂不可用")
        total, free = value(snapshot.get("ram_total_bytes")), value(snapshot.get("ram_available_bytes"))
        self.performance_labels["ram"].setText(
            f"可用 {self._gib(free)} / {self._gib(total)}" if total and free is not None else "暂不可用")
        rss = value(snapshot.get("process_rss_bytes"))
        self.performance_labels["process"].setText(self._gib(rss) if rss is not None else "暂不可用")
        total, used = value(snapshot.get("gpu_total_bytes")), value(snapshot.get("gpu_used_bytes"))
        self.performance_labels["vram"].setText(
            f"{self._gib(used)} / {self._gib(total)}" if total and used is not None else "暂不可用")
        gpu, temperature = value(snapshot.get("gpu_util_percent")), value(snapshot.get("gpu_temperature_c"))
        gpu_text = f"{gpu:.0f}%" if gpu is not None else "暂不可用"
        if temperature is not None:
            gpu_text += f" · {temperature:.0f}°C"
        self.performance_labels["gpu"].setText(gpu_text)
        self._update_performance_warning()

    def begin_session(self) -> None:
        self._session_active = True
        self._slow_updates = 0
        self._session_metrics = {}
        self.set_session_metrics({"elapsed_seconds": 0, "queue_depth": 0, "queue_audio_seconds": 0,
                                  "dropped_segments": 0, "driver_overflows": 0, "queue_dropped_seconds": 0})

    def end_session(self) -> None:
        self._session_active = False
        self._slow_updates = 0
        if self._archive_path and not self._archive_error:
            self.archive_status_label.setText("已保存")
        self._update_performance_warning()

    def set_archive_status(self, path: str | None, error: str = "") -> None:
        self._archive_error = str(error)
        self._archive_path = path
        self.archive_status_label.setText("保存异常" if error else "自动保存中" if path else "待开始")
        self.archive_status_label.setToolTip(error or path or "开始同传后，每条字幕自动写入本地课程文件。")
        self.archive_status_label.setStyleSheet("color: #edc78f;" if error else "")
        self._update_performance_warning()

    def set_session_metrics(self, snapshot: dict[str, Any]) -> None:
        self._session_metrics.update(snapshot)
        data, value = self._session_metrics, self._number
        elapsed = value(data.get("elapsed_seconds"))
        self.performance_labels["elapsed"].setText(_clock(elapsed) if elapsed is not None else "—")
        self.performance_labels["elapsed"].setToolTip(f"本次已生成 {int(value(data.get('subtitle_count')) or 0)} 条字幕；无 90 分钟停止限制。")
        depth, seconds = value(data.get("queue_depth")), value(data.get("queue_audio_seconds"))
        self.performance_labels["queue"].setText(
            f"{int(depth)} 段 · {seconds:.1f} 秒" if depth is not None and seconds is not None else "—")
        rtf = value(data.get("realtime_factor"))
        self.performance_labels["rtf"].setText(f"RTF {rtf:.2f} · {'计算可跟上' if rtf < 1 else '偏慢'}" if rtf is not None else "等待语音")
        asr, translation = value(data.get("asr_seconds")), value(data.get("translation_seconds"))
        self.performance_labels["inference"].setText(
            f"{asr:.2f} 秒 / 关闭" if asr is not None and data.get("translate_enabled") is False else
            f"{asr:.2f} / {translation:.2f} 秒" if asr is not None and translation is not None else "等待语音")
        queue_wait = value(data.get("queue_wait_seconds"))
        endpoint = value(data.get("endpoint_wait_seconds"))
        chunk_duration = value(data.get("audio_seconds"))
        self.performance_labels["queue"].setToolTip(
            "待处理队列容量最多 3 段。\n" +
            (f"最近一段音频 {chunk_duration:.2f} 秒；" if chunk_duration is not None else "") +
            (f"断句后排队 {queue_wait:.2f} 秒。\n" if queue_wait is not None else "尚无排队耗时。\n") +
            (f"保留音频末尾至断句的等待 {endpoint:.2f} 秒；" if endpoint is not None else "") +
            "不等于从说话到字幕显示的总延迟。")
        if self._session_active and rtf is not None and rtf >= 1 and depth:
            self._slow_updates += 1
        else:
            self._slow_updates = 0
        driver, dropped = value(data.get("driver_overflows")), value(data.get("dropped_segments"))
        skipped = value(data.get("queue_dropped_seconds"))
        self.performance_labels["drops"].setText(
            f"{int(driver or 0)} 次 / {int(dropped or 0)} 段")
        self.performance_labels["drops"].setToolTip(
            f"设备溢出 {int(driver or 0)} 次；采集积压跳过 {skipped or 0:.2f} 秒；识别积压跳过 {int(dropped or 0)} 段。")
        self._update_performance_warning()

    def _update_performance_warning(self) -> None:
        data, value = self._performance, self._number
        warnings = []
        if self._archive_error:
            warnings.append("字幕自动保存异常，请及时手动导出。")
        total, free = value(data.get("ram_total_bytes")), value(data.get("ram_available_bytes"))
        if total and free is not None and (free < 1024 ** 3 or free / total < .1):
            warnings.append("可用内存偏低，请留意其他程序占用。")
        total, used = value(data.get("gpu_total_bytes")), value(data.get("gpu_used_bytes"))
        if self.compute_combo.currentData() == "cuda" and total and used is not None and total - used < 512 * 1024 ** 2:
            warnings.append("可用显存不足 512 MB，存在推理失败风险。")
        if self._performance and (
            value(data.get("ram_total_bytes")) is None
            or value(data.get("ram_available_bytes")) is None
            or (self.compute_combo.currentData() == "cuda" and (total is None or used is None))
        ):
            warnings.append("部分资源监测暂不可用，无法完整判断余量。")
        if self._session_active and self._slow_updates >= 3:
            warnings.append("处理速度持续偏慢且有积压；可在停课后调整设置。")
        if self._session_active and any((value(self._session_metrics.get(key)) or 0) > 0
                                        for key in ("driver_overflows", "queue_dropped_seconds", "dropped_segments")):
            warnings.append("本次发生过丢帧或跳段，请留意字幕完整性。")
        if warnings:
            text, color = "\n".join(warnings), "#edc78f"
        elif not self._performance:
            text, color = "正在读取本机资源…", "#a7d9ce"
        else:
            text, color = "当前未触发资源 / 积压预警。", "#a7d9ce"
        self.performance_warning.setText(text)
        self.performance_warning.setStyleSheet(f"color: {color}; font-size: 12px;")

    def selected_model(self) -> str:
        return str(self.model_combo.currentData())

    def configuration(self) -> dict[str, Any]:
        return {"device_id": self.device_combo.currentData(),
                "language": self.language_combo.currentData(),
                "model_size": self.selected_model(),
                "device": str(self.compute_combo.currentData()),
                "asr_profile": str(self.asr_profile_combo.currentData()),
                "translate_enabled": self.translate_check.isChecked(),
                "translation_threads": int(self.translation_threads_combo.currentData())}

    def apply_preferences(self, preferences: dict[str, Any]) -> None:
        for key, control in (("language", self.language_combo), ("model_size", self.model_combo),
                             ("device", self.compute_combo),
                             ("asr_profile", self.asr_profile_combo),
                             ("translation_threads", self.translation_threads_combo)):
            index = control.findData(preferences.get(key))
            if key == "asr_profile" and index < 0:
                index = control.findData(DEFAULT_ASR_PROFILE)
            if index >= 0:
                blocked = control.blockSignals(True)
                control.setCurrentIndex(index)
                control.blockSignals(blocked)
        enabled = preferences.get("translate_enabled", True)
        blocked = self.translate_check.blockSignals(True)
        self.translate_check.setChecked(enabled if type(enabled) is bool else True)
        self.translate_check.blockSignals(blocked)
        self._update_controls()
        self._update_performance_warning()

    def _preferences_changed(self, *_: Any) -> None:
        self._update_controls()
        self.preferences_changed.emit(self.configuration())

    def _translation_changed(self, *_: Any) -> None:
        self._update_controls()
        self.model_changed.emit(self.selected_model())
        self.preferences_changed.emit(self.configuration())

    def set_devices(self, devices: list[dict[str, Any]], default_id: Any = None) -> None:
        previous = self.device_combo.currentData()
        self.device_combo.clear()
        for item in devices:
            prefix = "系统回环 · " if item.get("is_loopback") else "麦克风 · "
            self.device_combo.addItem(prefix + str(item["name"]), item["id"])
            self.device_combo.setItemData(self.device_combo.count() - 1,
                                          prefix + str(item["name"]), Qt.ItemDataRole.ToolTipRole)
        selected = previous if self.device_combo.findData(previous) >= 0 else default_id
        index = self.device_combo.findData(selected)
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
        if not devices:
            self.set_status("未找到音频设备，请连接设备后刷新。")
        self._update_controls()

    def set_model_status(self, text: str, ready: bool) -> None:
        self._ready = ready
        self.model_status.setText(text)
        self.start_button.setToolTip("" if ready else "请打开设置，下载 / 准备所选本地模型。")
        self._update_controls()

    def set_running(self, running: bool) -> None:
        self._running = running
        self._paused = False
        self.state_badge.setText("正在聆听" if running else "已停止")
        if not running:
            self.set_level(0)
        self._update_controls()

    def set_paused(self, paused: bool) -> None:
        """Acknowledge a completed pause/resume without resetting this class."""
        self._paused = paused
        self._running = True
        self._busy = False
        self.state_badge.setText("已暂停" if paused else "正在聆听")
        if paused:
            self.set_level(0)
            self._slow_updates = 0
            self.set_status("已暂停 · 模型与本课字幕保留，点击“继续”恢复采集；模型仍占用内存与显存。")
        self._update_controls()
        self._update_performance_warning()

    def set_pausing(self) -> None:
        self.set_busy(True, "正在暂停，处理最后几段语音 …")

    def set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        if message:
            self.set_status(message)
        if busy:
            self._busy_caption = ("暂停中…" if "暂停" in message else
                                  "继续中…" if "继续" in message else
                                  "停止中…" if "停止" in message else
                                  "结束中…" if "结束" in message else
                                  "取消中…" if "取消" in message else "准备中…")
            self.state_badge.setText(self._busy_caption)
        elif not self._running:
            self.state_badge.setText("待开始")
        elif self._paused:
            self.state_badge.setText("已暂停")
        else:
            self.state_badge.setText("正在聆听")
        self._update_controls()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setToolTip(text)

    def set_level(self, level: float) -> None:
        self.level_bar.setValue(round(max(0.0, min(1.0, float(level))) * 1000))

    def set_metrics(self, queue_size: int, latency: float) -> None:
        self.metrics_label.setText(f"排队 {max(0, queue_size)}  ·  处理 {max(0, latency):.1f} 秒")

    def add_subtitle(self, entry: dict[str, Any]) -> None:
        self._history.append(dict(entry))
        self._subtitle_total += 1
        text = str(entry.get("text", ""))
        source = str(entry.get("source", ""))
        secondary_source = source if source.strip() != text.strip() else ""
        self._latest_source = secondary_source
        self.subtitle_label.setText(text)
        self.subtitle_label.setStyleSheet(f"color: #effbf7; font-size: {self.font_slider.value()}px; font-weight: 600;")
        self.source_label.setText(secondary_source)
        self.overlay.set_subtitle(text, secondary_source)
        self.history_count.setText(f"{self._subtitle_total} 条" + (" · 显示最近 200 条" if self._subtitle_total > 200 else ""))
        self.export_button.setEnabled(True)
        self._render_history()

    def clear_subtitles(self) -> None:
        self._history.clear()
        self._subtitle_total = 0
        self._latest_source = ""
        self.subtitle_label.setText("字幕将在这里出现")
        self.source_label.setText("等待下一段语音…" if self._running else "选择音频来源，点击“开始同传”。")
        self.overlay.set_subtitle("", "")
        self.history_count.setText("0 条")
        self.export_button.setEnabled(False)
        self._render_history()

    def show_error(self, text: str) -> None:
        self.set_status(text)
        QMessageBox.warning(self, "本地同传", text)

    def _update_controls(self) -> None:
        idle = not self._busy and not self._running
        language_ready = self.language_combo.currentData() in ("de", "en")
        self.start_button.setEnabled(idle and self._ready and self.device_combo.count() > 0 and language_ready)
        self.start_button.setToolTip("请先打开设置，指定本次课程语言。" if not language_ready else
                                    "请打开设置，下载 / 准备所选本地模型。" if not self._ready else "")
        self.start_button.setText("开始同传" if not self._busy else self._busy_caption)
        self.stop_button.setEnabled(self._running or self._busy)
        self.pause_button.setEnabled(self._running and not self._busy)
        self.pause_button.setText("继续" if self._paused else "暂停")
        self.pause_button.setToolTip("继续本课采集，复用已加载模型和字幕记录。" if self._paused else
                                     "停止采集并处理尾句，保留本课字幕和已加载模型。")
        self.course_hint.setText("已暂停：模型仍占用内存与显存。" if self._paused else
                                 "暂停保留模型；结束释放资源。")
        for control in (self.device_combo, self.language_combo, self.model_combo,
                        self.compute_combo, self.asr_profile_combo, self.translation_threads_combo,
                        self.translate_check, self.refresh_button, self.prepare_button, self.cleanup_button):
            control.setEnabled(idle)
        translate_enabled = self.translate_check.isChecked()
        self.translation_threads_combo.setEnabled(idle and translate_enabled)
        self.source_check.setEnabled(translate_enabled)
        self.source_check.setToolTip("同时显示识别原文。" if translate_enabled else "仅原文模式中，原文已作为主字幕显示。")
        self.output_hint.setText("德语 / 英语语音  →  简体中文字幕" if translate_enabled else
                                 "德语 / 英语语音  →  原文字幕（不翻译）")
        self.privacy_badge.setText("音频与翻译均在本机处理" if translate_enabled else "语音识别在本机处理")
        self.setWindowTitle("本地同传 · 德语 / 英语 → 中文字幕" if translate_enabled else
                            "本地同传 · 德语 / 英语 → 原文字幕")
        self.settings_hint.setText("当前课程尚未结束，请先“结束并释放”，再修改设置。" if not idle
                                   else "开始前指定课程语言；设置会自动保存，下次开始时生效。")
        diagnostic_state = self._diagnostic.get("state", "idle")
        self.diagnostic_button.setEnabled(self._running and not self._busy and not self._paused
                                          and diagnostic_state not in ("recording", "saving"))
        self.diagnostic_stop_button.setEnabled(diagnostic_state == "recording")

    def _request_pause_or_resume(self) -> None:
        if self._paused:
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()

    def _toggle_source(self, visible: bool) -> None:
        self.source_label.setVisible(visible)
        self.overlay.set_source_visible(visible)
        self._render_history()

    def _font_changed(self, size: int) -> None:
        self.subtitle_label.setStyleSheet(f"color: #effbf7; font-size: {size}px; font-weight: 600;")
        self.overlay.set_font_size(size)

    def _overlay_visibility(self, visible: bool) -> None:
        self.overlay_button.blockSignals(True)
        self.overlay_button.setChecked(visible)
        self.overlay_button.blockSignals(False)

    def _render_history(self) -> None:
        scrollbar = self.history_browser.verticalScrollBar()
        at_bottom = scrollbar.maximum() - scrollbar.value() < 30
        previous = scrollbar.value()
        if not self._history:
            self.history_browser.setHtml('<p style="color:#7f929f; margin-top:30px;">开始后，字幕会按时间记录在这里。<br/><br/>可导出 SRT 字幕或 TXT 文本。</p>')
            return
        blocks = []
        for item in self._history:
            stamp = _clock(float(item.get("start", 0)))
            language = {"de": "DE", "en": "EN"}.get(str(item.get("language", "")), "")
            text = html.escape(str(item.get("text", ""))).replace("\n", "<br/>")
            source = html.escape(str(item.get("source", ""))).replace("\n", "<br/>")
            original = f'<p style="margin:4px 0 0; color:#8fa5b4; font-size:12px;">{source}</p>' if self.source_check.isChecked() and source and source.strip() != text.strip() else ""
            blocks.append(f'<p style="color:#6f958f; font-size:11px; margin:12px 0 5px;">{stamp} &nbsp; {language}</p><p style="font-size:15px; margin:0; color:#e0ede8;">{text}</p>{original}')
        self.history_browser.setHtml("".join(blocks))
        scrollbar.setValue(scrollbar.maximum() if at_bottom else previous)

    def _request_clear(self) -> None:
        self.clear_subtitles()
        self.clear_requested.emit()

    def _choose_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出字幕", "同传字幕.srt", "SRT 字幕 (*.srt);;纯文本 (*.txt)")
        if path:
            if not path.lower().endswith((".srt", ".txt")):
                path += ".srt"
            self.export_requested.emit(path)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.overlay.close()
        self.settings_dialog.close()
        if self.cleanup_dialog is not None:
            self.cleanup_dialog.close()
        super().closeEvent(event)
