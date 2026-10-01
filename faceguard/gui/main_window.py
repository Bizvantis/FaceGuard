"""Main window: sidebar navigation, page stack, toasts, tray icon, and routing of worker signals."""

from __future__ import annotations

import logging

from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QPushButton, QStackedWidget,
    QSystemTrayIcon, QVBoxLayout, QWidget,
)

from faceguard import APP_NAME, __version__
from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext
from faceguard.gui.pages.enroll import EnrollPage
from faceguard.gui.pages.guard import GuardPage
from faceguard.gui.pages.intruders import IntrudersPage
from faceguard.gui.pages.lab import LabPage
from faceguard.gui.pages.live import LivePage
from faceguard.gui.pages.people import PeoplePage
from faceguard.gui.pages.settings import SettingsPage
from faceguard.gui.pages.verify import VerifyPage
from faceguard.gui.widgets import Pill, label
from faceguard.security import winsys
from faceguard.security.guard import GuardState

log = logging.getLogger(__name__)

TOAST_COLORS = {"good": theme.GOOD, "warn": theme.WARNING, "error": theme.CRITICAL, "info": theme.ACCENT}
GUARD_PILL = {
    GuardState.PRESENT: theme.GOOD, GuardState.WATCHING: theme.WARNING, GuardState.INTRUDER: theme.CRITICAL,
    GuardState.LOCKED: theme.ACCENT, GuardState.NO_OWNER: theme.WARNING, GuardState.NO_CAMERA: theme.SERIOUS,
}


class Toast(QLabel):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, level: str, text: str) -> None:
        color = TOAST_COLORS.get(level, theme.ACCENT)
        self.setStyleSheet(f"background: {theme.RAISED}; border: 1px solid {color}; border-radius: 10px; "
                           f"padding: 10px 16px; color: {theme.INK};")
        self.setText(text)
        width = min(560, self.parentWidget().width() - 40)
        self.setFixedWidth(width)
        self.adjustSize()
        self.move((self.parentWidget().width() - self.width()) // 2, self.parentWidget().height() - self.height() - 24)
        self.show()
        self.raise_()
        self._timer.start(4000 if level != "error" else 7000)


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.worker = ctx.worker
        self._quitting = False
        self._tray_hint_shown = False
        self._new_intruders = 0
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(icons.app_icon())
        self.resize(1320, 840)
        self.setMinimumSize(1060, 680)

        root = QWidget()
        root.setObjectName("Root")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        self.stack.setObjectName("Pages")
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.toast = Toast(root)

        self.pages = [LivePage(ctx), EnrollPage(ctx), PeoplePage(ctx), VerifyPage(ctx), GuardPage(ctx),
                      IntrudersPage(ctx), LabPage(ctx), SettingsPage(ctx)]
        self.by_key = {p.key: p for p in self.pages}
        for i, page in enumerate(self.pages):
            self.stack.addWidget(page)
            btn = self._nav_button(page.glyph, page.title if page.key != "intruders" else "Intruders")
            btn.clicked.connect(lambda _=False, k=page.key: self.go(k))
            self.nav_group.addButton(btn, i)
            self.nav_layout.addWidget(btn)
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            shortcut.activated.connect(lambda k=page.key: self.go(k))
        self.nav_layout.addStretch(1)
        self._current = None

        ctx.navigate.connect(self.go)
        ctx.enroll_person.connect(self._enroll_person)
        ctx.toast.connect(self.show_toast)
        ctx.settings_changed.connect(self.by_key["settings"].refresh_from_settings)
        w = self.worker
        w.frame_ready.connect(self._on_frame)
        w.camera_state.connect(self._on_camera)
        w.guard_status.connect(self._on_guard_status)
        w.notify.connect(self._on_notify)
        w.snapshot_saved.connect(self._on_snapshot)
        w.task_finished.connect(self._on_task_finished)
        w.learned.connect(lambda name: self.by_key["guard"].add_event(f"Adaptive learning: updated {name}'s faceprint"))
        w.fatal.connect(self._on_fatal)

        self.tray = self._build_tray()
        ctx.settings_changed.connect(self.sync_tray_guard)
        self.go("live" if any(p.samples for p in ctx.db.persons) else "enroll")

    # ------------------------------------------------------------ construction
    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Sidebar")
        side.setFixedWidth(236)
        v = QVBoxLayout(side)
        v.setContentsMargins(14, 18, 14, 16)
        v.setSpacing(4)
        brand = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(icons.logo_pixmap(38))
        brand.addWidget(logo)
        names = QVBoxLayout()
        names.setSpacing(0)
        n = QLabel(APP_NAME)
        n.setObjectName("AppName")
        t = QLabel("AI face recognition & security")
        t.setObjectName("AppTag")
        names.addWidget(n)
        names.addWidget(t)
        brand.addLayout(names, 1)
        v.addLayout(brand)
        v.addSpacing(18)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_layout = QVBoxLayout()
        self.nav_layout.setSpacing(2)
        v.addLayout(self.nav_layout, 1)
        v.addWidget(label("Guard", "small"))
        self.guard_pill = Pill("Off", theme.MUTED)
        v.addWidget(self.guard_pill)
        v.addSpacing(6)
        v.addWidget(label("Camera", "small"))
        self.cam_pill = Pill("Off", theme.MUTED)
        v.addWidget(self.cam_pill)
        v.addSpacing(10)
        v.addWidget(label(f"v{__version__}  |  runs 100% on this PC", "small"))
        return side

    def _nav_button(self, glyph: str, text: str) -> QPushButton:
        b = QPushButton("  " + text)
        b.setObjectName("Nav")
        b.setCheckable(True)
        b.setIcon(icons.glyph_icon(glyph, theme.INK2, 18))
        b.setIconSize(QSize(18, 18))
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def _build_tray(self) -> QSystemTrayIcon | None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return None
        tray = QSystemTrayIcon(icons.app_icon(), self)
        tray.setToolTip(APP_NAME)
        menu = QMenu()
        menu.addAction("Open FaceGuard", self.show_normal)
        menu.addSeparator()
        self.act_guard = QAction("Guard on", menu)
        self.act_guard.setCheckable(True)
        self.act_guard.setChecked(self.ctx.settings.guard_enabled)
        self.act_guard.toggled.connect(self._tray_guard)
        menu.addAction(self.act_guard)
        menu.addAction("Pause for 15 minutes", lambda: self.worker.guard_command("pause", 15 * 60))
        menu.addAction("Pause for 1 hour", lambda: self.worker.guard_command("pause", 60 * 60))
        menu.addAction("Resume", lambda: self.worker.guard_command("resume"))
        menu.addSeparator()
        menu.addAction("Lock PC now", lambda: winsys.lock_workstation())
        menu.addSeparator()
        menu.addAction("Quit FaceGuard", self.quit_app)
        tray.setContextMenu(menu)
        self._tray_menu = menu
        tray.activated.connect(self._tray_activated)
        tray.show()
        return tray

    # ------------------------------------------------------------ navigation
    def go(self, key: str) -> None:
        page = self.by_key.get(key)
        if page is None:
            return
        if self._current is not None and self._current is not page:
            self._current.on_hide()
        self._current = page
        idx = self.pages.index(page)
        self.stack.setCurrentIndex(idx)
        self.nav_group.button(idx).setChecked(True)
        if key == "intruders" and self._new_intruders:
            self._new_intruders = 0
            self.nav_group.button(idx).setText("  Intruders")
        page.on_show()
        self._update_camera_need()

    def _enroll_person(self, name: str) -> None:
        page: EnrollPage = self.by_key["enroll"]
        self.go("enroll")
        if page.panels.currentIndex() == 0:
            page.name.setText(name)
            page.name.setFocus()

    def _window_active(self) -> bool:
        return self.isVisible() and not self.isMinimized()

    def _update_camera_need(self) -> None:
        active = self._window_active() and self._current is not None and self._current.wants_camera()
        self.worker.set_view_active(active)

    def changeEvent(self, e) -> None:
        if e.type() == QEvent.Type.WindowStateChange:
            self._update_camera_need()
        super().changeEvent(e)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        self._update_camera_need()

    def hideEvent(self, e) -> None:
        super().hideEvent(e)
        if self._current is not None:
            self._current.on_hide()
        self._update_camera_need()

    def show_normal(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if self._current is not None:
            self._current.on_show()
        self._update_camera_need()

    def closeEvent(self, e) -> None:
        if not self._quitting and self.tray is not None and self.ctx.settings.minimize_to_tray:
            e.ignore()
            self.hide()
            if not self._tray_hint_shown:
                self._tray_hint_shown = True
                msg = ("The guard keeps protecting your PC from the tray." if self.ctx.settings.guard_enabled
                       else "FaceGuard is still running in the tray. Right-click the icon to quit.")
                self.tray.showMessage(APP_NAME, msg, icons.app_icon(), 4000)
            return
        self.quit_app()
        e.accept()

    def quit_app(self) -> None:
        self._quitting = True
        self.worker.stop()
        if self.tray is not None:
            self.tray.hide()
        QApplication.quit()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.toast.isVisible():
            self.toast.move((self.centralWidget().width() - self.toast.width()) // 2,
                            self.centralWidget().height() - self.toast.height() - 24)

    # ------------------------------------------------------------ worker signals
    def _on_frame(self, packet) -> None:
        try:
            if self._current is not None and self._current.wants_camera():
                self._current.on_frame(packet)
        finally:
            self.worker.frame_consumed()

    def _on_camera(self, state: str, message: str) -> None:
        colors = {"on": theme.GOOD, "starting": theme.ACCENT, "error": theme.CRITICAL}
        texts = {"on": "On", "starting": "Starting", "error": "Unavailable", "off": "Off"}
        self.cam_pill.set_state(texts.get(state, state), colors.get(state, theme.MUTED))
        self.by_key["live"].set_camera_state(state, message)
        for key in ("enroll", "verify"):
            if state in ("starting", "error"):
                self.by_key[key].video.clear("Starting camera..." if state == "starting" else "Camera unavailable", message)
        if state == "error":
            log.warning("camera: %s", message)

    def _on_guard_status(self, status) -> None:
        self.ctx.guard_status = status
        self.by_key["guard"].update_status(status)
        on = self.ctx.settings.guard_enabled
        text = status.text if on else "Off"
        if len(text) > 26:
            text = text[:25] + "..."
        self.guard_pill.set_state(text, GUARD_PILL.get(status.state, theme.MUTED) if on else theme.MUTED)
        if self.tray is not None:
            self.tray.setToolTip(f"{APP_NAME} - {status.text if on else 'guard off'}")
            self.tray.setIcon(icons.app_icon(alert=status.state == GuardState.INTRUDER))

    def _on_notify(self, kind: str, title: str, message: str) -> None:
        self.by_key["guard"].add_event(f"{title}: {message}")
        alarming = kind in ("intruder", "shoulder", "absence", "locked")
        if self.tray is not None:
            icon = QSystemTrayIcon.MessageIcon.Warning if alarming else QSystemTrayIcon.MessageIcon.Information
            self.tray.showMessage(title, message, icon, 6000)
        if self._window_active():
            self.show_toast("warn" if alarming else "info", f"{title} - {message}")

    def _on_snapshot(self, event) -> None:
        self.ctx.intruders_changed.emit()
        if event.reason in ("intruder", "shoulder", "locked"):
            self._new_intruders += 1
            btn = self.nav_group.button(self.pages.index(self.by_key["intruders"]))
            btn.setText(f"  Intruders  ({self._new_intruders} new)")

    def _on_task_finished(self, kind: str, task, outcome) -> None:
        page = self.by_key.get(kind)
        if page is not None:
            page.task_finished(task, outcome)

    def _on_fatal(self, message: str) -> None:
        QMessageBox.critical(self, APP_NAME, message)

    # ------------------------------------------------------------ tray
    def _tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_normal()

    def _tray_guard(self, on: bool) -> None:
        owner = self.ctx.db.owner()
        if on and (owner is None or not owner.samples):
            self.act_guard.blockSignals(True)
            self.act_guard.setChecked(False)
            self.act_guard.blockSignals(False)
            self.show_normal()
            self.go("enroll")
            self.show_toast("warn", "Enroll your face first so the guard knows who you are")
            return
        self.by_key["guard"].set_enabled(on)

    def sync_tray_guard(self) -> None:
        if self.tray is not None:
            self.act_guard.blockSignals(True)
            self.act_guard.setChecked(self.ctx.settings.guard_enabled)
            self.act_guard.blockSignals(False)

    def show_toast(self, level: str, text: str) -> None:
        self.toast.show_message(level, text)
