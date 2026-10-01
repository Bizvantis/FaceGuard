from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QListWidget, QMessageBox, QVBoxLayout, QWidget

from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext
from faceguard.gui.pages.base import Page, header, page_layout, scrolling
from faceguard.gui.widgets import SettingRow, Toggle, button, card, glyph_label, label, rounded_pixmap
from faceguard.recognition.database import decode_thumbnail
from faceguard.security import winsys
from faceguard.security.guard import GuardState

STATE_STYLE = {
    GuardState.PRESENT: (icons.SHIELD, theme.GOOD),
    GuardState.WATCHING: (icons.EYE, theme.WARNING),
    GuardState.INTRUDER: (icons.WARNING, theme.CRITICAL),
    GuardState.LOCKED: (icons.LOCK, theme.ACCENT),
    GuardState.PAUSED: (icons.PAUSE, theme.MUTED),
    GuardState.NO_OWNER: (icons.WARNING, theme.WARNING),
    GuardState.NO_CAMERA: (icons.CAMERA, theme.SERIOUS),
    GuardState.STOPPED: (icons.SHIELD, theme.MUTED),
}

TIMEOUTS = [(15, "15 seconds"), (30, "30 seconds"), (60, "1 minute"), (120, "2 minutes"), (300, "5 minutes")]
CONFIRMS = [(1.5, "1.5 seconds (fast)"), (2.5, "2.5 seconds"), (4.0, "4 seconds (fewer false alarms)")]
GUARD_FPS = [(2.0, "2 per second (lowest CPU)"), (4.0, "4 per second"), (8.0, "8 per second (fastest reaction)")]
LOCKED_WATCH = [(0, "Off"), (10, "For 10 minutes"), (30, "For 30 minutes"), (120, "For 2 hours"), (7 * 24 * 60, "Until I unlock")]


def combo(options: list[tuple[object, str]], current) -> QComboBox:
    c = QComboBox()
    for value, text in options:
        c.addItem(text, value)
    idx = min(range(len(options)), key=lambda i: abs(float(options[i][0]) - float(current)))
    c.setCurrentIndex(idx)
    c.setMinimumWidth(190)
    return c


class GuardPage(Page):
    key = "guard"
    title = "Guard"
    glyph = icons.SHIELD

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        s = ctx.settings
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        content.setObjectName("Page")
        lay = page_layout(content)
        lay.addWidget(header("Presence guard", "Locks your PC when you walk away and catches anyone else who sits down at it."))

        # --- status card -------------------------------------------------
        status, sl = card(margins=20)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.state_icon = glyph_label(icons.SHIELD, 30, theme.MUTED)
        self.state_icon.setFixedWidth(52)
        row.addWidget(self.state_icon)
        col = QVBoxLayout()
        col.setSpacing(2)
        self.state_text = label("Guard is off", "h2")
        self.state_detail = label("Turn it on to start protecting this PC.", "muted", wrap=True)
        col.addWidget(self.state_text)
        col.addWidget(self.state_detail)
        row.addLayout(col, 1)
        self.toggle = Toggle(s.guard_enabled)
        self.toggle.toggled.connect(self._toggle_guard)
        row.addWidget(label("Guard"), 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        sl.addLayout(row)
        btns = QHBoxLayout()
        self.pause15 = button("Pause 15 min", None, icons.PAUSE)
        self.pause60 = button("Pause 1 hour", None, icons.PAUSE)
        self.resume = button("Resume", "primary", icons.PLAY)
        lock_now = button("Lock PC now", "ghost", icons.LOCK, "Same as pressing Win+L")
        self.pause15.clicked.connect(lambda: self._pause(15 * 60))
        self.pause60.clicked.connect(lambda: self._pause(60 * 60))
        self.resume.clicked.connect(self._resume)
        lock_now.clicked.connect(lambda: winsys.lock_workstation())
        for b in (self.pause15, self.pause60, self.resume):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(lock_now)
        sl.addLayout(btns)
        lay.addWidget(status)

        # --- owner --------------------------------------------------------
        owner_card, ol = card("Protected owner")
        orow = QHBoxLayout()
        self.owner_pic = QLabel()
        self.owner_text = label("", "secondary", wrap=True)
        orow.addWidget(self.owner_pic)
        orow.addWidget(self.owner_text, 1)
        ol.addLayout(orow)
        lay.addWidget(owner_card)

        # --- rules -------------------------------------------------------------
        rules, rl = card("Rules")
        self.t_absence = Toggle(s.guard_lock_on_absence)
        rl.addWidget(SettingRow("Lock when I walk away", "No recognised face in front of the PC", self.t_absence))
        self.c_timeout = combo(TIMEOUTS, s.guard_absence_timeout)
        rl.addWidget(SettingRow("Lock after", "How long you can be away before the PC locks", self.c_timeout))
        self.t_activity = Toggle(s.guard_respect_activity)
        rl.addWidget(SettingRow("Stay unlocked while I'm typing", "Keyboard or mouse use counts as being present "
                                "(e.g. when you look away from the camera)", self.t_activity))
        self.t_intruder = Toggle(s.guard_lock_on_intruder)
        rl.addWidget(SettingRow("Lock when a stranger sits down", "An unrecognised face at the screen while you're away", self.t_intruder))
        self.c_confirm = combo(CONFIRMS, s.guard_intruder_seconds)
        rl.addWidget(SettingRow("Stranger must be seen for", "Longer = fewer false alarms", self.c_confirm))
        self.t_photos = Toggle(s.guard_save_intruders)
        rl.addWidget(SettingRow("Take a photo of intruders", "Saved to the Intruders page on this PC only", self.t_photos))
        self.c_locked = combo(LOCKED_WATCH, s.guard_locked_watch_minutes)
        rl.addWidget(SettingRow("Watch my PC while it's locked", "Photograph anyone unrecognised who sits down at your locked "
                                "PC (the camera stays on this long after it locks)", self.c_locked))
        self.t_shoulder = Toggle(s.guard_shoulder_alert)
        rl.addWidget(SettingRow("Warn me about shoulder surfers", "Someone unrecognised watching your screen", self.t_shoulder))
        lay.addWidget(rules)

        bg, bl = card("Background")
        self.t_tray = Toggle(s.minimize_to_tray)
        bl.addWidget(SettingRow("Keep protecting when the window is closed", "FaceGuard stays in the system tray", self.t_tray))
        self.t_startup = Toggle(winsys.get_run_at_startup() is not None)
        self.t_startup.setEnabled(winsys.IS_WINDOWS)
        bl.addWidget(SettingRow("Start with Windows", "Launch in the tray with the guard on when you sign in", self.t_startup))
        self.c_fps = combo(GUARD_FPS, s.guard_fps)
        bl.addWidget(SettingRow("Checks per second", "Higher reacts faster but uses more CPU", self.c_fps))
        lay.addWidget(bg)

        for w in (self.t_absence, self.t_activity, self.t_intruder, self.t_photos, self.t_shoulder, self.t_tray):
            w.toggled.connect(self._save_rules)
        for w in (self.c_timeout, self.c_confirm, self.c_fps, self.c_locked):
            w.currentIndexChanged.connect(self._save_rules)
        self.t_startup.toggled.connect(self._toggle_startup)

        note, nl = card("How this fits with Windows sign-in")
        nl.addWidget(label(
            "Windows Hello face sign-in needs an infrared (IR) camera so it can't be fooled by a photo; normal webcams "
            "aren't allowed to unlock Windows. FaceGuard therefore never replaces your PIN or password. It adds what a "
            "normal webcam can do safely: it locks the PC the moment you leave and records anyone else who tries to use "
            "it. You always unlock with your usual Windows PIN or password.", "secondary", wrap=True))
        lay.addWidget(note)

        log_card, ll = card("Event log")
        self.log = QListWidget()
        self.log.setMinimumHeight(140)
        ll.addWidget(self.log)
        lay.addWidget(log_card)
        lay.addStretch(1)
        outer.addWidget(scrolling(content))

        ctx.db_changed.connect(self._refresh_owner)
        self._refresh_owner()
        self._apply_status(None)

    # ------------------------------------------------------------ owner / status
    def _refresh_owner(self) -> None:
        owner = self.ctx.db.owner()
        if owner is None or not owner.samples:
            self.owner_pic.setPixmap(rounded_pixmap(None, 56, 12))
            self.owner_text.setText("No owner enrolled yet. Enroll your face first - the guard needs to know who you are.")
        else:
            thumb = decode_thumbnail(owner.thumbnails[0]) if owner.thumbnails else None
            self.owner_pic.setPixmap(rounded_pixmap(thumb, 56, 12))
            trusted = [p.name for p in self.ctx.db.persons if p.trusted and not p.is_owner]
            extra = f" Also allowed: {', '.join(trusted)}." if trusted else ""
            self.owner_text.setText(f"{owner.name} ({owner.count()} face samples).{extra}")

    def _apply_status(self, status) -> None:
        on = self.ctx.settings.guard_enabled
        state = status.state if (status is not None and on) else GuardState.STOPPED
        glyph, color = STATE_STYLE.get(state, (icons.SHIELD, theme.MUTED))
        self.state_icon.set_glyph(glyph, color)
        if state == GuardState.STOPPED:
            self.state_text.setText("Guard is off")
            self.state_detail.setText("Turn it on to lock the PC when you leave and capture intruders.")
        else:
            self.state_text.setText(status.text)
            details = {
                GuardState.PRESENT: "You're recognised. The PC stays unlocked.",
                GuardState.WATCHING: "Look at the camera to stay unlocked.",
                GuardState.INTRUDER: "An unrecognised person was seen at the PC.",
                GuardState.LOCKED: ("Anyone unrecognised who sits down will be photographed. " if "watching" in status.text else "")
                + "Unlock with your Windows PIN or password; the guard resumes automatically.",
                GuardState.PAUSED: "Paused - the camera is off.",
                GuardState.NO_OWNER: "Enroll yourself on the Enroll page.",
                GuardState.NO_CAMERA: "Check that no other app is using the camera.",
            }
            self.state_detail.setText(details.get(state, ""))
        paused = state == GuardState.PAUSED
        self.pause15.setVisible(on and not paused)
        self.pause60.setVisible(on and not paused)
        self.resume.setVisible(on and paused)

    def update_status(self, status) -> None:
        prev = getattr(self, "_last_state", None)
        self._apply_status(status)
        if status is not None and status.state != prev and self.ctx.settings.guard_enabled:
            self.add_event(status.text)
        self._last_state = status.state if status is not None else None

    def add_event(self, text: str) -> None:
        self.log.insertItem(0, f"{time.strftime('%H:%M:%S')}   {text}")
        while self.log.count() > 200:
            self.log.takeItem(self.log.count() - 1)

    # ------------------------------------------------------------ actions
    def _toggle_guard(self, on: bool) -> None:
        owner = self.ctx.db.owner()
        if on and (owner is None or not owner.samples):
            QMessageBox.information(self, "Guard", "Enroll your face first so the guard knows who you are.")
            self.toggle.blockSignals(True)
            self.toggle.setChecked(False)
            self.toggle.blockSignals(False)
            return
        self.set_enabled(on)

    def set_enabled(self, on: bool) -> None:
        """Also used by the tray menu."""
        s = self.ctx.settings
        if s.guard_enabled == on and self.toggle.isChecked() == on:
            return
        s.guard_enabled = on
        self.ctx.save_settings()
        self.ctx.worker.guard_command("start" if on else "stop")
        if self.toggle.isChecked() != on:
            self.toggle.blockSignals(True)
            self.toggle.setChecked(on)
            self.toggle.blockSignals(False)
        self.add_event("Guard turned on" if on else "Guard turned off")
        self._apply_status(None if not on else self.ctx.guard_status)

    def _pause(self, seconds: int) -> None:
        self.ctx.worker.guard_command("pause", seconds)
        self.add_event(f"Paused for {seconds // 60} minutes")

    def _resume(self) -> None:
        self.ctx.worker.guard_command("resume")
        self.add_event("Resumed")

    def _save_rules(self) -> None:
        s = self.ctx.settings
        s.guard_lock_on_absence = self.t_absence.isChecked()
        s.guard_absence_timeout = int(self.c_timeout.currentData())
        s.guard_respect_activity = self.t_activity.isChecked()
        s.guard_lock_on_intruder = self.t_intruder.isChecked()
        s.guard_intruder_seconds = float(self.c_confirm.currentData())
        s.guard_save_intruders = self.t_photos.isChecked()
        s.guard_locked_watch_minutes = int(self.c_locked.currentData())
        s.guard_shoulder_alert = self.t_shoulder.isChecked()
        s.minimize_to_tray = self.t_tray.isChecked()
        s.guard_fps = float(self.c_fps.currentData())
        self.ctx.save_settings()

    def _toggle_startup(self, on: bool) -> None:
        try:
            winsys.set_run_at_startup(on)
            self.ctx.toast.emit("good", "FaceGuard will start when you sign in" if on else "Removed from Windows startup")
        except OSError as exc:
            QMessageBox.warning(self, "Start with Windows", str(exc))
            self.t_startup.blockSignals(True)
            self.t_startup.setChecked(not on)
            self.t_startup.blockSignals(False)
