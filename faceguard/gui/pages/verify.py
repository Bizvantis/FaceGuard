from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QListWidget, QListWidgetItem, QProgressBar, QVBoxLayout, QWidget

from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext
from faceguard.gui.pages.base import Page
from faceguard.gui.widgets import Guide, Meter, VideoView, button, card, glyph_label, label
from faceguard.gui.worker import VerifyTask
from faceguard.security.liveness import Challenge, LivenessConfig, VerificationResult

CHALLENGE_TEXT = {
    Challenge.TURN_LEFT: "Turn left", Challenge.TURN_RIGHT: "Turn right",
    Challenge.BLINK: "Blink", Challenge.SMILE: "Smile",
}


class VerifyPage(Page):
    key = "verify"
    title = "Face unlock"
    glyph = icons.FACE
    uses_camera = True

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        self.task: VerifyTask | None = None
        self._result_until = 0.0
        self._last_result: VerificationResult | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)
        self.video = VideoView()
        self.video.clear("Starting camera...")
        root.addWidget(self.video, 1)

        side = QVBoxLayout()
        side.setSpacing(14)
        side.addWidget(label("Face unlock", "h1"))
        side.addWidget(label("Proves that a real, live person is in front of the camera (not a photo or a screen) "
                             "and identifies them. You'll get 3 random actions - follow the prompts.", "muted", wrap=True))
        self.owner_only = QCheckBox("Only accept the owner of this PC")
        side.addWidget(self.owner_only)
        self.start_btn = button("Start verification", "primary", icons.FACE)
        self.start_btn.clicked.connect(self._start)
        side.addWidget(self.start_btn)

        steps, sl = card("Challenge")
        self.step_title = label("Press start", "h2", wrap=True)
        sl.addWidget(self.step_title)
        self.step_hint = label("", "secondary", wrap=True)
        sl.addWidget(self.step_hint)
        sl.addWidget(label("Action progress", "small"))
        self.meter = Meter()
        sl.addWidget(self.meter)
        sl.addWidget(label("Time left", "small"))
        self.timer = QProgressBar()
        self.timer.setRange(0, 1000)
        self.timer.setTextVisible(False)
        sl.addWidget(self.timer)
        self.seq_row = QHBoxLayout()
        self.seq_labels: list = []
        sl.addLayout(self.seq_row)
        side.addWidget(steps)

        self.result_card, rl = card()
        top = QHBoxLayout()
        self.result_icon = glyph_label(icons.CHECK, 22, theme.GOOD)
        self.result_title = label("", "h2", wrap=True)
        top.addWidget(self.result_icon)
        top.addWidget(self.result_title, 1)
        rl.addLayout(top)
        self.result_detail = label("", "secondary", wrap=True)
        rl.addWidget(self.result_detail)
        self.result_card.hide()
        side.addWidget(self.result_card)

        hist, hl = card("Recent attempts")
        self.history = QListWidget()
        self.history.setMinimumHeight(110)
        hl.addWidget(self.history)
        side.addWidget(hist, 1)

        wrap = QWidget()
        wrap.setLayout(side)
        wrap.setFixedWidth(380)
        root.addWidget(wrap)

    # ------------------------------------------------------------ actions
    def _start(self) -> None:
        s = self.ctx.settings
        cfg = LivenessConfig(num_challenges=s.liveness_challenges, challenge_timeout=s.liveness_timeout)
        self.task = VerifyTask(cfg, s.liveness_expressions, self.owner_only.isChecked())
        self.result_card.hide()
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Verifying...")
        self.step_title.setText("Look straight at the camera")
        self._set_sequence([])
        self.ctx.worker.start_task(self.task)

    def _set_sequence(self, seq: list, done: int = 0, active: int | None = None) -> None:
        while self.seq_row.count():
            item = self.seq_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.seq_labels = []
        for i, c in enumerate(seq):
            color = theme.GOOD if i < done else (theme.ACCENT if i == active else theme.MUTED)
            glyph = icons.CHECK if i < done else str(i + 1)
            box = QHBoxLayout()
            g = glyph_label(glyph, 9, color) if i < done else label(glyph, "small")
            box.addWidget(g)
            box.addWidget(label(CHALLENGE_TEXT[c], "secondary" if i != active else "h3"))
            w = QWidget()
            w.setLayout(box)
            self.seq_row.addWidget(w)
        self.seq_row.addStretch(1)
        self._seq_key = (tuple(seq), done, active)

    def on_frame(self, packet) -> None:
        state = packet.task_state if packet.task_kind == "verify" else None
        guide, banner = None, ""
        if state is not None:
            banner = state.instruction
            self.step_title.setText(state.instruction if state.phase != "align" else "Look straight at the camera")
            self.step_hint.setText(state.hint)
            self.meter.set(state.meter if state.phase == "challenge" else 0.0, theme.ACCENT)
            limit = max(0.1, self.ctx.settings.liveness_timeout)
            self.timer.setValue(int(1000 * state.time_left / limit) if state.phase in ("challenge", "recenter", "final") else 1000)
            guide = Guide(state.progress, theme.GOOD if state.phase == "passed" else theme.ACCENT)
            seq = packet.task_sequence
            key = (tuple(seq), len(state.completed), state.step - 1 if state.phase == "challenge" else None)
            if seq and key != getattr(self, "_seq_key", None):
                self._set_sequence(seq, len(state.completed), key[2])
        elif time.monotonic() < self._result_until and self._last_result is not None:
            r = self._last_result
            banner = ("Verified: " + r.match.person.name) if r.verified and r.match.person else "Not verified"
        titles = state is None
        self.video.set_frame(packet.image, self.ctx.overlays(packet.result, landmarks=False, mesh=False, pose=False, titles=titles),
                             packet.mirrored, guide, banner)

    def task_finished(self, task, outcome: VerificationResult | None) -> None:
        if task is not self.task:
            return  # a late signal from an earlier, already-cancelled attempt
        self.task = None
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start verification")
        self.meter.set(0.0)
        self.timer.setValue(0)
        if outcome is None:
            self.step_title.setText("Cancelled")
            return
        self._last_result = outcome
        self._result_until = time.monotonic() + 4.0
        ok = outcome.verified
        name = outcome.match.person.name if outcome.match.person else "unknown"
        self.result_icon.set_glyph(icons.CHECK if ok else icons.CROSS, theme.GOOD if ok else theme.CRITICAL)
        self.result_card.setStyleSheet(f"QFrame#Card {{ border-color: {theme.GOOD if ok else theme.CRITICAL}; }}")
        if ok:
            self.result_title.setText(f"Verified: {name}")
            self.result_detail.setText(f"Live person confirmed in {outcome.duration:.1f} s  |  similarity {outcome.match.similarity:.2f} "
                                       f"(threshold {self.ctx.settings.match_threshold:.2f})")
            self.step_title.setText("Done")
        else:
            self.result_title.setText("Not verified")
            detail = outcome.reason or "Liveness check failed"
            if outcome.live and outcome.match.person is not None:
                detail += f"  |  closest match {name} ({outcome.match.similarity:.2f})"
            self.result_detail.setText(detail)
            self.step_title.setText("Try again")
        self.result_card.show()
        stamp = time.strftime("%H:%M:%S")
        item = QListWidgetItem(f"{stamp}   {'Verified ' + name if ok else 'Denied - ' + (outcome.reason or 'failed')}")
        item.setForeground(Qt.GlobalColor.white if ok else Qt.GlobalColor.lightGray)
        self.history.insertItem(0, item)
        while self.history.count() > 8:
            self.history.takeItem(self.history.count() - 1)

    def on_hide(self) -> None:
        if self.task is not None:
            self.ctx.worker.cancel_task()
            self.task_finished(self.task, None)
