from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QCompleter, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QStackedWidget, QVBoxLayout, QWidget,
)

from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext
from faceguard.gui.pages.base import Page
from faceguard.gui.widgets import Guide, ProgressRing, VideoView, button, card, glyph_label, label, rounded_pixmap, run_job
from faceguard.gui.worker import EnrollTask
from faceguard.recognition.enrollment import (
    EnrollmentResult, EnrollmentSession, PhotoImportReport, find_duplicate, samples_from_photos,
)
from faceguard.recognition.gallery import Matcher
from faceguard.vision.camera import IMAGE_EXTS

TIPS = [
    "Face a window or lamp - light in front of you, not behind",
    "Take off sunglasses and masks (normal glasses are fine)",
    "Keep your whole face inside the oval",
    "Move slowly when asked to turn",
]


class PoseStep(QWidget):
    def __init__(self, text: str) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.icon = glyph_label(icons.CHECK, 11, theme.DISABLED)
        self.icon.setFixedWidth(22)
        self.text = label(text, "secondary")
        self.count = label("", "small")
        lay.addWidget(self.icon)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.count)

    def set(self, status: str, collected: int, needed: int) -> None:
        glyph, color = {
            "done": (icons.CHECK, theme.GOOD), "active": (icons.PLAY, theme.ACCENT),
            "skipped": (icons.CROSS, theme.MUTED), "pending": (icons.CHECK, theme.DISABLED),
        }[status]
        self.icon.set_glyph(glyph, color)
        self.text.setProperty("role", "h3" if status == "active" else "secondary")
        self.text.style().unpolish(self.text)
        self.text.style().polish(self.text)
        self.count.setText("skipped" if status == "skipped" else f"{min(collected, needed)}/{needed}")


class EnrollPage(Page):
    key = "enroll"
    title = "Enroll"
    glyph = icons.ADD_PERSON
    uses_camera = True

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        self.task: EnrollTask | None = None
        self.pending: EnrollmentResult | PhotoImportReport | None = None
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)
        self.video = VideoView()
        self.video.clear("Starting camera...")
        root.addWidget(self.video, 1)
        self.panels = QStackedWidget()
        self.panels.setFixedWidth(380)
        root.addWidget(self.panels)
        self.panels.addWidget(self._build_setup())
        self.panels.addWidget(self._build_capture())
        self.panels.addWidget(self._build_review())
        ctx.db_changed.connect(self._refresh_names)
        self._refresh_names()

    # ------------------------------------------------------------ panels
    def _build_setup(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        lay.addWidget(label("Enroll a face", "h1"))
        lay.addWidget(label("FaceGuard captures about 20 samples of your face from different angles and turns "
                            "them into a 128-number 'faceprint' with a deep neural network. Photos are never stored "
                            "- only the encrypted faceprint and a few small thumbnails.", "muted", wrap=True))
        form, fl = card()
        fl.addWidget(label("Name"))
        self.name = QLineEdit()
        self.name.setPlaceholderText("Your name")
        self.name.textChanged.connect(self._name_changed)
        fl.addWidget(self.name)
        self.existing_hint = label("", "small", wrap=True)
        fl.addWidget(self.existing_hint)
        self.owner = QCheckBox("This is the owner of this PC")
        self.trusted = QCheckBox("Trusted - may use this PC without triggering the guard")
        fl.addWidget(self.owner)
        fl.addWidget(self.trusted)
        lay.addWidget(form)
        self.start_btn = button("Start camera enrollment", "primary", icons.VIDEO)
        self.start_btn.clicked.connect(self._start)
        self.photos_btn = button("Import photos instead...", None, icons.PHOTO)
        self.photos_btn.clicked.connect(self._import_photos)
        lay.addWidget(self.start_btn)
        lay.addWidget(self.photos_btn)
        tips, tl = card("For best results")
        for t in TIPS:
            row = QHBoxLayout()
            row.addWidget(glyph_label(icons.CHECK, 9, theme.GOOD), 0, Qt.AlignmentFlag.AlignTop)
            row.addWidget(label(t, "secondary", wrap=True), 1)
            tl.addLayout(row)
        lay.addWidget(tips)
        lay.addStretch(1)
        self._name_changed("")
        return w

    def _build_capture(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        self.cap_title = label("Enrolling", "muted")
        lay.addWidget(self.cap_title)
        self.instruction = label("Look straight at the camera", "h1", wrap=True)
        lay.addWidget(self.instruction)
        self.hint = label("", "secondary", wrap=True)
        self.hint.setMinimumHeight(40)
        lay.addWidget(self.hint)
        ring_row = QHBoxLayout()
        self.ring = ProgressRing(132)
        ring_row.addStretch(1)
        ring_row.addWidget(self.ring)
        ring_row.addStretch(1)
        lay.addLayout(ring_row)
        steps, sl = card("Poses")
        self.steps = {}
        for key, text in (("center", "Straight"), ("left", "Turn left"), ("right", "Turn right"), ("up", "Chin up"), ("down", "Chin down")):
            self.steps[key] = PoseStep(text)
            sl.addWidget(self.steps[key])
        lay.addWidget(steps)
        row = QHBoxLayout()
        self.skip_btn = button("Skip this pose", None, None, "Can't do this pose? Skip it (a straight view is required)")
        self.skip_btn.clicked.connect(lambda: self.task and self.task.skip_pose())
        cancel = button("Cancel", "ghost")
        cancel.clicked.connect(self._cancel)
        row.addWidget(self.skip_btn)
        row.addStretch(1)
        row.addWidget(cancel)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _build_review(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(14)
        self.rev_title = label("Looking good!", "h1")
        lay.addWidget(self.rev_title)
        c, cl = card()
        thumbs = QHBoxLayout()
        self.thumb_labels = [QLabel() for _ in range(3)]
        for t in self.thumb_labels:
            t.setFixedSize(88, 88)
            thumbs.addWidget(t)
        thumbs.addStretch(1)
        cl.addLayout(thumbs)
        self.rev_summary = label("", "secondary", wrap=True)
        cl.addWidget(self.rev_summary)
        lay.addWidget(c)
        self.dup_card, dl = card()
        self.dup_text = label("", None, wrap=True)
        dl.addWidget(self.dup_text)
        self.dup_card.setStyleSheet(f"QFrame#Card {{ border-color: {theme.WARNING}; }}")
        lay.addWidget(self.dup_card)
        row = QHBoxLayout()
        self.save_btn = button("Save", "primary", icons.CHECK)
        self.save_btn.clicked.connect(self._save)
        discard = button("Discard", "ghost")
        discard.clicked.connect(self._reset)
        row.addWidget(self.save_btn)
        row.addStretch(1)
        row.addWidget(discard)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------ setup logic
    def _refresh_names(self) -> None:
        names = [p.name for p in self.ctx.db.persons]
        self.name.setCompleter(QCompleter(names))
        self._name_changed(self.name.text())

    def _name_changed(self, text: str) -> None:
        existing = self.ctx.db.find(text) if text.strip() else None
        has_owner = self.ctx.db.owner() is not None
        if existing:
            self.existing_hint.setText(f"{existing.name} is already enrolled ({existing.count()} samples) - new samples will be added.")
            self.owner.setChecked(existing.is_owner)
            self.trusted.setChecked(existing.trusted)
        else:
            self.existing_hint.setText("")
            self.owner.setChecked(not has_owner)
        self.start_btn.setEnabled(bool(text.strip()))
        self.photos_btn.setEnabled(bool(text.strip()))

    def _start(self) -> None:
        name = self.name.text().strip()
        if not name:
            return
        self.task = EnrollTask(EnrollmentSession(name))
        self.cap_title.setText(f"Enrolling {name}")
        self.ring.set_value(0)
        self.panels.setCurrentIndex(1)
        self.ctx.worker.start_task(self.task)

    def _cancel(self) -> None:
        if self.task:
            self.ctx.worker.cancel_task()
        self.task = None
        self._reset()

    def _import_photos(self) -> None:
        name = self.name.text().strip()
        patterns = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, f"Photos of {name}", "", f"Images ({patterns})")
        if not files:
            return
        self.photos_btn.setEnabled(False)
        self.photos_btn.setText("Analysing photos...")
        settings = self.ctx.settings

        def work():
            from faceguard.recognition.engine import FaceEngine

            engine = FaceEngine.create(settings)  # separate instance: the camera thread owns the other one
            return samples_from_photos(engine, files)

        run_job(work, on_done=self._photos_done, on_error=lambda e: self._photos_failed(e))

    def _photos_failed(self, error: str) -> None:
        self.photos_btn.setEnabled(True)
        self.photos_btn.setText("Import photos instead...")
        self.ctx.toast.emit("error", f"Photo import failed: {error}")

    def _photos_done(self, report: PhotoImportReport) -> None:
        self.photos_btn.setEnabled(True)
        self.photos_btn.setText("Import photos instead...")
        if not report.samples:
            reasons = "; ".join(f"{n}: {r}" for n, r in report.skipped[:4])
            self.ctx.toast.emit("warn", f"No usable faces in those photos. {reasons}")
            return
        skipped = f" Skipped {len(report.skipped)}: " + "; ".join(f"{n} ({r})" for n, r in report.skipped[:3]) if report.skipped else ""
        self._show_review(report.samples, report.thumbnails, f"{len(report.samples)} photo(s) used.{skipped}", report)

    # ------------------------------------------------------------ capture feedback
    def on_frame(self, packet) -> None:
        state = packet.task_state if packet.task_kind == "enroll" else None
        guide = None
        banner = ""
        if state is not None and self.panels.currentIndex() == 1:
            guide = Guide(state.progress, theme.GOOD if state.captured else theme.ACCENT)
            banner = state.instruction
            self.instruction.setText(state.instruction)
            self.hint.setText(state.hint if state.hint not in ("Captured", "Hold that pose...") else "")
            self.ring.set_value(state.progress, caption="captured")
            for tp in state.targets:
                self.steps[tp.target.key].set(tp.status, tp.collected, tp.target.count)
            self.skip_btn.setEnabled(state.current is not None and state.current.key != "center")
        self.video.set_frame(packet.image, self.ctx.overlays(packet.result, landmarks=False, mesh=False, pose=False,
                                                             titles=guide is None), packet.mirrored, guide, banner)

    def task_finished(self, task, outcome: EnrollmentResult | None) -> None:
        if task is not self.task:
            return  # a late signal from an earlier, already-cancelled enrollment
        self.task = None
        if outcome is None:
            self._reset()
            return
        skipped = f" Skipped: {', '.join(outcome.skipped)}." if outcome.skipped else ""
        self._show_review(outcome.samples, outcome.thumbnails,
                          f"{len(outcome.samples)} samples captured, consistency {outcome.consistency:.2f}.{skipped}", outcome)

    def _show_review(self, samples, thumbnails, summary: str, pending) -> None:
        import numpy as np

        from faceguard.vision.embedder import normalize

        self.pending = pending
        self._pending_samples, self._pending_thumbs = samples, thumbnails
        for i, t in enumerate(self.thumb_labels):
            t.setPixmap(rounded_pixmap(thumbnails[i] if i < len(thumbnails) else None, 88, 14))
        self.rev_summary.setText(summary)
        s = self.ctx.settings
        existing = self.ctx.db.find(self.name.text())
        centroid = normalize(np.mean([x.embedding for x in samples], axis=0))
        dup = find_duplicate(self.ctx.db.snapshot(), Matcher(s.match_threshold, s.unknown_threshold, s.top_k), centroid,
                             exclude_id=existing.id if existing else None)
        if dup:
            self.dup_text.setText(f"This face already matches {dup.person.name} (similarity {dup.similarity:.2f}). "
                                  f"Saving would teach FaceGuard that {self.name.text().strip()} and {dup.person.name} are "
                                  "different people who look alike. If it is the same person, discard and enroll under "
                                  f"the name {dup.person.name} instead.")
            self.dup_card.show()
            self.rev_title.setText("Wait - is this someone else?")
        else:
            self.dup_card.hide()
            self.rev_title.setText("Looking good!")
        self.panels.setCurrentIndex(2)

    def _save(self) -> None:
        db = self.ctx.db
        name = self.name.text().strip()
        try:
            person = db.find(name) or db.add_person(name, is_owner=self.owner.isChecked(), trusted=self.trusted.isChecked())
            db.add_samples(person.id, self._pending_samples, self._pending_thumbs)
            if self.owner.isChecked() and not person.is_owner:
                db.set_owner(person.id)
            if self.trusted.isChecked() != person.trusted and not self.owner.isChecked():
                db.set_trusted(person.id, self.trusted.isChecked())
        except (ValueError, OSError) as exc:
            self.ctx.toast.emit("error", f"Could not save: {exc}")
            return
        self.ctx.toast.emit("good", f"{name} enrolled with {db.get(person.id).count()} samples")
        self.name.clear()
        self._reset()
        self.ctx.navigate.emit("people")

    def _reset(self) -> None:
        self.pending = None
        self.panels.setCurrentIndex(0)
        self._name_changed(self.name.text())

    def on_hide(self) -> None:
        if self.task is not None:
            self._cancel()
