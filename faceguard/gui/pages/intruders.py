from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem, QMessageBox,
    QSizePolicy, QWidget,
)

from faceguard.gui import icons, theme
from faceguard.gui.context import AppContext
from faceguard.gui.pages.base import Page, header, page_layout
from faceguard.gui.widgets import button, card, label
from faceguard.security import winsys
from faceguard.security.intruders import IntruderEvent

RETENTION = [(7, "7 days"), (30, "30 days"), (90, "90 days"), (0, "Forever")]


class IntrudersPage(Page):
    key = "intruders"
    title = "Intruders"
    glyph = icons.CAMERA

    def __init__(self, ctx: AppContext) -> None:
        super().__init__(ctx)
        self.events: list[IntruderEvent] = []
        lay = page_layout(self)
        right = QHBoxLayout()
        open_btn = button("Open folder", None, icons.FOLDER)
        open_btn.clicked.connect(lambda: winsys.open_path(ctx.intruders.folder))
        self.del_btn = button("Delete selected", "ghost", icons.DELETE)
        self.del_btn.clicked.connect(self._delete_selected)
        clear = button("Clear all", "danger", icons.DELETE)
        clear.clicked.connect(self._clear)
        for b in (open_btn, self.del_btn, clear):
            right.addWidget(b)
        rw = QWidget()
        rw.setLayout(right)
        lay.addWidget(header("Intruders & snapshots", "Photos the guard took of unrecognised people at your PC, plus "
                             "snapshots you saved. Stored only on this PC.", rw))

        body = QHBoxLayout()
        body.setSpacing(16)
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setIconSize(QSize(208, 117))
        self.grid.setGridSize(QSize(228, 184))
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.setWordWrap(True)
        self.grid.currentRowChanged.connect(self._preview)
        body.addWidget(self.grid, 3)

        pv, pl = card()
        pv.setMinimumWidth(360)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(220)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet(f"background: {theme.PAGE}; border-radius: 10px;")
        pl.addWidget(self.preview, 1)
        self.p_title = label("Nothing selected", "h2")
        self.p_detail = label("", "secondary", wrap=True)
        pl.addWidget(self.p_title)
        pl.addWidget(self.p_detail)
        keep = QHBoxLayout()
        keep.addWidget(label("Keep snapshots for", "muted"))
        self.retention = QComboBox()
        for v, t in RETENTION:
            self.retention.addItem(t, v)
        cur = ctx.settings.intruder_retention_days
        self.retention.setCurrentIndex(next((i for i, (v, _) in enumerate(RETENTION) if v == cur), 1))
        self.retention.currentIndexChanged.connect(self._retention_changed)
        keep.addWidget(self.retention)
        keep.addStretch(1)
        pl.addLayout(keep)
        body.addWidget(pv, 2)
        lay.addLayout(body, 1)

        self.empty = label("No intruders so far. When the guard is on and an unrecognised person sits at your PC, "
                           "their photo appears here.", "muted", wrap=True)
        lay.addWidget(self.empty)
        ctx.intruders_changed.connect(self.refresh)
        self.refresh()

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.events = self.ctx.intruders.events()
        self.grid.clear()
        for ev in self.events:
            pm = QPixmap(str(self.ctx.intruders.path_of(ev))).scaled(
                QSize(208, 117), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            icon = QIcon(pm)
            icon.addPixmap(pm, QIcon.Mode.Selected)  # no blue tint on the selected photo
            item = QListWidgetItem(icon, f"{ev.when}\n{ev.label}")
            item.setSizeHint(QSize(220, 180))
            self.grid.addItem(item)
        self.empty.setVisible(not self.events)
        self.del_btn.setEnabled(bool(self.events))
        if self.events:
            self.grid.setCurrentRow(0)
        else:
            self._preview(-1)

    def _preview(self, row: int) -> None:
        if row < 0 or row >= len(self.events):
            self.preview.clear()
            self.p_title.setText("Nothing selected")
            self.p_detail.setText("")
            return
        ev = self.events[row]
        pm = QPixmap(str(self.ctx.intruders.path_of(ev)))
        self._current_pm = pm
        self._fit_preview()
        self.p_title.setText(ev.label)
        people = ", ".join(
            f"{f['name'] or 'Unknown'} ({f['similarity']:.2f})" for f in ev.faces
        ) or "-"
        self.p_detail.setText(f"{ev.when}\nFaces: {people}\nFile: {ev.file}")

    def _fit_preview(self) -> None:
        pm = getattr(self, "_current_pm", None)
        if pm is not None and not pm.isNull():
            self.preview.setPixmap(pm.scaled(self.preview.size() - QSize(8, 8), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit_preview()

    def _delete_selected(self) -> None:
        rows = sorted({i.row() for i in self.grid.selectedIndexes()})
        if not rows:
            return
        self.ctx.intruders.delete([self.events[r] for r in rows])
        self.refresh()

    def _clear(self) -> None:
        if not self.events:
            return
        answer = QMessageBox.question(self, "Clear all", f"Delete all {len(self.events)} snapshots?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                      QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Yes:
            self.ctx.intruders.clear()
            self.refresh()

    def _retention_changed(self) -> None:
        self.ctx.settings.intruder_retention_days = int(self.retention.currentData())
        self.ctx.save_settings()
        removed = self.ctx.intruders.cleanup(self.ctx.settings.intruder_retention_days)
        if removed:
            self.refresh()
