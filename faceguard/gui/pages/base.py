from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from faceguard.gui.context import AppContext
from faceguard.gui.widgets import label


class Page(QWidget):
    key = ""
    title = ""
    glyph = ""
    uses_camera = False

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Page")

    def wants_camera(self) -> bool:
        return self.uses_camera

    def on_show(self) -> None:
        pass

    def on_hide(self) -> None:
        pass

    def on_frame(self, packet) -> None:
        pass


def header(title: str, subtitle: str = "", right: QWidget | None = None) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    text = QVBoxLayout()
    text.setSpacing(2)
    text.addWidget(label(title, "h1"))
    if subtitle:
        text.addWidget(label(subtitle, "muted", wrap=True))
    lay.addLayout(text, 1)
    if right is not None:
        lay.addWidget(right, 0, Qt.AlignmentFlag.AlignTop)
    return w


def scrolling(content: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(content)
    return area


def page_layout(page: QWidget, margins: int = 24, spacing: int = 16) -> QVBoxLayout:
    lay = QVBoxLayout(page)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    return lay
