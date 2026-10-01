"""Reusable widgets: live video view with overlays, progress ring, tiles, pills, toggles, jobs."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import (
    QEasingCurve, QLineF, QObject, QPointF, QRectF, QRunnable, QSize, Qt, QThreadPool, QVariantAnimation, Signal,
)
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from faceguard.gui import icons, theme

log = logging.getLogger(__name__)

# Alignment and text flags are different enum types in PySide6; combine them as plain ints.
WRAP_CENTER = Qt.AlignmentFlag.AlignHCenter.value | Qt.AlignmentFlag.AlignTop.value | Qt.TextFlag.TextWordWrap.value


# ---------------------------------------------------------------- small helpers
def label(text: str = "", role: str | None = None, wrap: bool = False, selectable: bool = False) -> QLabel:
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    lab.setWordWrap(wrap)
    if selectable:
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def button(text: str, kind: str | None = None, glyph: str | None = None, tooltip: str | None = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if glyph:
        color = theme.INK if kind == "primary" else ("#e66767" if kind == "danger" else theme.INK2)
        b.setIcon(icons.glyph_icon(glyph, color, 16))
        b.setIconSize(QSize(16, 16))
    if tooltip:
        b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


class GlyphLabel(QLabel):
    """An icon-font glyph. The font is set in the widget's own stylesheet because the app-wide
    ``* { font-family }`` rule would otherwise override setFont() and render empty boxes."""

    def __init__(self, glyph: str, size: float = 14, color: str = theme.INK2) -> None:
        super().__init__()
        self._size, self._color = size, color
        self.set_glyph(glyph, color)

    def set_glyph(self, glyph: str | None = None, color: str | None = None) -> None:
        if glyph is not None:
            self.setText(glyph)
        if color is not None:
            self._color = color
        self.setStyleSheet(f'font-family: "{icons.icon_family()}"; font-size: {self._size}pt; '
                           f"color: {self._color}; background: transparent;")


def glyph_label(glyph: str, size: float = 14, color: str = theme.INK2) -> GlyphLabel:
    return GlyphLabel(glyph, size, color)


def divider() -> QFrame:
    f = QFrame()
    f.setObjectName("Divider")
    return f


def card(title: str | None = None, subtitle: str | None = None, margins: int = 16, spacing: int = 10) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    if title:
        lay.addWidget(label(title, "h3"))
    if subtitle:
        lay.addWidget(label(subtitle, "muted", wrap=True))
    return frame, lay


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()


def rounded_pixmap(bgr: np.ndarray | None, size: int, radius: float = 10.0) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, size, size), radius, radius)
    p.setClipPath(path)
    if bgr is None:
        p.fillRect(pm.rect(), QColor(theme.RAISED))
        p.setPen(QColor(theme.MUTED))
        p.setFont(icons.icon_font(size / 4))
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, icons.CONTACT)
    else:
        img = bgr_to_qimage(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA))
        p.drawImage(QRectF(0, 0, size, size), img)
    p.end()
    return pm


# ---------------------------------------------------------------- background jobs
class _JobSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    def __init__(self, fn: Callable, *args) -> None:
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = _JobSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args)
        except Exception as exc:  # reported to the UI
            log.exception("background job failed")
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(result)


_jobs: set[Job] = set()


def run_job(fn: Callable, *args, on_done: Callable | None = None, on_error: Callable | None = None) -> Job:
    job = Job(fn, *args)
    job.setAutoDelete(False)
    _jobs.add(job)
    job.signals.done.connect(lambda r: (_jobs.discard(job), on_done and on_done(r)))
    job.signals.failed.connect(lambda e: (_jobs.discard(job), on_error and on_error(e)))
    QThreadPool.globalInstance().start(job)
    return job


# ---------------------------------------------------------------- video view
@dataclass
class FaceBox:
    rect: tuple[float, float, float, float]
    color: str
    title: str
    subtitle: str = ""
    landmarks: np.ndarray | None = None
    mesh: np.ndarray | None = None
    pose: tuple[float, float] | None = None  # (yaw, pitch) degrees, for the head-pose arrow


@dataclass
class Guide:
    progress: float
    color: str = theme.ACCENT


class VideoView(QWidget):
    """Letterboxed camera frame with anti-aliased overlays drawn in (mirrored) frame coordinates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._faces: list[FaceBox] = []
        self._mirrored = True
        self._guide: Guide | None = None
        self._banner = ""
        self._placeholder = "Camera is off"
        self._detail = ""
        self.setMinimumSize(360, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, image: QImage, faces: list[FaceBox], mirrored: bool, guide: Guide | None = None, banner: str = "") -> None:
        self._image, self._faces, self._mirrored, self._guide, self._banner = image, faces, mirrored, guide, banner
        self.update()

    def clear(self, text: str, detail: str = "") -> None:
        self._image, self._faces, self._guide, self._banner = None, [], None, ""
        self._placeholder, self._detail = text, detail
        self.update()

    @property
    def has_image(self) -> bool:
        return self._image is not None

    def paintEvent(self, _event) -> None:  # noqa: C901 - one linear drawing routine
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        area = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(area, 12, 12)
        p.fillPath(clip, QColor(theme.SURFACE))
        p.setClipPath(clip)

        if self._image is None:
            p.setPen(QColor(theme.MUTED))
            p.setFont(icons.icon_font(30))
            c = area.center()
            p.drawText(QRectF(c.x() - 40, c.y() - 60, 80, 50), Qt.AlignmentFlag.AlignCenter, icons.CAMERA)
            p.setFont(QFont(theme.FONT, 11))
            p.setPen(QColor(theme.INK2))
            p.drawText(QRectF(area.left() + 20, c.y() - 4, area.width() - 40, 26), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            if self._detail:
                p.setFont(QFont(theme.FONT, 9))
                p.setPen(QColor(theme.MUTED))
                p.drawText(QRectF(area.left() + 30, c.y() + 24, area.width() - 60, 60), WRAP_CENTER, self._detail)
            return

        iw, ih = self._image.width(), self._image.height()
        s = min(area.width() / iw, area.height() / ih)
        tw, th = iw * s, ih * s
        ox, oy = area.left() + (area.width() - tw) / 2, area.top() + (area.height() - th) / 2
        target = QRectF(ox, oy, tw, th)
        p.drawImage(target, self._image)

        def mx(x):
            return ox + ((iw - x) if self._mirrored else x) * s

        # dense mesh first so boxes and labels sit on top
        mesh_faces = [f for f in self._faces if f.mesh is not None]
        if mesh_faces:
            from faceguard.vision.mesh import tesselation_edges

            edges = tesselation_edges()
            for f in mesh_faces:
                xs = mx(f.mesh[:, 0])
                ys = oy + f.mesh[:, 1] * s
                a, b = edges[:, 0], edges[:, 1]
                lines = [QLineF(x0, y0, x1, y1) for x0, y0, x1, y1 in zip(xs[a], ys[a], xs[b], ys[b])]
                p.setPen(QPen(theme.qcolor(f.color, 0.45), 0.8))
                p.drawLines(lines)

        for f in self._faces:
            x, y, w, h = f.rect
            left = mx(x + w) if self._mirrored else mx(x)
            box = QRectF(left, oy + y * s, w * s, h * s)
            color = QColor(f.color)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(color, 2.0))
            p.drawRoundedRect(box, 10, 10)
            if f.landmarks is not None:
                p.setPen(QPen(QColor(theme.SURFACE), 1.5))
                p.setBrush(QColor(theme.INK))
                for lx, ly in f.landmarks:
                    p.drawEllipse(QPointF(mx(lx), oy + ly * s), 2.6, 2.6)
            if f.pose is not None and f.landmarks is not None:
                yaw, pitch = f.pose
                nx, ny = mx(f.landmarks[2][0]), oy + f.landmarks[2][1] * s
                length = box.width() * 0.45
                dx = -math.sin(math.radians(yaw)) * length if self._mirrored else math.sin(math.radians(yaw)) * length
                dy = -math.sin(math.radians(pitch)) * length
                p.setPen(QPen(QColor(theme.ACCENT), 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(nx, ny), QPointF(nx + dx, ny + dy))
            if f.title:
                self._draw_tag(p, box, f.title, f.subtitle, color, area)

        if self._guide is not None:
            self._draw_guide(p, target, self._guide)
        if self._banner:
            p.setFont(QFont(theme.FONT, 12, QFont.Weight.DemiBold))
            fm = p.fontMetrics()
            bw = fm.horizontalAdvance(self._banner) + 36
            rect = QRectF(target.center().x() - bw / 2, target.bottom() - 58, bw, 40)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor(theme.PAGE, 0.78))
            p.drawRoundedRect(rect, 20, 20)
            p.setPen(QColor(theme.INK))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._banner)

    @staticmethod
    def _draw_tag(p: QPainter, box: QRectF, title: str, subtitle: str, color: QColor, area: QRectF) -> None:
        bold = QFont(theme.FONT, 10, QFont.Weight.DemiBold)
        light = QFont(theme.FONT, 9)
        p.setFont(bold)
        w_title = p.fontMetrics().horizontalAdvance(title)
        p.setFont(light)
        w_sub = p.fontMetrics().horizontalAdvance(subtitle) if subtitle else 0
        width = w_title + (w_sub + 8 if subtitle else 0) + 20
        height = 24.0
        x = min(max(box.left(), area.left() + 4), area.right() - width - 4)
        y = box.top() - height - 6
        if y < area.top() + 4:
            y = box.bottom() + 6
        tag = QRectF(x, y, width, height)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(tag, 12, 12)
        text_color = theme.ink_on(color)
        p.setPen(text_color)
        p.setFont(bold)
        p.drawText(QRectF(tag.left() + 10, tag.top(), w_title + 2, height), Qt.AlignmentFlag.AlignVCenter, title)
        if subtitle:
            p.setFont(light)
            sub = QColor(text_color)
            sub.setAlphaF(0.85)
            p.setPen(sub)
            p.drawText(QRectF(tag.left() + 18 + w_title, tag.top(), w_sub + 2, height), Qt.AlignmentFlag.AlignVCenter, subtitle)

    @staticmethod
    def _draw_guide(p: QPainter, target: QRectF, guide: Guide) -> None:
        eh = target.height() * 0.66
        ew = eh * 0.74
        c = target.center()
        oval = QRectF(c.x() - ew / 2, c.y() - eh / 2 - target.height() * 0.02, ew, eh)
        outside = QPainterPath()
        outside.addRect(target)
        hole = QPainterPath()
        hole.addEllipse(oval)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(theme.PAGE, 0.5))
        p.drawPath(outside.subtracted(hole))
        ring = oval.adjusted(-8, -8, 8, 8)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(theme.qcolor(theme.INK, 0.25), 4))
        p.drawEllipse(ring)
        if guide.progress > 0:
            p.setPen(QPen(QColor(guide.color), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(ring, 90 * 16, -int(360 * 16 * min(1.0, guide.progress)))


# ---------------------------------------------------------------- progress ring
class ProgressRing(QWidget):
    def __init__(self, size: int = 120, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self._text = "0%"
        self._caption = ""
        self._color = theme.ACCENT
        self.setFixedSize(size, size)

    def set_value(self, value: float, text: str | None = None, caption: str = "", color: str | None = None) -> None:
        self._value = max(0.0, min(1.0, value))
        self._text = text if text is not None else f"{int(round(self._value * 100))}%"
        self._caption = caption
        if color:
            self._color = color
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        p.setPen(QPen(QColor(theme.RAISED), 9))
        p.drawEllipse(r)
        if self._value > 0:
            p.setPen(QPen(QColor(self._color), 9, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(r, 90 * 16, -int(360 * 16 * self._value))
        p.setPen(QColor(theme.INK))
        p.setFont(QFont(theme.FONT, max(10, int(self.width() / 7.5)), QFont.Weight.DemiBold))
        text_rect = QRectF(r.left(), r.top(), r.width(), r.height() * (0.62 if self._caption else 1.0))
        p.drawText(text_rect, Qt.AlignmentFlag.AlignHCenter | (Qt.AlignmentFlag.AlignBottom if self._caption else Qt.AlignmentFlag.AlignVCenter), self._text)
        if self._caption:
            p.setPen(QColor(theme.MUTED))
            p.setFont(QFont(theme.FONT, 8))
            p.drawText(QRectF(r.left(), r.top() + r.height() * 0.62, r.width(), 20), Qt.AlignmentFlag.AlignHCenter, self._caption)


# ---------------------------------------------------------------- pill / tile / toggle / rows
class Pill(QWidget):
    """Status indicator: coloured dot + text (colour never carries the meaning alone)."""

    def __init__(self, text: str = "", color: str = theme.MUTED, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text, self._color = text, color
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    def set_state(self, text: str, color: str) -> None:
        if (text, color) != (self._text, self._color):
            self._text, self._color = text, color
            self.updateGeometry()
            self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self._text) + 34, 26)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.qcolor(self._color, 0.16))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setBrush(QColor(self._color))
        p.drawEllipse(QPointF(r.left() + 13, r.center().y()), 4, 4)
        p.setPen(QColor(theme.INK))
        p.drawText(r.adjusted(24, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, self._text)


class StatTile(QFrame):
    def __init__(self, title: str, value: str = "-", caption: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)
        self._title = label(title, "muted")
        self._value = label(value, "big")
        self._caption = label(caption, "small", wrap=True)
        lay.addWidget(self._title)
        lay.addWidget(self._value)
        lay.addWidget(self._caption)

    def set(self, value: str, caption: str | None = None) -> None:
        self._value.setText(value)
        if caption is not None:
            self._caption.setText(caption)


class Toggle(QAbstractButton):
    def __init__(self, checked: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pos = 1.0 if checked else 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._set_pos)
        self.toggled.connect(self._animate)

    def _set_pos(self, v) -> None:
        self._pos = float(v)
        self.update()

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool) -> None:  # noqa: N802 - Qt API
        super().setChecked(on)
        if hasattr(self, "_anim") and self._anim.state() != QVariantAnimation.State.Running:
            self._pos = 1.0 if on else 0.0
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(42, 24)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0, 0, 42, 24).translated((self.width() - 42) / 2, (self.height() - 24) / 2)
        off, on = QColor(theme.BASELINE), QColor(theme.ACCENT)
        t = self._pos
        track = QColor(
            int(off.red() + (on.red() - off.red()) * t),
            int(off.green() + (on.green() - off.green()) * t),
            int(off.blue() + (on.blue() - off.blue()) * t),
        )
        if not self.isEnabled():
            track = QColor(theme.RAISED)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, 12, 12)
        p.setBrush(QBrush(QColor(theme.INK if self.isEnabled() else theme.DISABLED)))
        x = r.left() + 12 + t * (r.width() - 24)
        p.drawEllipse(QPointF(x, r.center().y()), 8, 8)


class SettingRow(QWidget):
    """Title + description on the left, a control on the right."""

    def __init__(self, title: str, description: str, control: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(16)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(label(title))
        if description:
            text.addWidget(label(description, "muted", wrap=True))
        lay.addLayout(text, 1)
        lay.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
        self.control = control


class Meter(QWidget):
    """Thin horizontal bar: fill = value, optional marker (e.g. the acceptance threshold)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self._marker: float | None = None
        self._color = theme.ACCENT
        self.setFixedHeight(10)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set(self, value: float, color: str | None = None, marker: float | None = None) -> None:
        self._value = max(0.0, min(1.0, value))
        self._color = color or self._color
        self._marker = marker
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0, 2, 0, -2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.RAISED))
        p.drawRoundedRect(r, 3, 3)
        if self._value > 0:
            p.setBrush(QColor(self._color))
            p.drawRoundedRect(QRectF(r.left(), r.top(), max(6.0, r.width() * self._value), r.height()), 3, 3)
        if self._marker is not None:
            x = r.left() + r.width() * self._marker
            p.setPen(QPen(QColor(theme.INK), 2))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
