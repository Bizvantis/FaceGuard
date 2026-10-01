"""Icon glyphs (Windows' built-in Segoe Fluent Icons font) and the app logo, drawn in code."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

from faceguard.gui import theme

VIDEO = ""
ADD_PERSON = ""
PEOPLE = ""
FACE = ""
SHIELD = ""
CAMERA = ""
CHART = ""
SETTINGS = ""
CHECK = ""
CROSS = ""
FOLDER = ""
DELETE = ""
EDIT = ""
ADD = ""
PLAY = ""
PAUSE = ""
LOCK = ""
EYE = ""
INFO = ""
ERROR = ""
WARNING = ""
REFRESH = ""
STAR = ""
SPARKLE = ""
CONTACT = ""
KEY = ""
PHOTO = ""

_family: str | None = None


def icon_family() -> str:
    global _family
    if _family is None:
        fams = set(QFontDatabase.families())
        _family = next((f for f in ("Segoe Fluent Icons", "Segoe MDL2 Assets") if f in fams), "")
    return _family


def icon_font(point_size: float = 12.0) -> QFont:
    f = QFont(icon_family() or theme.FONT)
    f.setPointSizeF(point_size)
    return f


def glyph_icon(glyph: str, color: str = theme.INK2, size: int = 20) -> QIcon:
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    f = QFont(icon_family())
    f.setPixelSize(int(size * 1.5))
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, glyph)
    p.end()
    pm.setDevicePixelRatio(2.0)
    return QIcon(pm)


def paint_logo(p: QPainter, rect: QRectF, alert: bool = False) -> None:
    """A rounded shield-blue tile with a face outline inside camera-scan brackets."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = min(rect.width(), rect.height())
    r = QRectF(rect.center().x() - s / 2, rect.center().y() - s / 2, s, s)
    grad = QLinearGradient(r.topLeft(), r.bottomRight())
    if alert:
        grad.setColorAt(0, QColor("#e66767"))
        grad.setColorAt(1, QColor(theme.CRITICAL))
    else:
        grad.setColorAt(0, QColor("#5598e7"))
        grad.setColorAt(1, QColor("#1c5cab"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(r, s * 0.22, s * 0.22)

    pen = QPen(QColor("#ffffff"), max(1.0, s * 0.065), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    m, b = s * 0.2, s * 0.16  # bracket margin and arm length
    x0, y0, x1, y1 = r.left() + m, r.top() + m, r.right() - m, r.bottom() - m
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        path = QPainterPath(QPointF(cx, cy + dy * b))
        path.lineTo(cx, cy)
        path.lineTo(cx + dx * b, cy)
        p.drawPath(path)
    c = r.center()
    fw, fh = s * 0.26, s * 0.34
    p.drawEllipse(QRectF(c.x() - fw / 2, c.y() - fh / 2, fw, fh))
    eye = s * 0.028
    p.setBrush(QColor("#ffffff"))
    p.setPen(Qt.PenStyle.NoPen)
    for ex in (-fw * 0.2, fw * 0.2):
        p.drawEllipse(QPointF(c.x() + ex, c.y() - fh * 0.1), eye, eye)
    p.setPen(QPen(QColor("#ffffff"), max(1.0, s * 0.04), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(c.x() - fw * 0.22, c.y() + fh * 0.02, fw * 0.44, fh * 0.2), 200 * 16, 140 * 16)
    p.restore()


def logo_pixmap(size: int, alert: bool = False) -> QPixmap:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    paint_logo(p, QRectF(0, 0, size, size), alert)
    p.end()
    return QPixmap.fromImage(img)


def app_icon(alert: bool = False) -> QIcon:
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        icon.addPixmap(logo_pixmap(size, alert))
    return icon
