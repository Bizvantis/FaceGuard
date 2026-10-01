"""Charts for the Model Lab, painted with QPainter.

Follows the data-viz rules used across the app: thin marks (2 px lines, 8 px
dots with a 2 px surface ring, bars <= 24 px with 4 px rounded data-ends),
recessive hairline grid, a legend whenever there are two or more series,
text in ink colours (never the series colour), and hover tooltips with
generous hit areas.  The categorical palette was validated for colour-vision
deficiency on the dark surface; scatter plots only validate three colours
against each other, so with more people the map switches to "highlight one,
grey the rest" plus direct name labels.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from faceguard.gui import theme

SCATTER_COLOR_LIMIT = 3


def _font(size: float = 9, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont(theme.FONT)
    f.setPointSizeF(size)
    f.setWeight(weight)
    return f


def _nice_step(span: float, target_ticks: int = 4) -> float:
    raw = span / max(1, target_ticks)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _halo_text(p: QPainter, pos: QPointF, text: str, font: QFont, color: str) -> None:
    path = QPainterPath()
    path.addText(pos, font, text)
    p.setPen(QPen(QColor(theme.SURFACE), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(color))
    p.drawPath(path)


class _Chart(QWidget):
    def __init__(self, parent: QWidget | None = None, min_height: int = 240) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.empty_text = "Not enough data yet"

    def _begin(self) -> QPainter:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        return p

    def _empty(self, p: QPainter) -> None:
        p.setPen(QColor(theme.MUTED))
        p.setFont(_font(10))
        p.drawText(QRectF(self.rect()), Qt.AlignmentFlag.AlignCenter, self.empty_text)

    def _legend(self, p: QPainter, items: list[tuple[str, str, str]], x: float, y: float) -> float:
        """items = (swatch kind 'dot'|'line', colour, text). Returns the bottom y."""
        p.setFont(_font(9))
        fm = p.fontMetrics()
        cx = x
        for kind, color, text in items:
            w = fm.horizontalAdvance(text)
            if cx + w + 30 > self.width() - 8 and cx > x:
                cx = x
                y += 20
            if kind == "line":
                p.setPen(QPen(QColor(color), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(cx, y + 8), QPointF(cx + 14, y + 8))
                tx = cx + 20
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(color))
                p.drawEllipse(QPointF(cx + 5, y + 8), 4.5, 4.5)
                tx = cx + 14
            p.setPen(QColor(theme.INK2))
            p.drawText(QRectF(tx, y, w + 4, 16), Qt.AlignmentFlag.AlignVCenter, text)
            cx = tx + w + 18
        return y + 20


class ScatterChart(_Chart):
    """2-D map of face embeddings (PCA / t-SNE); one dot per enrolled sample."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, 280)
        self._pts = np.zeros((0, 2))
        self._labels = np.zeros(0, dtype=int)
        self._names: list[str] = []
        self._highlight: int | None = None
        self._screen = np.zeros((0, 2))
        self._hover: int | None = None

    def set_data(self, points: np.ndarray, labels: np.ndarray, names: list[str], highlight: int | None = None) -> None:
        self._pts, self._labels, self._names, self._highlight = np.asarray(points, float), np.asarray(labels, int), names, highlight
        self._hover = None
        self.update()

    @property
    def highlight_mode(self) -> bool:
        return len(self._names) > SCATTER_COLOR_LIMIT

    def _color(self, label_: int) -> tuple[str, float]:
        if not self.highlight_mode:
            return theme.SERIES[label_ % len(theme.SERIES)], 1.0
        if label_ == self._highlight:
            return theme.SERIES[0], 1.0
        return theme.MUTED, 0.5

    def paintEvent(self, _event) -> None:
        p = self._begin()
        if len(self._pts) < 2:
            self._empty(p)
            return
        if self.highlight_mode:
            name = self._names[self._highlight] if self._highlight is not None else "none"
            legend = [("dot", theme.SERIES[0], f"{name} (highlighted)"), ("dot", theme.MUTED, "Everyone else")]
        else:
            legend = [("dot", self._color(i)[0], n) for i, n in enumerate(self._names)]
        top = self._legend(p, legend, 12, 8) + 6
        plot = QRectF(24, top + 8, self.width() - 48, self.height() - top - 32)
        lo, hi = self._pts.min(axis=0), self._pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        scale = min(plot.width() / span[0], plot.height() / span[1])  # equal aspect: distances stay honest
        cx, cy = (lo + hi) / 2
        sx = plot.center().x() + (self._pts[:, 0] - cx) * scale
        sy = plot.center().y() - (self._pts[:, 1] - cy) * scale
        self._screen = np.stack([sx, sy], axis=1)

        order = np.argsort([1 if (self.highlight_mode and lab == self._highlight) else 0 for lab in self._labels], kind="stable")
        for i in order:  # highlighted person drawn last (on top)
            color, alpha = self._color(int(self._labels[i]))
            p.setPen(QPen(QColor(theme.SURFACE), 2))
            p.setBrush(theme.qcolor(color, alpha))
            r = 5.5 if i == self._hover else 4.0
            p.drawEllipse(QPointF(sx[i], sy[i]), r, r)

        font = _font(9, QFont.Weight.DemiBold)
        for j, name in enumerate(self._names):  # direct labels at each cluster centre
            mask = self._labels == j
            if not mask.any():
                continue
            mxp, myp = sx[mask].mean(), sy[mask].min() - 10
            fm_w = p.fontMetrics().horizontalAdvance(name)
            _halo_text(p, QPointF(mxp - fm_w / 2, max(top + 14, myp)), name, font, theme.INK2)

        p.setPen(QColor(theme.MUTED))
        p.setFont(_font(8.5))
        p.drawText(QRectF(12, self.height() - 22, self.width() - 24, 18), Qt.AlignmentFlag.AlignLeft,
                   "Each dot is one face sample. Dots close together look alike to the neural network.")

    def mouseMoveEvent(self, e) -> None:
        if len(self._screen) == 0:
            return
        pos = np.array([e.position().x(), e.position().y()])
        d = np.hypot(*(self._screen - pos).T)
        i = int(np.argmin(d))
        hover = i if d[i] <= 24 else None  # generous hit area, not pixel hunting
        if hover != self._hover:
            self._hover = hover
            self.update()
        if hover is None:
            QToolTip.hideText()
        else:
            n = int(np.sum(self._labels[: hover + 1] == self._labels[hover]))
            QToolTip.showText(e.globalPosition().toPoint(), f"{self._names[self._labels[hover]]}\nsample #{n}", self)

    def leaveEvent(self, _e) -> None:
        self._hover = None
        self.update()


class HistogramChart(_Chart):
    """Overlaid distributions of cosine similarity (genuine vs impostor pairs) with the decision threshold."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, 240)
        self._series: list[tuple[str, np.ndarray, str]] = []
        self._threshold: float | None = None
        self._edges = np.linspace(-0.2, 1.0, 49)
        self._shares: list[np.ndarray] = []
        self._plot = QRectF()
        self._hover_bin: int | None = None

    def set_data(self, series: list[tuple[str, np.ndarray, str]], threshold: float | None) -> None:
        self._series = [(n, np.asarray(v, float), c) for n, v, c in series if len(v)]
        self._threshold = threshold
        self._shares = []
        for _, v, _ in self._series:
            counts, _ = np.histogram(np.clip(v, self._edges[0], self._edges[-1] - 1e-9), self._edges)
            self._shares.append(counts / max(1, len(v)))
        self.update()

    def paintEvent(self, _event) -> None:
        p = self._begin()
        if not self._series:
            self._empty(p)
            return
        legend = [("line", c, f"{n} ({len(v):,} pairs)") for n, v, c in self._series]
        top = self._legend(p, legend, 12, 8) + 4
        left, bottom = 46, 28
        plot = QRectF(left, top + 10, self.width() - left - 16, self.height() - top - 10 - bottom)
        self._plot = plot
        ymax = max(float(s.max()) for s in self._shares)
        step = _nice_step(ymax)
        ytop = step * math.ceil(ymax / step) if ymax > 0 else 1.0
        x0, x1 = self._edges[0], self._edges[-1]

        def X(v):
            return plot.left() + (v - x0) / (x1 - x0) * plot.width()

        def Y(v):
            return plot.bottom() - v / ytop * plot.height()

        p.setFont(_font(8.5))
        decimals = 0 if abs(step * 100 - round(step * 100)) < 1e-9 else 1  # 2.5% steps need a decimal
        for k in range(int(round(ytop / step)) + 1):  # hairline grid + y ticks
            yv = k * step
            y = Y(yv)
            p.setPen(QPen(QColor(theme.GRID if k else theme.BASELINE), 1))
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(0, y - 8, left - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{yv * 100:.{decimals}f}%")
        for xv in (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            p.drawText(QRectF(X(xv) - 20, plot.bottom() + 6, 40, 16), Qt.AlignmentFlag.AlignCenter, f"{xv:.1f}")

        for (name, _, color), shares in zip(self._series, self._shares):
            path = QPainterPath(QPointF(X(x0), Y(0)))
            for b, share in enumerate(shares):
                path.lineTo(X(self._edges[b]), Y(share))
                path.lineTo(X(self._edges[b + 1]), Y(share))
            path.lineTo(X(x1), Y(0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor(color, 0.10))
            p.drawPath(path)
            p.setPen(QPen(QColor(color), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        if self._threshold is not None:
            tx = X(self._threshold)
            p.setPen(QPen(QColor(theme.INK), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(tx, plot.top()), QPointF(tx, plot.bottom()))
            _halo_text(p, QPointF(tx + 6, plot.top() + 12), f"threshold {self._threshold:.2f}", _font(8.5), theme.INK)

        if self._hover_bin is not None:
            cx = X((self._edges[self._hover_bin] + self._edges[self._hover_bin + 1]) / 2)
            p.setPen(QPen(theme.qcolor(theme.INK, 0.35), 1))
            p.drawLine(QPointF(cx, plot.top()), QPointF(cx, plot.bottom()))

    def mouseMoveEvent(self, e) -> None:
        if not self._series or self._plot.isEmpty():
            return
        x = e.position().x()
        if not (self._plot.left() <= x <= self._plot.right()):
            self._hover_bin = None
            QToolTip.hideText()
            self.update()
            return
        frac = (x - self._plot.left()) / self._plot.width()
        b = int(min(len(self._edges) - 2, max(0, frac * (len(self._edges) - 1))))
        self._hover_bin = b
        lines = [f"similarity {self._edges[b]:.3f} - {self._edges[b + 1]:.3f}"]
        for (name, _, _), shares in zip(self._series, self._shares):
            lines.append(f"{name}: {shares[b] * 100:.1f}% of pairs")
        QToolTip.showText(e.globalPosition().toPoint(), "\n".join(lines), self)
        self.update()

    def leaveEvent(self, _e) -> None:
        self._hover_bin = None
        self.update()


class BarChart(_Chart):
    """Horizontal bars for one measure (single series -> one colour, no legend; value at each tip)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, 120)
        self._items: list[tuple[str, float, str]] = []
        self._rows: list[QRectF] = []

    def set_data(self, items: list[tuple[str, float, str]]) -> None:
        """items = (label, value in 0..1, tooltip)."""
        self._items = items
        self.setMinimumHeight(max(120, 34 * len(items) + 16))
        self.update()

    def paintEvent(self, _event) -> None:
        p = self._begin()
        if not self._items:
            self._empty(p)
            return
        p.setFont(_font(9))
        fm = p.fontMetrics()
        label_w = max(fm.horizontalAdvance(t) for t, _, _ in self._items) + 14
        value_w = fm.horizontalAdvance("100.0%") + 10
        x0 = 8 + label_w
        width = max(10.0, self.width() - x0 - value_w - 8)
        row_h = 34.0
        self._rows = []
        for i, (text, value, _) in enumerate(self._items):
            y = 8 + i * row_h
            p.setPen(QColor(theme.INK2))
            p.drawText(QRectF(8, y, label_w - 10, row_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)
            bar_h = 16.0
            by = y + (row_h - bar_h) / 2
            w = max(2.0, width * max(0.0, min(1.0, value)))
            path = QPainterPath()  # square at the baseline, 4 px rounded data-end
            path.moveTo(x0, by)
            path.lineTo(x0 + w - 4, by)
            path.quadTo(x0 + w, by, x0 + w, by + 4)
            path.lineTo(x0 + w, by + bar_h - 4)
            path.quadTo(x0 + w, by + bar_h, x0 + w - 4, by + bar_h)
            path.lineTo(x0, by + bar_h)
            path.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(theme.SERIES[0]))
            p.drawPath(path)
            p.setPen(QColor(theme.INK))
            p.drawText(QRectF(x0 + w + 6, y, value_w, row_h), Qt.AlignmentFlag.AlignVCenter, f"{value * 100:.1f}%")
            self._rows.append(QRectF(0, y, self.width(), row_h))
        p.setPen(QPen(QColor(theme.BASELINE), 1))
        p.drawLine(QPointF(x0, 6), QPointF(x0, 8 + len(self._items) * row_h))

    def mouseMoveEvent(self, e) -> None:
        for (text, value, tip), row in zip(self._items, self._rows):
            if row.contains(e.position()):
                QToolTip.showText(e.globalPosition().toPoint(), f"{text}: {value * 100:.2f}%\n{tip}", self)
                return
        QToolTip.hideText()
