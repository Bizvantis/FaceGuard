"""Colours and the Qt stylesheet.

Neutral dark theme built on the data-viz reference palette (dark mode):
chart/card surface #1a1a19, page #0d0d0d.  The categorical series colours
were validated for colour-vision deficiency on that surface.  Status colours
(good / warning / serious / critical) are reserved for state and always ship
with a text label, never colour alone.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

PAGE = "#0d0d0d"
SURFACE = "#1a1a19"
RAISED = "#242422"
HOVER = "#2c2c2a"
GRID = "#2c2c2a"
BASELINE = "#383835"
INK = "#ffffff"
INK2 = "#c3c2b7"
MUTED = "#898781"
DISABLED = "#5c5b57"

ACCENT = "#3987e5"
ACCENT_HOVER = "#5598e7"
ACCENT_PRESSED = "#2a78d6"

GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# Categorical slots (dark steps), fixed order - colour follows the entity, never its rank.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

FONT = "Segoe UI"
ICON_FONT = "Segoe Fluent Icons"


def qcolor(hex_: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_)
    c.setAlphaF(alpha)
    return c


def ink_on(fill: str | QColor) -> QColor:
    """White or near-black text, whichever is legible on ``fill``."""
    c = QColor(fill)
    lum = 0.2126 * c.redF() ** 2.2 + 0.7152 * c.greenF() ** 2.2 + 0.0722 * c.blueF() ** 2.2
    return QColor(PAGE) if lum > 0.35 else QColor(INK)


STYLESHEET = f"""
* {{ font-family: "{FONT}"; font-size: 10pt; color: {INK}; }}
QMainWindow, QWidget#Root, QStackedWidget#Pages, QWidget#Page {{ background: {PAGE}; }}
QDialog, QMessageBox, QInputDialog, QFileDialog {{ background: {SURFACE}; }}
QWidget#Sidebar {{ background: {SURFACE}; border-right: 1px solid {GRID}; }}
QLabel#AppName {{ font-size: 13pt; font-weight: 600; }}
QLabel#AppTag {{ color: {MUTED}; font-size: 8.5pt; }}

QPushButton#Nav {{
    text-align: left; padding: 9px 12px; border: none; border-radius: 8px;
    color: {INK2}; background: transparent; font-size: 10.5pt;
}}
QPushButton#Nav:hover {{ background: {RAISED}; color: {INK}; }}
QPushButton#Nav:checked {{ background: {RAISED}; color: {INK}; font-weight: 600; }}

QFrame#Card {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 12px; }}
QFrame#Inset {{ background: {PAGE}; border: 1px solid {GRID}; border-radius: 10px; }}
QFrame#Divider {{ background: {GRID}; max-height: 1px; min-height: 1px; border: none; }}

QLabel[role="h1"] {{ font-size: 17pt; font-weight: 600; }}
QLabel[role="h2"] {{ font-size: 12pt; font-weight: 600; }}
QLabel[role="h3"] {{ font-size: 10.5pt; font-weight: 600; }}
QLabel[role="big"] {{ font-size: 20pt; font-weight: 600; }}
QLabel[role="secondary"] {{ color: {INK2}; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel[role="small"] {{ color: {MUTED}; font-size: 8.5pt; }}

QPushButton {{
    background: {RAISED}; border: 1px solid {BASELINE}; border-radius: 8px; padding: 7px 14px;
}}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {BASELINE}; }}
QPushButton:disabled {{ color: {DISABLED}; border-color: {GRID}; background: {SURFACE}; }}
QPushButton[kind="primary"] {{ background: {ACCENT}; border: 1px solid {ACCENT}; font-weight: 600; }}
QPushButton[kind="primary"]:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton[kind="primary"]:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton[kind="primary"]:disabled {{ background: {RAISED}; border-color: {GRID}; color: {DISABLED}; }}
QPushButton[kind="danger"] {{ background: transparent; border: 1px solid {CRITICAL}; color: #e66767; }}
QPushButton[kind="danger"]:hover {{ background: rgba(208, 59, 59, 0.16); }}
QPushButton[kind="ghost"] {{ background: transparent; border: none; color: {INK2}; padding: 6px 8px; }}
QPushButton[kind="ghost"]:hover {{ color: {INK}; background: {RAISED}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {PAGE}; border: 1px solid {BASELINE}; border-radius: 8px; padding: 6px 10px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {BASELINE}; selection-background-color: {RAISED}; outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ spacing: 8px; color: {INK2}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {DISABLED}; background: {PAGE}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}

QProgressBar {{ background: {RAISED}; border: none; border-radius: 4px; max-height: 8px; min-height: 8px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BASELINE}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {DISABLED}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BASELINE}; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ width: 0; height: 0; background: none; }}

QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 8px; padding: 4px; color: {INK2}; }}
QListWidget::item:selected {{ background: {RAISED}; color: {INK}; }}
QListWidget::item:hover {{ background: {SURFACE}; }}

QTableWidget {{ background: transparent; border: none; gridline-color: {GRID}; outline: none; }}
QTableWidget::item {{ padding: 4px 8px; border-bottom: 1px solid {GRID}; color: {INK2}; }}
QHeaderView::section {{
    background: transparent; color: {MUTED}; border: none; border-bottom: 1px solid {BASELINE};
    padding: 6px 8px; font-weight: 600;
}}
QTableCornerButton::section {{ background: transparent; border: none; }}

QToolTip {{ background: {RAISED}; color: {INK}; border: 1px solid {BASELINE}; padding: 6px 8px; border-radius: 6px; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BASELINE}; padding: 4px; border-radius: 8px; }}
QMenu::item {{ padding: 7px 22px; border-radius: 6px; color: {INK2}; }}
QMenu::item:selected {{ background: {RAISED}; color: {INK}; }}
QMenu::item:disabled {{ color: {DISABLED}; }}
QMenu::separator {{ height: 1px; background: {GRID}; margin: 4px 8px; }}
QSlider::groove:horizontal {{ height: 4px; background: {RAISED}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {INK}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }}
"""
