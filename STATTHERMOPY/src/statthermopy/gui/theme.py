"""Design system for the StatThermoPy Qt GUI.

A tiny, self-contained theming layer: a pair of semantic color palettes (light / dark), a
Qt Style Sheet builder driven by those tokens, a matching :class:`QPalette` for the native
bits QSS does not fully reach (spinbox arrows, tooltips, the matplotlib toolbar), and a
factory of small vector glyphs drawn with :class:`QPainter` so they recolor with the theme.

PySide6-only and imported lazily by :mod:`statthermopy.gui.mainwindow` only — it never enters
the core import path.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QApplication

__all__ = ["Palette", "LIGHT", "DARK", "qss", "apply_qt_palette", "make_icons", "detect_dark"]


@dataclass(frozen=True)
class Palette:
    """Semantic color tokens used by the QSS and the icon factory."""

    bg: str
    surface: str
    surface_alt: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_disabled: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_text: str
    accent_soft: str
    success: str
    success_soft: str
    danger: str
    danger_soft: str
    radius: str
    radius_card: str


LIGHT = Palette(
    bg="#f4f6f9",
    surface="#ffffff",
    surface_alt="#eef2f7",
    border="#e2e8f0",
    border_strong="#cbd5e1",
    text="#0f172a",
    text_muted="#64748b",
    text_disabled="#94a3b8",
    accent="#2563eb",
    accent_hover="#1d4ed8",
    accent_pressed="#1e40af",
    accent_text="#ffffff",
    accent_soft="#dbeafe",
    success="#16a34a",
    success_soft="#dcfce7",
    danger="#dc2626",
    danger_soft="#fee2e2",
    radius="8px",
    radius_card="12px",
)

DARK = Palette(
    bg="#0b1220",
    surface="#111827",
    surface_alt="#1e293b",
    border="#334155",
    border_strong="#475569",
    text="#e2e8f0",
    text_muted="#94a3b8",
    text_disabled="#475569",
    accent="#3b82f6",
    accent_hover="#60a5fa",
    accent_pressed="#2563eb",
    accent_text="#ffffff",
    accent_soft="#1e3a8a",
    success="#22c55e",
    success_soft="#14532d",
    danger="#ef4444",
    danger_soft="#7f1d1d",
    radius="8px",
    radius_card="12px",
)


def qss(p: Palette) -> str:
    """Build a Qt Style Sheet from the semantic tokens in ``p``."""
    return f"""
    QWidget {{
        background-color: {p.bg};
        color: {p.text};
        font-size: 10pt;
    }}
    QMainWindow, QDialog {{ background-color: {p.bg}; }}
    QFrame#Card {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {p.radius_card};
    }}
    QGroupBox {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {p.radius_card};
        margin-top: 14px;
        padding: 14px 12px 10px 12px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {p.accent};
        background-color: {p.surface};
    }}
    QLabel {{ background: transparent; color: {p.text}; }}
    QLabel[role="hint"] {{ color: {p.text_muted}; font-size: 9pt; }}
    QLabel[verdict="idle"] {{
        background-color: {p.surface_alt};
        color: {p.text_muted};
        border: 1px solid {p.border};
        border-radius: {p.radius};
        padding: 6px 10px;
    }}
    QLabel[verdict="pass"] {{
        background-color: {p.success_soft};
        color: {p.success};
        border: 1px solid {p.success};
        border-radius: {p.radius};
        padding: 6px 10px;
        font-weight: 600;
    }}
    QLabel[verdict="fail"] {{
        background-color: {p.danger_soft};
        color: {p.danger};
        border: 1px solid {p.danger};
        border-radius: {p.radius};
        padding: 6px 10px;
        font-weight: 600;
    }}
    QPushButton {{
        background-color: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: {p.radius};
        padding: 6px 14px;
        min-height: 22px;
    }}
    QPushButton:hover {{ border: 1px solid {p.border_strong}; background-color: {p.surface_alt}; }}
    QPushButton:pressed {{ background-color: {p.surface_alt}; border: 1px solid {p.accent}; }}
    QPushButton:disabled {{ color: {p.text_disabled}; border: 1px solid {p.border}; background-color: {p.surface_alt}; }}
    QPushButton[primary="true"] {{
        background-color: {p.accent};
        color: {p.accent_text};
        border: 1px solid {p.accent};
        font-weight: 600;
    }}
    QPushButton[primary="true"]:hover {{ background-color: {p.accent_hover}; border: 1px solid {p.accent_hover}; }}
    QPushButton[primary="true"]:pressed {{ background-color: {p.accent_pressed}; border: 1px solid {p.accent_pressed}; }}
    QPushButton[primary="true"]:disabled {{ background-color: {p.surface_alt}; color: {p.text_disabled}; border: 1px solid {p.border}; }}
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QTextEdit {{
        background-color: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: {p.radius};
        padding: 4px 8px;
        selection-background-color: {p.accent_soft};
        selection-color: {p.text};
    }}
    QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover, QLineEdit:hover, QTextEdit:hover {{
        border: 1px solid {p.border_strong};
    }}
    QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QTextEdit:focus {{
        border: 1px solid {p.accent};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{
        background-color: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        selection-background-color: {p.accent_soft};
        selection-color: {p.text};
        outline: 0;
    }}
    QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {p.border_strong};
        border-radius: 4px;
        background-color: {p.surface};
    }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border: 1px solid {p.accent}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {p.accent};
        border: 1px solid {p.accent};
    }}
    QTableWidget {{
        background-color: {p.surface};
        alternate-background-color: {p.surface_alt};
        color: {p.text};
        gridline-color: {p.border};
        border: 1px solid {p.border};
        border-radius: {p.radius};
        selection-background-color: {p.accent_soft};
        selection-color: {p.text};
        outline: 0;
    }}
    QHeaderView::section {{
        background-color: {p.surface_alt};
        color: {p.text};
        padding: 6px 8px;
        border: none;
        border-right: 1px solid {p.border};
        border-bottom: 1px solid {p.border};
        font-weight: 600;
    }}
    QTabWidget::pane {{
        border: 1px solid {p.border};
        border-radius: {p.radius};
        top: -1px;
        background-color: {p.bg};
    }}
    QTabBar::tab {{
        background-color: {p.surface_alt};
        color: {p.text_muted};
        padding: 8px 18px;
        margin-right: 2px;
        border: 1px solid {p.border};
        border-bottom: none;
        border-top-left-radius: {p.radius};
        border-top-right-radius: {p.radius};
    }}
    QTabBar::tab:selected {{
        background-color: {p.surface};
        color: {p.text};
        border-color: {p.border};
        border-top: 3px solid {p.accent};
    }}
    QTabBar::tab:hover:!selected {{ color: {p.text}; }}
    QMenuBar {{ background-color: {p.bg}; color: {p.text}; border-bottom: 1px solid {p.border}; }}
    QMenuBar::item:selected {{ background-color: {p.accent_soft}; border-radius: 4px; }}
    QMenu {{
        background-color: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        padding: 4px;
    }}
    QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
    QMenu::item:selected {{ background-color: {p.accent_soft}; color: {p.text}; }}
    QMenu::separator {{ height: 1px; background-color: {p.border}; margin: 4px 8px; }}
    QStatusBar {{ background-color: {p.surface}; color: {p.text_muted}; border-top: 1px solid {p.border}; }}
    QToolTip {{ background-color: {p.surface}; color: {p.text}; border: 1px solid {p.border}; border-radius: {p.radius}; padding: 4px 8px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
        background: {p.surface_alt};
        border: 1px solid {p.border};
        border-radius: 4px;
        min-height: 24px; min-width: 24px;
    }}
    QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {p.border_strong}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ border: none; background: transparent; width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    """


def _qcolor(hexstr: str) -> QColor:
    return QColor(hexstr)


def apply_qt_palette(app: QApplication, p: Palette) -> None:
    """Set a :class:`QPalette` on ``app`` matching the tokens (covers native widgets)."""
    pal = QPalette()
    base = _qcolor(p.bg)
    surface = _qcolor(p.surface)
    text = _qcolor(p.text)
    muted = _qcolor(p.text_muted)
    border = _qcolor(p.border)
    accent = _qcolor(p.accent)
    accent_text = _qcolor(p.accent_text)

    pal.setColor(QPalette.ColorRole.Window, base)
    pal.setColor(QPalette.ColorRole.Base, surface)
    pal.setColor(QPalette.ColorRole.AlternateBase, _qcolor(p.surface_alt))
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Button, surface)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.PlaceholderText, muted)
    pal.setColor(QPalette.ColorRole.ToolTipBase, surface)
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, accent_text)
    pal.setColor(QPalette.ColorRole.Light, surface)
    pal.setColor(QPalette.ColorRole.Mid, border)
    pal.setColor(QPalette.ColorRole.Dark, border)
    pal.setColor(QPalette.ColorRole.Shadow, border)

    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, _qcolor(p.text_disabled))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, _qcolor(p.text_disabled))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, _qcolor(p.text_disabled))
    app.setPalette(pal)


def _glyph(size: int, draw) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    qp = QPainter(pix)
    qp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    try:
        draw(qp)
    finally:
        qp.end()
    return pix


def _draw_plus(qp: QPainter, color: QColor) -> None:
    qp.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    m = 5
    qp.drawLine(m, 10, 20 - m, 10)
    qp.drawLine(10, m, 10, 20 - m)


def _draw_minus(qp: QPainter, color: QColor) -> None:
    qp.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    m = 5
    qp.drawLine(m, 10, 20 - m, 10)


def _draw_play(qp: QPainter, color: QColor) -> None:
    qp.setPen(Qt.PenStyle.NoPen)
    qp.setBrush(color)
    qp.drawPolygon([QPointF(8, 4), QPointF(8, 16), QPointF(17, 10)])


def _draw_check(qp: QPainter, color: QColor) -> None:
    qp.setPen(QPen(color, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    qp.drawLine(4, 11, 9, 16)
    qp.drawLine(9, 16, 17, 5)


def _draw_chart(qp: QPainter, color: QColor) -> None:
    qp.setPen(Qt.PenStyle.NoPen)
    qp.setBrush(color)
    qp.drawRect(QRectF(4, 11, 3, 6))
    qp.drawRect(QRectF(9, 7, 3, 10))
    qp.drawRect(QRectF(14, 4, 3, 13))


def _draw_refresh(qp: QPainter, color: QColor) -> None:
    qp.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    # open arc
    qp.drawArc(QRectF(4, 4, 12, 12), 30 * 16, 270 * 16)
    # arrow head
    qp.setBrush(color)
    qp.setPen(Qt.PenStyle.NoPen)
    qp.drawPolygon([QPointF(15, 5), QPointF(17, 9), QPointF(12, 8)])


def make_icons(p: Palette) -> dict[str, QIcon]:
    """Build the set of theme-colored vector glyphs used by the main window."""
    text = _qcolor(p.text)
    accent_text = _qcolor(p.accent_text)

    def icon(draw, color) -> QIcon:
        return QIcon(_glyph(20, lambda qp: draw(qp, color)))

    return {
        "plus": icon(_draw_plus, text),
        "minus": icon(_draw_minus, text),
        "play": icon(_draw_play, accent_text),
        "check": icon(_draw_check, accent_text),
        "chart": icon(_draw_chart, text),
        "refresh": icon(_draw_refresh, text),
    }


def detect_dark() -> bool:
    """True when the platform reports a dark color scheme (else light, incl. Unknown)."""
    app = QApplication.instance()
    if app is None:
        return False
    try:
        scheme = app.styleHints().colorScheme()
    except AttributeError:  # pragma: no cover - very old PySide6
        return False
    return scheme == Qt.ColorScheme.Dark


def default_font() -> QFont:
    """A clean, legible default application font."""
    return QFont("Segoe UI", 10)
