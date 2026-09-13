"""Bundled image resources for the GUI.

The PNGs live under ``statthermopy/gui/icons/`` so they travel with the installed wheel the
same way the YAML species database does. Everything here degrades quietly: a missing or
unreadable file yields a null :class:`QIcon`, which Qt simply draws as nothing, so the GUI
never fails to start over an image.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap

#: Directory holding the bundled artwork.
ICON_DIR = Path(__file__).resolve().parent / "icons"

#: Application mark, used for the window, the taskbar and the About dialog.
APP_ICON = "app"

#: Tab name -> artwork stem. Tabs with no natural match are left to the generated vector
#: glyphs in :mod:`statthermopy.gui.theme` rather than given a misleading picture.
TAB_ICONS = {
    "Properties": "properties",
    "Plot": "plot",
    "Transport": "transport",
    "Humid Air": "humid-air",
    "Thermodynamic Comparisons": "app",
    "Air Transport": "mixture-transport",
}


@lru_cache(maxsize=None)
def icon(stem: str) -> QIcon:
    """Return the bundled icon ``stem``, or a null icon when it is not available."""
    path = ICON_DIR / f"{stem}.png"
    if not path.is_file():
        return QIcon()
    return QIcon(str(path))


@lru_cache(maxsize=None)
def pixmap(stem: str, size: int) -> QPixmap:
    """Return the bundled artwork ``stem`` scaled to ``size`` square, or a null pixmap."""
    from PySide6.QtCore import Qt

    path = ICON_DIR / f"{stem}.png"
    if not path.is_file():
        return QPixmap()
    return QPixmap(str(path)).scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def available() -> list[str]:
    """Stems of the artwork actually present, for diagnostics and tests."""
    if not ICON_DIR.is_dir():
        return []
    return sorted(p.stem for p in ICON_DIR.glob("*.png"))
