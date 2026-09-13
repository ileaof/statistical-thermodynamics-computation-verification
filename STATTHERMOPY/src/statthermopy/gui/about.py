"""Application identity and the About dialog.

Kept out of :mod:`statthermopy.gui.mainwindow` so the identity text has one home and the main
window stays about layout. Nothing here touches the scientific core.

The version is read from the installed package metadata and is simply omitted when there is
none -- it is never invented.
"""

from __future__ import annotations

import platform
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from .resources import APP_ICON, icon, pixmap

#: Product name, used for the window title and the typographic logo.
APP_NAME = "StatThermoPy"
#: One-line subtitle shown under the name.
APP_TAGLINE = "Statistical Thermodynamics in Python"

AUTHOR = "Prof. Ivaldo Leão Ferreira"
AFFILIATION_LINES = (
    "Faculty of Mechanical Engineering - FEM",
    "Institute of Technology - ITEC",
    "Federal University of Pará - UFPA",
)
#: Compact form for the status bar and window title.
AFFILIATION_SHORT = "FEM - ITEC - UFPA"

DESCRIPTION = (
    "StatThermoPy is a scientific computing package for the analysis of statistical "
    "thermodynamics, thermophysical properties, gas mixtures, humid air and transport "
    "properties using Python.\n\n"
    "The software is intended for research, teaching and computational engineering "
    "applications."
)

COPYRIGHT = f"© {AUTHOR}"


def package_version() -> str | None:
    """Installed version of the package, or ``None`` when no metadata is available."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8 only
        return None
    try:
        return version("statthermopy")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        pass
    try:  # fall back to the package attribute, if it declares one
        import statthermopy

        return getattr(statthermopy, "__version__", None)
    except Exception:  # pragma: no cover - defensive
        return None


def environment_lines() -> list[str]:
    """Runtime facts worth quoting in a bug report."""
    lines = [f"Python {platform.python_version()} ({sys.platform})"]
    try:
        from PySide6 import __version__ as pyside_version

        lines.append(f"PySide6 {pyside_version}")
    except Exception:  # pragma: no cover - PySide6 is present if this module loaded
        pass
    try:
        import matplotlib

        lines.append(f"matplotlib {matplotlib.__version__}")
    except Exception:  # pragma: no cover - matplotlib is a hard dependency
        pass
    lines.append(platform.platform(terse=True))
    return lines


def about_text() -> str:
    """Plain-text form of the About content, for a console or a bug report."""
    parts = [APP_NAME, APP_TAGLINE, ""]
    release = package_version()
    if release:
        parts += [f"Version {release}", ""]
    parts += [AUTHOR, *AFFILIATION_LINES, "", DESCRIPTION, "", *environment_lines(), "",
              COPYRIGHT]
    return "\n".join(parts)


class AboutDialog(QDialog):
    """Modal About window: identity, authorship, description and runtime environment.

    The bundled application mark leads, with the name and tagline beside it. If the artwork is
    missing the layout falls back to the typography alone, so the dialog stays presentable in
    an installation where the image did not travel.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setWindowIcon(icon(APP_ICON))

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 20)
        root.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(18)
        mark = pixmap(APP_ICON, 88)
        if not mark.isNull():
            logo = QLabel()
            logo.setPixmap(mark)
            logo.setAlignment(Qt.AlignTop)
            header.addWidget(logo, 0, Qt.AlignTop)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        name = QLabel(APP_NAME)
        name.setProperty("role", "aboutTitle")
        titles.addWidget(name)

        tagline = QLabel(APP_TAGLINE)
        tagline.setProperty("role", "aboutTagline")
        titles.addWidget(tagline)
        titles.addStretch()
        header.addLayout(titles, 1)
        root.addLayout(header)

        release = package_version()
        if release:
            root.addSpacing(10)
            version_label = QLabel(f"Version {release}")
            version_label.setProperty("role", "hint")
            root.addWidget(version_label)

        root.addSpacing(16)
        root.addWidget(self._rule())
        root.addSpacing(16)

        author = QLabel(AUTHOR)
        author.setProperty("role", "aboutAuthor")
        root.addWidget(author)

        affiliation = QLabel("\n".join(AFFILIATION_LINES))
        affiliation.setProperty("role", "hint")
        root.addWidget(affiliation)

        root.addSpacing(16)
        description = QLabel(DESCRIPTION)
        description.setWordWrap(True)
        root.addWidget(description)

        root.addSpacing(16)
        root.addWidget(self._rule())
        root.addSpacing(12)

        environment = QLabel("\n".join(environment_lines()))
        environment.setProperty("role", "hint")
        environment.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(environment)

        root.addSpacing(14)
        copyright_label = QLabel(COPYRIGHT)
        copyright_label.setProperty("role", "hint")
        root.addWidget(copyright_label)

        root.addSpacing(10)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    @staticmethod
    def _rule() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setProperty("role", "rule")
        return line
