"""Entry point for the StatThermoPy Qt GUI.

The GUI is an **optional** layer that wraps the pure-statistical-mechanics core; it adds no
physics. Install it with ``pip install statthermopy[gui]`` and launch with
``statthermopy-gui``.

Backend note
------------
:mod:`statthermopy.plots.plotting` lazily selects the headless ``Agg`` matplotlib backend on
first use, which is incompatible with an interactive Qt canvas. We therefore set the ``QtAgg``
backend and import ``matplotlib.pyplot`` *before* anything in :mod:`statthermopy.plots` is
touched. Once pyplot is imported, ``matplotlib.use(...)`` calls elsewhere are no-ops, so the
embedded canvases keep working.
"""

from __future__ import annotations

import sys


def _ensure_qt_backend() -> None:
    """Select the matplotlib ``QtAgg`` backend and import pyplot, before plotting is used."""
    import matplotlib

    matplotlib.use("QtAgg")
    import matplotlib.pyplot as _plt  # noqa: F401  (force backend initialisation)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GUI event loop
    """Launch the StatThermoPy GUI. Requires PySide6 (``pip install statthermopy[gui]``)."""
    try:
        import PySide6  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PySide6 is required for the StatThermoPy GUI.\n"
            "Install it with:  pip install statthermopy[gui]"
        ) from exc

    _ensure_qt_backend()

    from PySide6.QtWidgets import QApplication

    from .mainwindow import StatThermoPyWindow
    from .theme import default_font

    args = list(argv) if argv is not None else sys.argv
    app = QApplication.instance() or QApplication(args)
    app.setFont(default_font())
    win = StatThermoPyWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
