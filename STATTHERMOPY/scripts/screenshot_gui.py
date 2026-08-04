"""Capture StatThermoPy GUI screenshots for the README (headless, offscreen Qt).

Runs the PySide6 GUI on the ``offscreen`` Qt platform, drives each tab to a populated
state, and saves a PNG of the whole window via ``QWidget.grab()``. No display needed.

    python scripts/screenshot_gui.py

Outputs to ``docs/images/gui_*.png``.
"""
from __future__ import annotations

import os
import sys

# Headless Qt + matplotlib backend BEFORE any widget/Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from PySide6.QtWidgets import QApplication  # noqa: E402

from statthermopy.gui.mainwindow import StatThermoPyWindow  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)


def grab(win, name: str) -> None:
    """Process pending paint events and save a PNG of the whole window."""
    app = QApplication.instance()
    app.processEvents()
    pm = win.grab()
    path = os.path.abspath(os.path.join(OUT, name))
    pm.save(path, "PNG")
    print(f"saved {path}  ({pm.width()}x{pm.height()})")


def select_items(lw, texts: list[str]) -> None:
    """Select the list items whose text matches `texts` (multi-selection)."""
    from PySide6.QtWidgets import QListWidgetItem

    lw.clearSelection()
    for i in range(lw.count()):
        it: QListWidgetItem = lw.item(i)
        it.setSelected(it.text() in texts)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = StatThermoPyWindow()
    win.resize(1120, 780)
    win.show()
    app.processEvents()

    tabs = win._tabs

    # --- Properties tab (light) ---------------------------------------
    tabs.setCurrentIndex(0)
    win.gas_combo.setCurrentText("N2")
    win.T_spin.setValue(298.15)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    app.processEvents()
    grab(win, "gui_properties.png")

    # --- Plot tab (light): molar Cp of N2 vs T -------------------------
    tabs.setCurrentIndex(1)
    win.plot_prop.setCurrentText("Cp_m")
    win.plot_tmin.setValue(200.0)
    win.plot_tmax.setValue(1500.0)
    win.plot_npts.setValue(120)
    win.plot_p.setValue(101325.0)
    win._on_plot()
    app.processEvents()
    grab(win, "gui_plot.png")

    # --- Transport tab (light): N2 point eval + mu/k vs T --------------
    tabs.setCurrentIndex(2)
    win.transport_species.setCurrentText("N2")
    win.transport_T.setValue(300.0)
    win.transport_P.setValue(101325.0)
    win._on_transport_compute()
    app.processEvents()
    win.transport_mode.setCurrentText("vs T")
    select_items(win.transport_props, ["mu", "k"])
    win.transport_tmin.setValue(300.0)
    win.transport_tmax.setValue(1500.0)
    win.transport_npts.setValue(60)
    win.transport_pmin.setValue(101325.0)
    win.transport_pmax.setValue(101325.0)
    win._on_transport_plot()
    app.processEvents()
    grab(win, "gui_transport.png")

    # --- Validate tab (light): N2 / Cp --------------------------------
    tabs.setCurrentIndex(3)
    win.val_species.setCurrentText("N2")
    win.val_prop.setCurrentText("Cp")
    win._on_validate()
    app.processEvents()
    grab(win, "gui_validate.png")

    # --- Properties tab, dark theme -----------------------------------
    tabs.setCurrentIndex(0)
    win._apply_theme("dark")
    app.processEvents()
    grab(win, "gui_dark.png")

    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())