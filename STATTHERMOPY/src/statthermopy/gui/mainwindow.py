"""StatThermoPy main window (PySide6).

A thin Qt layer over the pure-statistical-mechanics core. The window holds the same session
state as the CLI (molecule *or* mixture, T/P/V/n/m) and calls the same public API
(:class:`~statthermopy.thermodynamics.Thermodynamics`, :func:`~statthermopy.mixture.IdealGasMixture`,
:func:`~statthermopy.plots.plotting.plot_property`, :class:`~statthermopy.io.exporters.Exporter`,
:func:`~statthermopy.validation.validate`). No physics is reimplemented here.

Presentation (palette, spacing, icons, light/dark theme) is driven by
:mod:`statthermopy.gui.theme`; the theme module is imported lazily so a broken/absent Qt
install never affects the core import.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.state import State
from ..database import get, list_molecules
from ..io import Exporter
from ..mixture import IdealGasMixture
from ..plots import (
    MIXTURE_PROPS,
    MOLAR_PROPS,
    PARTITION_PROPS,
    plot_mixture_property,
    plot_mixture_thermal_fields,
    plot_property,
    plot_thermal_fields,
)
from ..thermodynamics import Thermodynamics
from ..validation import list_references, validate

__all__ = ["StatThermoPyWindow"]

#: molar property rows shown in the results table (T_v/T_p are the thermal fields, in K).
_MOLAR_ROWS = ["U_m", "H_m", "S_m", "A_m", "G_m", "Cv_m", "Cp_m", "gamma", "T_v", "T_p", "mu_m"]
#: massic (per-kg) property rows shown in the results table.
_MASSIC_ROWS = ["U_s", "H_s", "S_s", "A_s", "G_s", "Cv_s", "Cp_s", "R_specific"]
_UNITS = {
    "U_m": "J/mol", "H_m": "J/mol", "A_m": "J/mol", "G_m": "J/mol", "mu_m": "J/mol",
    "S_m": "J/mol/K", "Cv_m": "J/mol/K", "Cp_m": "J/mol/K", "gamma": "-",
    "T_v": "K", "T_p": "K",
    "U_s": "J/kg", "H_s": "J/kg", "A_s": "J/kg", "G_s": "J/kg", "S_s": "J/kg/K",
    "Cv_s": "J/kg/K", "Cp_s": "J/kg/K", "R_specific": "J/kg/K",
}
_MODE_COLS = ["ln_q", "U_m", "S_m", "A_m", "Cv_m"]
#: Special Plot-tab entry that overlays both thermal fields (T_v and T_p) on one axes.
_THERMAL_FIELDS_ITEM = "T_v & T_p (thermal fields)"
_VALIDATION_TOL_PERCENT = 5.0


def _fmt(v: float) -> str:
    """Format a float for table display."""
    if isinstance(v, (int,)) or float(v).is_integer():
        return f"{v:.4g}"
    return f"{v:.6g}"


class _PlotCanvas(QWidget):
    """A matplotlib figure + canvas + toolbar inside a Qt widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Imported lazily, not at module top: importing backend_qtagg before
        # ``matplotlib.use("QtAgg")`` lets matplotlib probe for Qt bindings and load a
        # foreign Qt DLL (e.g. a Qt5 DLL on PATH from another app), which then clashes
        # with PySide6's Qt6 ("DLL load failed ... specified procedure not found").
        # ``app.main()`` calls ``matplotlib.use("QtAgg")`` before any _PlotCanvas is built,
        # so by the time we reach here the backend is pinned and the probe is skipped.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

        self.figure = Figure(figsize=(7, 4.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    @property
    def ax(self):
        """The current axes (created on demand)."""
        if not self.figure.axes:
            self.figure.add_subplot()
        return self.figure.axes[0]

    def refresh(self) -> None:
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def apply_theme(self, palette) -> None:
        """Recolor the figure to match the active palette (called on theme changes)."""
        self.figure.set_facecolor(palette.surface)
        for ax in self.figure.axes:
            ax.set_facecolor(palette.surface_alt)
            ax.tick_params(colors=palette.text)
            for spine in ax.spines.values():
                spine.set_edgecolor(palette.border)
            ax.xaxis.label.set_color(palette.text)
            ax.yaxis.label.set_color(palette.text)
            ax.title.set_color(palette.text)
            ax.grid(True, color=palette.border, alpha=0.5)
        self.canvas.draw_idle()


class StatThermoPyWindow(QMainWindow):
    """Main application window with Properties / Plot / Validate tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("StatThermoPy — Statistical Thermodynamics")
        self.resize(1100, 760)

        self._last_result = None  # last ThermoProperties / MixtureProperties (for export)
        self._theme_mode = "light"
        self._theme_palette = None
        self._theme_choice = "System"
        self._theme_actions: dict[str, QAction] = {}

        tabs = QTabWidget(self)
        tabs.addTab(self._build_properties_tab(), "Properties")
        tabs.addTab(self._build_plot_tab(), "Plot")
        tabs.addTab(self._build_transport_tab(), "Transport")
        tabs.addTab(self._build_validate_tab(), "Validate")
        self.setCentralWidget(tabs)
        self._tabs = tabs

        self._build_menu()

        # sensible defaults
        self.T_spin.setValue(298.15)
        self.P_spin.setValue(101325.0)
        self.plot_tmin.setValue(0.0)
        self.plot_tmax.setValue(1500.0)
        self.plot_npts.setValue(100)
        self.plot_p.setValue(101325.0)

        # apply the active platform theme (light/dark) to every widget + canvas
        self._apply_theme(self._detect_theme())

    # ------------------------------------------------------------------ UI build
    def _build_properties_tab(self) -> QWidget:
        tab = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # --- left column: selection + state + compute ---
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.setSpacing(10)

        # selection: pure gas vs mixture
        sel = QGroupBox("Selection")
        sel_lay = QVBoxLayout(sel)
        self.mode_group = QButtonGroup(self)
        self.radio_pure = QRadioButton("Pure gas")
        self.radio_mix = QRadioButton("Mixture")
        self.radio_pure.setChecked(True)
        self.mode_group.addButton(self.radio_pure, 0)
        self.mode_group.addButton(self.radio_mix, 1)
        self.radio_pure.toggled.connect(self._on_mode_changed)
        row = QHBoxLayout()
        row.addWidget(self.radio_pure)
        row.addWidget(self.radio_mix)
        row.addStretch()
        sel_lay.addLayout(row)

        self.gas_combo = QComboBox()
        for name in list_molecules():
            self.gas_combo.addItem(name)
        sel_lay.addWidget(QLabel("Species:"))
        sel_lay.addWidget(self.gas_combo)

        # mixture editor
        self.mix_table = QTableWidget(0, 2, self)
        self.mix_table.setHorizontalHeaderLabels(["Species", "Fraction"])
        self.mix_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.mix_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.basis_group = QButtonGroup(self)
        self.basis_mole = QRadioButton("mole")
        self.basis_mass = QRadioButton("mass")
        self.basis_mole.setChecked(True)
        self.basis_group.addButton(self.basis_mole, 0)
        self.basis_group.addButton(self.basis_mass, 1)
        mix_btns = QHBoxLayout()
        self.add_row_btn = QPushButton("Add row")
        self.del_row_btn = QPushButton("Remove row")
        self.add_row_btn.clicked.connect(self._add_mixture_row)
        self.del_row_btn.clicked.connect(self._del_mixture_row)
        mix_btns.addWidget(self.add_row_btn)
        mix_btns.addWidget(self.del_row_btn)
        mix_btns.addStretch()
        mix_btns.addWidget(self.basis_mole)
        mix_btns.addWidget(self.basis_mass)
        sel_lay.addWidget(QLabel("Mixture components:"))
        sel_lay.addWidget(self.mix_table)
        sel_lay.addLayout(mix_btns)
        # live sum indicator — fractions are coupled to always sum to 1
        self.mix_sum_label = QLabel("Σ = 1.0000")
        self.mix_sum_label.setAlignment(Qt.AlignRight)
        sel_lay.addWidget(self.mix_sum_label)
        self._add_mixture_row()  # start with one empty row
        llay.addWidget(sel)
        self._on_mode_changed()

        # state inputs
        state_box = QGroupBox("State")
        form = QFormLayout(state_box)
        self.T_spin = self._make_spin(1.0, 1.0e6, 298.15)
        self.P_spin = self._make_spin(0.0, 1.0e9, 101325.0)
        self.V_spin = self._make_spin(0.0, 1.0e6, 0.024)
        self.n_spin = self._make_spin(0.0, 1.0e6, 1.0)
        self.m_spin = self._make_spin(0.0, 1.0e6, 0.028)
        self.V_chk = QCheckBox("V")
        self.n_chk = QCheckBox("n")
        self.m_chk = QCheckBox("m")
        for label, spin, chk in [
            ("T (K)", self.T_spin, None),
            ("P (Pa)", self.P_spin, None),
            ("V (m^3)", self.V_spin, self.V_chk),
            ("n (mol)", self.n_spin, self.n_chk),
            ("m (kg)", self.m_spin, self.m_chk),
        ]:
            row = QHBoxLayout()
            row.addWidget(spin)
            if chk is not None:
                row.addWidget(chk)
            row.addStretch()
            form.addRow(label, row)
        llay.addWidget(state_box)

        self.compute_btn = QPushButton("Compute")
        self.compute_btn.setProperty("primary", True)
        self.compute_btn.clicked.connect(self._on_compute)
        llay.addWidget(self.compute_btn)
        llay.addStretch()
        splitter.addWidget(left)

        # --- right column: results + per-mode breakdown ---
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(10)

        res_box = QGroupBox("Results")
        res_lay = QVBoxLayout(res_box)
        res_lay.setContentsMargins(8, 8, 8, 8)
        self.results_table = QTableWidget(0, 3, self)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setHorizontalHeaderLabels(["Property", "Molar", "Massic"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        res_lay.addWidget(self.results_table)
        rlay.addWidget(res_box, 1)

        modes_box = QGroupBox("Per-mode contributions")
        modes_lay = QVBoxLayout(modes_box)
        modes_lay.setContentsMargins(8, 8, 8, 8)
        self.modes_table = QTableWidget(0, len(_MODE_COLS) + 1, self)
        self.modes_table.setAlternatingRowColors(True)
        self.modes_table.setHorizontalHeaderLabels(["Mode", *_MODE_COLS])
        self.modes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        modes_lay.addWidget(self.modes_table)
        rlay.addWidget(modes_box, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 700])
        root.addWidget(splitter)
        return tab

    def _build_plot_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        card = QFrame()
        card.setObjectName("Card")
        ctrl = QHBoxLayout(card)
        ctrl.setContentsMargins(10, 8, 10, 8)
        ctrl.setSpacing(8)
        self.plot_prop = QComboBox()
        for p in [*MOLAR_PROPS, *PARTITION_PROPS, _THERMAL_FIELDS_ITEM]:
            self.plot_prop.addItem(p)
        ctrl.addWidget(QLabel("Property:"))
        ctrl.addWidget(self.plot_prop)
        self.plot_tmin = self._make_spin(0.0, 1.0e5, 0.0)
        self.plot_tmax = self._make_spin(0.0, 1.0e5, 1500.0)
        self.plot_npts = QSpinBox()
        self.plot_npts.setRange(2, 100000)
        self.plot_npts.setValue(100)
        self.plot_p = self._make_spin(0.0, 1.0e9, 101325.0)
        ctrl.addWidget(QLabel("Tmin (K):"))
        ctrl.addWidget(self.plot_tmin)
        ctrl.addWidget(QLabel("Tmax (K):"))
        ctrl.addWidget(self.plot_tmax)
        ctrl.addWidget(QLabel("N:"))
        ctrl.addWidget(self.plot_npts)
        ctrl.addWidget(QLabel("P (Pa):"))
        ctrl.addWidget(self.plot_p)
        ctrl.addStretch()
        self.plot_btn = QPushButton("Plot")
        self.plot_btn.setProperty("primary", True)
        self.plot_btn.clicked.connect(self._on_plot)
        ctrl.addWidget(self.plot_btn)
        lay.addWidget(card)

        self.plot_canvas = _PlotCanvas()
        lay.addWidget(self.plot_canvas, 1)
        return tab

    def _build_transport_tab(self) -> QWidget:
        """Statistical Transport Properties tab.

        Two stacked panels: a *point-evaluation* card (species, T, P → a results table of all
        transport/thermophysical properties) and a *plot* card (mode ``vs T`` / ``vs P`` /
        ``2-D map``, multi-property selection, ranges, Plot) above a ``_PlotCanvas``. Export
        buttons cover CSV, Excel, Tecplot (2-D map), PNG and PDF. No physics is implemented here —
        it wraps :class:`~statthermopy.transport.TransportCalculator` and the transport plots.
        """
        from ..transport import TRANSPORT_PROPS

        tab = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # --- left column: point evaluation + results table -------------------
        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        llay.setSpacing(10)

        card1 = QFrame()
        card1.setObjectName("Card")
        c1 = QHBoxLayout(card1)
        c1.setContentsMargins(10, 8, 10, 8)
        c1.setSpacing(8)
        self.transport_species = QComboBox()
        for name in list_molecules():
            self.transport_species.addItem(name)
        c1.addWidget(QLabel("Species:"))
        c1.addWidget(self.transport_species)
        self.transport_T = self._make_spin(0.0, 1.0e5, 300.0)
        self.transport_P = self._make_spin(0.0, 1.0e9, 101325.0)
        c1.addWidget(QLabel("T (K):"))
        c1.addWidget(self.transport_T)
        c1.addWidget(QLabel("P (Pa):"))
        c1.addWidget(self.transport_P)
        self.transport_btn = QPushButton("Compute")
        self.transport_btn.setProperty("primary", True)
        self.transport_btn.clicked.connect(self._on_transport_compute)
        c1.addWidget(self.transport_btn)
        c1.addStretch()
        llay.addWidget(card1)

        self.transport_table = QTableWidget(0, 3, self)
        self.transport_table.setAlternatingRowColors(True)
        self.transport_table.setHorizontalHeaderLabels(["Property", "Value", "Unit"])
        self.transport_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.transport_table.verticalHeader().setVisible(False)
        llay.addWidget(self.transport_table, 1)
        llay.addStretch(0)

        # --- right column: plot controls + canvas ----------------------------
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(10)

        # plot card — controls wrapped over two rows so nothing is squeezed
        card2 = QFrame()
        card2.setObjectName("Card")
        c2 = QVBoxLayout(card2)
        c2.setContentsMargins(10, 8, 10, 8)
        c2.setSpacing(8)
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self.transport_mode = QComboBox()
        self.transport_mode.addItems(["vs T", "vs P", "2-D map"])
        row1.addWidget(QLabel("Mode:"))
        row1.addWidget(self.transport_mode)
        self.transport_props = QListWidget()
        self.transport_props.setSelectionMode(QAbstractItemView.MultiSelection)
        self.transport_props.setMaximumHeight(64)
        for p in TRANSPORT_PROPS:
            QListWidgetItem(p, self.transport_props)
        self.transport_props.item(0).setSelected(True)
        row1.addWidget(QLabel("Property:"))
        row1.addWidget(self.transport_props, 1)
        self.transport_plot_btn = QPushButton("Plot")
        self.transport_plot_btn.setProperty("primary", True)
        self.transport_plot_btn.clicked.connect(self._on_transport_plot)
        row1.addWidget(self.transport_plot_btn)
        c2.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self.transport_tmin = self._make_spin(0.0, 1.0e5, 300.0)
        self.transport_tmax = self._make_spin(0.0, 1.0e5, 1500.0)
        self.transport_pmin = self._make_spin(1.0, 1.0e9, 1e3)
        self.transport_pmax = self._make_spin(1.0, 1.0e9, 1e7)
        self.transport_npts = QSpinBox()
        self.transport_npts.setRange(10, 100000)
        self.transport_npts.setValue(50)
        row2.addWidget(QLabel("Tmin:"))
        row2.addWidget(self.transport_tmin)
        row2.addWidget(QLabel("Tmax:"))
        row2.addWidget(self.transport_tmax)
        row2.addWidget(QLabel("Pmin:"))
        row2.addWidget(self.transport_pmin)
        row2.addWidget(QLabel("Pmax:"))
        row2.addWidget(self.transport_pmax)
        row2.addWidget(QLabel("N:"))
        row2.addWidget(self.transport_npts)
        row2.addStretch()
        c2.addLayout(row2)
        rlay.addWidget(card2)

        # export row
        exp_row = QHBoxLayout()
        exp_row.setSpacing(6)
        self.transport_csv_btn = QPushButton("Export CSV")
        self.transport_xlsx_btn = QPushButton("Export Excel")
        self.transport_dat_btn = QPushButton("Export Tecplot")
        self.transport_png_btn = QPushButton("Export PNG")
        self.transport_pdf_btn = QPushButton("Export PDF")
        for b in (self.transport_csv_btn, self.transport_xlsx_btn, self.transport_dat_btn,
                  self.transport_png_btn, self.transport_pdf_btn):
            b.clicked.connect(self._on_transport_export)
        exp_row.addWidget(self.transport_csv_btn)
        exp_row.addWidget(self.transport_xlsx_btn)
        exp_row.addWidget(self.transport_dat_btn)
        exp_row.addWidget(self.transport_png_btn)
        exp_row.addWidget(self.transport_pdf_btn)
        exp_row.addStretch()
        rlay.addLayout(exp_row)

        self.transport_status = QLabel(
            "Compute transport properties at (T, P), or plot curves / 2-D maps. "
            "Dilute ideal gas: μ, k are T-only; ν, α, D scale as 1/P.")
        self.transport_status.setWordWrap(True)
        rlay.addWidget(self.transport_status)

        self.transport_canvas = _PlotCanvas()
        rlay.addWidget(self.transport_canvas, 1)

        # assemble the splitter; give the plot (right) the larger share
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([440, 660])
        root.addWidget(splitter)

        self._transport_last = None   # last TransportProperties (point eval, for CSV/Excel)
        self._transport_map = None    # (prop, T_range, P_range, n) of the last 2-D map (Tecplot)
        return tab

    def _build_validate_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        card = QFrame()
        card.setObjectName("Card")
        ctrl = QHBoxLayout(card)
        ctrl.setContentsMargins(10, 8, 10, 8)
        ctrl.setSpacing(8)
        self.val_species = QComboBox()
        for name in list_references():
            self.val_species.addItem(name)
        self.val_prop = QComboBox()
        self.val_prop.addItem("Cp")
        self.val_prop.addItem("S")
        ctrl.addWidget(QLabel("Species:"))
        ctrl.addWidget(self.val_species)
        ctrl.addWidget(QLabel("Property:"))
        ctrl.addWidget(self.val_prop)
        ctrl.addStretch()
        self.val_btn = QPushButton("Run")
        self.val_btn.setProperty("primary", True)
        self.val_btn.clicked.connect(self._on_validate)
        ctrl.addWidget(self.val_btn)
        lay.addWidget(card)

        self.val_status = QLabel("Click Run to validate the engine against embedded NIST/JANAF data.")
        self.val_status.setProperty("verdict", "idle")
        lay.addWidget(self.val_status)

        splitter = QSplitter(Qt.Horizontal)
        left = QGroupBox("Reference comparison")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(8, 8, 8, 8)
        self.val_table = QTableWidget(0, 4, self)
        self.val_table.setAlternatingRowColors(True)
        self.val_table.setHorizontalHeaderLabels(["T (K)", "Predicted", "Reference", "Error %"])
        self.val_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        left_lay.addWidget(self.val_table)
        splitter.addWidget(left)
        self.val_canvas = _PlotCanvas()
        splitter.addWidget(self.val_canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 580])
        lay.addWidget(splitter, 1)
        return tab

    def _build_menu(self) -> None:
        mb = self.menuBar()
        export = mb.addMenu("&Export")
        save_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        for fmt in ("csv", "json", "yaml", "excel", "latex"):
            act = export.addAction(save_icon, fmt.upper())
            act.triggered.connect(lambda _checked=False, f=fmt: self._on_export(f))

        view = mb.addMenu("&View")
        theme_menu = view.addMenu("Theme")
        grp = QActionGroup(self)
        grp.setExclusive(True)
        for label, key in (("System", "System"), ("Light", "Light"), ("Dark", "Dark")):
            act = QAction(label, self)
            act.setCheckable(True)
            grp.addAction(act)
            act.triggered.connect(lambda _checked=False, k=key: self._on_theme_chosen(k))
            theme_menu.addAction(act)
            self._theme_actions[key] = act
        self._theme_actions["System"].setChecked(True)

    # ------------------------------------------------------------------ theming
    def _detect_theme(self) -> str:
        from . import theme

        return "dark" if theme.detect_dark() else "light"

    def _set_icons(self, icons: dict) -> None:
        self.add_row_btn.setIcon(icons["plus"])
        self.del_row_btn.setIcon(icons["minus"])
        self.compute_btn.setIcon(icons["check"])
        self.plot_btn.setIcon(icons["play"])
        self.val_btn.setIcon(icons["play"])
        if hasattr(self, "transport_btn"):
            self.transport_btn.setIcon(icons["check"])
            self.transport_plot_btn.setIcon(icons["play"])

    def _apply_theme(self, mode: str) -> None:
        from . import theme

        palette = theme.DARK if mode == "dark" else theme.LIGHT
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.qss(palette))
            theme.apply_qt_palette(app, palette)
        icons = theme.make_icons(palette)
        self._set_icons(icons)
        self._theme_mode = mode
        self._theme_palette = palette
        # re-polish primary buttons so the dynamic [primary="true"] rule re-applies
        for btn in (self.compute_btn, self.plot_btn, self.val_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        if hasattr(self, "transport_btn"):
            for btn in (self.transport_btn, self.transport_plot_btn):
                btn.style().unpolish(btn)
                btn.style().polish(btn)
        self.plot_canvas.apply_theme(palette)
        self.val_canvas.apply_theme(palette)
        if hasattr(self, "transport_canvas"):
            self.transport_canvas.apply_theme(palette)

    def _on_theme_chosen(self, choice: str) -> None:
        self._theme_choice = choice
        effective = self._detect_theme() if choice == "System" else choice.lower()
        self._apply_theme(effective)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _make_spin(minv: float, maxv: float, val: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(minv, maxv)
        sb.setDecimals(6)
        sb.setValue(val)
        sb.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        return sb

    def _on_mode_changed(self) -> None:
        pure = self.radio_pure.isChecked()
        self.gas_combo.setEnabled(pure)
        self.mix_table.setEnabled(not pure)
        self.add_row_btn.setEnabled(not pure)
        self.del_row_btn.setEnabled(not pure)
        self.basis_mole.setEnabled(not pure)
        self.basis_mass.setEnabled(not pure)
        self._sync_plot_props(pure)

    def _sync_plot_props(self, pure: bool) -> None:
        """Keep the Plot-tab property list in step with the pure/mixture mode.

        Partition-function factors are per-species, so while a mixture is selected
        the list is restricted to the molar properties in :data:`MIXTURE_PROPS`.
        The current selection is preserved when it is still valid.
        """
        # The Plot tab is built after the Properties tab, which calls _on_mode_changed
        # once during construction; the Plot tab then populates its own list (pure mode).
        if not hasattr(self, "plot_prop"):
            return
        current = self.plot_prop.currentText()
        props = (MOLAR_PROPS + PARTITION_PROPS) if pure else list(MIXTURE_PROPS)
        # The combined thermal-fields view is available in both modes.
        props = [*props, _THERMAL_FIELDS_ITEM]
        self.plot_prop.blockSignals(True)
        self.plot_prop.clear()
        for p in props:
            self.plot_prop.addItem(p)
        idx = self.plot_prop.findText(current)
        self.plot_prop.setCurrentIndex(idx if idx >= 0 else 0)
        self.plot_prop.blockSignals(False)

    def _add_mixture_row(self) -> None:
        r = self.mix_table.rowCount()
        self.mix_table.insertRow(r)
        combo = QComboBox()
        for name in list_molecules():
            combo.addItem(name)
        frac = QDoubleSpinBox()
        # free entry: each fraction is independent and only bounded to [0, 1];
        # the user is free to dial any composition. The Σ indicator below the
        # table shows the running sum (green when it is exactly 1), and the
        # mixture is normalised at compute time.
        frac.setRange(0.0, 1.0)
        frac.setDecimals(4)
        frac.setSingleStep(0.01)
        frac.setValue(1.0 if r == 0 else 0.0)
        frac.valueChanged.connect(self._update_fraction_sum)
        self.mix_table.setCellWidget(r, 0, combo)
        self.mix_table.setCellWidget(r, 1, frac)
        self._update_fraction_sum()

    def _del_mixture_row(self) -> None:
        r = self.mix_table.currentRow()
        if r >= 0:
            self.mix_table.removeRow(r)
            self._update_fraction_sum()

    # -- fractions -------------------------------------------------------------
    # Fractions are free: each spinbox is bounded to [0, 1] and editing one row
    # never changes another. ``_build_mixture`` normalises whatever the user
    # entered, so the computed mixture always uses a proper composition; the Σ
    # label just reports where the raw sum stands.

    def _fraction_rows(self) -> list[int]:
        return [r for r in range(self.mix_table.rowCount())
                if self.mix_table.cellWidget(r, 1) is not None]

    def _update_fraction_sum(self) -> None:
        total = sum(
            float(self.mix_table.cellWidget(r, 1).value())
            for r in self._fraction_rows()
        )
        ok = abs(total - 1.0) < 1e-4
        self.mix_sum_label.setText(f"Σ = {total:.4f}")
        self.mix_sum_label.setStyleSheet(
            "color: #2e7d32;" if ok else "color: #c62828;"
        )

    def _current_molecule(self):
        name = self.gas_combo.currentText()
        return get(name) if name else None

    def _build_mixture(self):
        fractions: dict[str, float] = {}
        for r in range(self.mix_table.rowCount()):
            combo = self.mix_table.cellWidget(r, 0)
            frac = self.mix_table.cellWidget(r, 1)
            if combo is None or frac is None:
                continue
            name = combo.currentText()
            val = frac.value()
            if name and val > 0.0:
                fractions[name] = val
        if not fractions:
            return None
        basis = "mass" if self.basis_mass.isChecked() else "mole"
        return IdealGasMixture.from_names(fractions, basis=basis)

    def _make_state(self) -> State | None:
        kwargs: dict = {"T": float(self.T_spin.value()), "P": float(self.P_spin.value())}
        if self.V_chk.isChecked():
            kwargs["V"] = float(self.V_spin.value())
        if self.n_chk.isChecked():
            kwargs["n"] = float(self.n_spin.value())
        if self.m_chk.isChecked():
            kwargs["m"] = float(self.m_spin.value())
        return State(**kwargs)

    # ------------------------------------------------------------------ actions
    def _on_compute(self) -> None:
        state = self._make_state()
        if state is None:  # pragma: no cover - T is always set via spinbox
            QMessageBox.warning(self, "StatThermoPy", "Set a temperature.")
            return
        try:
            if self.radio_pure.isChecked():
                mol = self._current_molecule()
                if mol is None:  # pragma: no cover - combo always has items
                    QMessageBox.warning(self, "StatThermoPy", "Select a species.")
                    return
                res = Thermodynamics(mol, state).compute()
                self._populate_results(res)
                self._populate_modes(mol, state)
            else:
                mix = self._build_mixture()
                if mix is None:
                    QMessageBox.warning(self, "StatThermoPy", "Add at least one mixture component.")
                    return
                res = mix.compute(state)
                self._populate_results(res)
                self.modes_table.setRowCount(0)
            self._last_result = res
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "StatThermoPy", f"Computation failed:\n{exc}")

    def _populate_results(self, res) -> None:
        rows = []
        for key in _MOLAR_ROWS:
            mv = getattr(res, key, None)
            sv = _massic_for(key)
            massic = getattr(res, sv, None) if sv else None
            rows.append((key, mv, massic))
        # gamma has no massic counterpart; pad missing massic with "".
        self.results_table.setRowCount(len(rows))
        for i, (key, mv, massic) in enumerate(rows):
            unit = _UNITS.get(key, "")
            self.results_table.setItem(i, 0, QTableWidgetItem(f"{key}  [{unit}]" if unit else key))
            self.results_table.setItem(i, 1, QTableWidgetItem(_fmt(mv) if mv is not None else ""))
            self.results_table.setItem(
                i, 2, QTableWidgetItem(_fmt(massic) if massic is not None else "—")
            )

    def _populate_modes(self, mol, state) -> None:
        pf = Thermodynamics(mol, state).partition
        # Show hindered internal rotation as its own row (only appears for molecules that have
        # rotors, e.g. C2H6/C3H8); "vibrational" then holds the harmonic oscillators alone.
        contribs = pf.contributions(state, split_internal_rotation=True)
        pv = pf.evaluate(state)
        self.modes_table.setRowCount(len(contribs) + 1)
        for i, (name, c) in enumerate(contribs.items()):
            self.modes_table.setItem(i, 0, QTableWidgetItem(name.replace("_", " ")))
            for j, col in enumerate(_MODE_COLS):
                val = getattr(c, col, float("nan"))
                self.modes_table.setItem(i, j + 1, QTableWidgetItem(_fmt(val)))
        # totals row
        tot = self.modes_table.rowCount() - 1
        self.modes_table.setItem(tot, 0, QTableWidgetItem("ln Q (total)"))
        self.modes_table.setItem(tot, 4, QTableWidgetItem(_fmt(pv.ln_Qtotal)))

    def _on_plot(self) -> None:
        prop = self.plot_prop.currentText()
        tmin = float(self.plot_tmin.value())
        tmax = float(self.plot_tmax.value())
        n = int(self.plot_npts.value())
        P = float(self.plot_p.value())
        if tmax <= tmin:
            QMessageBox.warning(self, "StatThermoPy", "Tmax must exceed Tmin.")
            return
        Ts = np.linspace(tmin, tmax, n)
        ax = self.plot_canvas.ax
        ax.clear()
        thermal = prop == _THERMAL_FIELDS_ITEM
        if self.radio_mix.isChecked():
            mix = self._build_mixture()
            if mix is None:
                QMessageBox.warning(self, "StatThermoPy", "Add at least one mixture component.")
                return
            if thermal:
                plot_mixture_thermal_fields(mix, Ts, P=P, ax=ax)
            else:
                plot_mixture_property(mix, prop, Ts, P=P, ax=ax, logy=prop in PARTITION_PROPS)
        else:
            mol = self._current_molecule()
            if mol is None:  # pragma: no cover - combo always has items
                QMessageBox.warning(
                    self, "StatThermoPy", "Select a species on the Properties tab."
                )
                return
            if thermal:
                plot_thermal_fields(mol, Ts, P=P, ax=ax)
            else:
                plot_property(mol, prop, Ts, P=P, ax=ax, logy=prop in PARTITION_PROPS)
        self.plot_canvas.refresh()

    # -- Transport tab ---------------------------------------------------------
    def _on_transport_compute(self) -> None:
        """Evaluate all transport/thermophysical properties at the chosen (T, P)."""
        from ..transport import TRANSPORT_PROPS, TRANSPORT_UNITS, TransportCalculator

        name = self.transport_species.currentText()
        mol = get(name)
        T = float(self.transport_T.value())
        P = float(self.transport_P.value())
        if not mol.has_lennard_jones:
            QMessageBox.warning(self, "Transport",
                                f"{name} has no Lennard–Jones parameters; cannot compute transport.")
            return
        try:
            res = TransportCalculator(mol, State(T=T, P=P)).compute()
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Transport", f"Computation failed:\n{exc}")
            return
        self._transport_last = res
        data = res.as_dict()
        self.transport_table.setRowCount(len(TRANSPORT_PROPS))
        for i, prop in enumerate(TRANSPORT_PROPS):
            val = data.get(prop, float("nan"))
            unit = TRANSPORT_UNITS.get(prop, "")
            self.transport_table.setItem(i, 0, QTableWidgetItem(prop))
            self.transport_table.setItem(i, 1, QTableWidgetItem(f"{val:.6g}"))
            self.transport_table.setItem(i, 2, QTableWidgetItem(unit))
        self.transport_status.setText(
            f"{name} @ T={T:.2f} K, P={P:.4g} Pa — μ={res.mu:.4g} Pa·s, "
            f"k={res.k:.4g} W/m·K, Pr={res.Pr:.3f}, a={res.a:.2f} m/s, Z={res.Z:.4f}")

    def _on_transport_plot(self) -> None:
        """Render the selected transport property/properties per the chosen mode."""
        from ..transport import plots as tplots

        name = self.transport_species.currentText()
        mol = get(name)
        if not mol.has_lennard_jones:
            QMessageBox.warning(self, "Transport",
                                f"{name} has no Lennard–Jones parameters; cannot compute transport.")
            return
        props = [it.text() for it in self.transport_props.selectedItems()]
        if not props:
            QMessageBox.warning(self, "Transport", "Select at least one property.")
            return
        mode = self.transport_mode.currentText()
        T_range = (float(self.transport_tmin.value()), float(self.transport_tmax.value()))
        P_range = (float(self.transport_pmin.value()), float(self.transport_pmax.value()))
        # Only the axis that is actually swept must be a proper range; the other is a single
        # constant (T for vs-P, P for vs-T). The 2-D map sweeps both.
        if mode == "2-D map":
            need_T = need_P = True
        elif mode == "vs P":
            need_T, need_P = False, True
        else:  # vs T
            need_T, need_P = True, False
        if (need_T and T_range[0] >= T_range[1]) or (need_P and P_range[0] >= P_range[1]):
            QMessageBox.warning(
                self, "Transport",
                "The swept axis must have max > min: Tmax>Tmin for 'vs T' / 2-D map, "
                "Pmax>Pmin for 'vs P' / 2-D map.")
            return
        n = int(self.transport_npts.value())
        ax = self.transport_canvas.ax
        ax.clear()
        try:
            if mode == "2-D map":
                if len(props) != 1:
                    QMessageBox.warning(self, "Transport",
                                       "2-D map uses a single property; plotting the first selection.")
                    props = props[:1]
                tplots.plot_transport_map(mol, props[0], T_range, P_range, n=n, ax=ax)
                self._transport_map = (props[0], T_range, P_range, n)
            elif mode == "vs P":
                import numpy as np
                Ps = np.linspace(P_range[0], P_range[1], n)
                for prop in props:
                    tplots.plot_transport_vs_P(mol, prop, Ps, T=T_range[0], ax=ax)
                self._transport_map = None
            else:  # vs T
                import numpy as np
                Ts = np.linspace(T_range[0], T_range[1], n)
                if len(props) == 1:
                    tplots.plot_transport_vs_T(mol, props[0], Ts,
                                                P=P_range[0], ax=ax)
                else:
                    tplots.plot_transport_multi(mol, props, Ts, P=P_range[0], ax=ax)
                self._transport_map = None
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Transport", f"Plot failed:\n{exc}")
            return
        self.transport_canvas.refresh()
        self.transport_status.setText(
            f"{mode}: {', '.join(props)} for {name} "
            f"(T {T_range[0]:.0f}–{T_range[1]:.0f} K, P {P_range[0]:.3g}–{P_range[1]:.3g} Pa).")

    def _on_transport_export(self) -> None:
        """Export the current transport result(s): CSV/Excel from the point eval or 2-D map,
        PNG/PDF from the canvas, Tecplot from the 2-D map."""
        from ..transport import TransportCalculator as TC
        from ..transport.export import write_tecplot_grid

        label = self.sender().text()
        ext = {"Export CSV": "csv", "Export Excel": "xlsx",
               "Export Tecplot": "dat", "Export PNG": "png",
               "Export PDF": "pdf"}[label]
        path, _ = QFileDialog.getSaveFileName(self, f"Export {ext.upper()}", f"transport.{ext}")
        if not path:  # pragma: no cover - user cancelled
            return
        try:
            if ext == "png":
                self.transport_canvas.figure.savefig(path, dpi=120, bbox_inches="tight")
            elif ext == "pdf":
                self.transport_canvas.figure.savefig(path, bbox_inches="tight")
            elif ext == "dat":
                if self._transport_map is None:
                    QMessageBox.warning(self, "Transport",
                                        "Tecplot export requires a 2-D map. Plot a map first.")
                    return
                prop, T_range, P_range, n = self._transport_map
                import numpy as np
                Ts = np.linspace(T_range[0], T_range[1], n)
                Ps = np.linspace(P_range[0], P_range[1], n)
                Z = np.empty((n, n))
                for i, T in enumerate(Ts):
                    for j, P in enumerate(Ps):
                        Z[j, i] = getattr(TC(get(self.transport_species.currentText()),
                                              State(T=float(T), P=float(P))).compute(), prop)
                write_tecplot_grid(path, Ts, Ps, Z, title=f"{prop} map")
            else:  # csv / xlsx — from point eval (as_dict) or last plotted curve
                if self._transport_last is not None:
                    from ..io import Exporter
                    if ext == "csv":
                        Exporter(self._transport_last).to_csv(path)
                    else:
                        Exporter(self._transport_last).to_excel(path)
                else:
                    QMessageBox.warning(self, "Transport",
                                        "Compute a point evaluation first (Compute button).")
                    return
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Transport", f"Export failed:\n{exc}")

    def _on_validate(self) -> None:
        species = self.val_species.currentText()
        prop = self.val_prop.currentText()
        try:
            report = validate(species, prop)
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "StatThermoPy", f"Validation failed:\n{exc}")
            return
        self.val_table.setRowCount(len(report.T))
        for i, (T, pred, ref, err) in enumerate(
            zip(report.T, report.predicted, report.reference, report.errors_percent, strict=False)
        ):
            self.val_table.setItem(i, 0, QTableWidgetItem(_fmt(T)))
            self.val_table.setItem(i, 1, QTableWidgetItem(_fmt(pred)))
            self.val_table.setItem(i, 2, QTableWidgetItem(_fmt(ref)))
            self.val_table.setItem(i, 3, QTableWidgetItem(f"{err:+.3f}"))
        mae = report.mean_abs_error_percent
        mx = report.max_abs_error_percent
        passed = mae < _VALIDATION_TOL_PERCENT
        verdict = "PASS" if passed else "FAIL"
        self.val_status.setText(
            f"{species} / {prop}: MAE = {mae:.3f}%  max = {mx:.3f}%  "
            f"[tolerance {_VALIDATION_TOL_PERCENT:.0f}%] -> {verdict}"
        )
        self._set_verdict("pass" if passed else "fail")
        # predicted vs reference plot
        ax = self.val_canvas.ax
        ax.clear()
        ax.plot(report.T, report.reference, "o-", label="reference (NIST/JANAF)")
        ax.plot(report.T, report.predicted, "s--", label="StatThermoPy")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(f"{prop} (J/mol/K)")
        ax.set_title(f"{species} — {prop} validation")
        ax.legend()
        ax.grid(alpha=0.3)
        self.val_canvas.refresh()

    def _set_verdict(self, verdict: str) -> None:
        self.val_status.setProperty("verdict", verdict)
        self.val_status.style().unpolish(self.val_status)
        self.val_status.style().polish(self.val_status)

    def _on_export(self, fmt: str) -> None:
        res = self._last_result
        if res is None:
            # try to compute from the current Properties-tab selection
            self._on_compute()
            res = self._last_result
        if res is None:  # pragma: no cover - compute always succeeds with valid selection
            QMessageBox.warning(self, "StatThermoPy", "Nothing to export; run Compute first.")
            return
        ext = {"csv": "csv", "json": "json", "yaml": "yaml", "excel": "xlsx", "latex": "tex"}[fmt]
        path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", f"statthermopy.{ext}")
        if not path:  # pragma: no cover - user cancelled dialog
            return
        try:
            getattr(Exporter(res), f"to_{fmt}")(path)
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "StatThermoPy", f"Export failed:\n{exc}")


def _massic_for(molar_key: str) -> str | None:
    """Map a molar property key to its massic (per-kg) counterpart, or None."""
    return {
        "U_m": "U_s", "H_m": "H_s", "S_m": "S_s", "A_m": "A_s", "G_m": "G_s",
        "Cv_m": "Cv_s", "Cp_m": "Cp_s", "mu_m": None, "gamma": None,
    }.get(molar_key)
