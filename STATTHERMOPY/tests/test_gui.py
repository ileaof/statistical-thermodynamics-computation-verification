"""Headless smoke tests for the Qt GUI (PySide6).

Skipped unless PySide6 is installed. Runs on the ``offscreen`` Qt platform so no display is
needed.
"""

from __future__ import annotations

import os
import sys

import json

import pytest

# Skip the whole module if PySide6 isn't available.
pytest.importorskip("PySide6")
# Force the headless Qt platform before any widget is constructed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")
from PySide6.QtWidgets import QApplication  # noqa: E402

from statthermopy.gui.app import main as gui_main  # noqa: E402
from statthermopy.gui.mainwindow import _MODE_COLS, StatThermoPyWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def win(qapp):
    w = StatThermoPyWindow()
    yield w
    w.close()


def test_top_level_does_not_import_gui():
    # importing the core package must not pull in the GUI (keeps it optional/lightweight).
    # Run in a fresh subprocess because this test module already imported the GUI.
    import subprocess

    code = (
        "import sys, statthermopy; "
        "assert 'statthermopy.gui' not in sys.modules, 'gui leaked into core import'; "
        "print('OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_gui_main_is_callable():
    assert callable(gui_main)


class TestFitsOnScreen:
    """The window must be resizable down to whatever display the user actually has.

    Every tab used to sit directly in the tab widget, so the widest tab's layout minimum
    became the window's minimum: 2509x663, which no laptop panel can show. The window then
    refused to shrink and the right-hand pane stayed off-screen, unreachable. Tabs now live
    in scroll areas, whose own minimum is small.
    """

    def test_every_tab_is_scrollable(self, win):
        from PySide6.QtWidgets import QScrollArea

        for i in range(win._tabs.count()):
            area = win._tabs.widget(i)
            assert isinstance(area, QScrollArea), win._tabs.tabText(i)
            # Without this the tab would not stretch to fill a large window.
            assert area.widgetResizable(), win._tabs.tabText(i)
            assert area.widget() is not None, win._tabs.tabText(i)

    def test_window_minimum_is_small_enough_for_a_laptop(self, win):
        """The binding constraint was horizontal: one tab wanted 2509 px of width."""
        minimum = win.minimumSizeHint()
        assert minimum.width() <= 1024
        assert minimum.height() <= 720

    def test_window_can_actually_be_resized_small(self, win):
        win.resize(800, 600)
        assert win.width() <= 800 and win.height() <= 600

    def test_scrollbars_appear_only_when_the_content_does_not_fit(self, win, qapp):
        """A tab wider than the viewport must scroll; one that fits must not show a bar."""
        win.show()
        win.resize(1000, 700)
        qapp.processEvents()
        bars = {}
        for i in range(win._tabs.count()):
            win._tabs.setCurrentIndex(i)
            qapp.processEvents()
            area = win._tabs.widget(i)
            needed = area.widget().minimumSizeHint().width() > area.viewport().width()
            bars[win._tabs.tabText(i)] = (needed, area.horizontalScrollBar().isVisible())
        for tab, (needed, shown) in bars.items():
            assert needed == shown, f"{tab}: needs scrollbar={needed}, shown={shown}"
        win.hide()

    def test_opening_size_never_exceeds_the_screen(self, win, qapp):
        available = qapp.primaryScreen().availableGeometry()
        assert win.width() <= available.width()
        assert win.height() <= available.height()


def test_window_constructs_with_tabs(win):
    assert win._tabs.count() == 7
    assert win._tabs.tabText(0) == "Properties"
    assert win._tabs.tabText(1) == "Plot"
    assert win._tabs.tabText(2) == "Transport"
    assert win._tabs.tabText(3) == "Humid Air"
    assert win._tabs.tabText(4) == "Thermodynamic Comparisons"
    assert win._tabs.tabText(5) == "Air Transport"
    assert win._tabs.tabText(6) == "Validate"


def test_compute_pure_gas_populates_results(win):
    win.gas_combo.setCurrentText("N2")
    win.T_spin.setValue(298.15)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    res = win._last_result
    assert res is not None
    assert res.Cp_m == pytest.approx(29.11, abs=0.05)
    assert res.gamma == pytest.approx(1.40, abs=0.01)
    # results + modes tables populated
    assert win.results_table.rowCount() > 0
    assert win.modes_table.rowCount() == 5  # 4 modes + totals


def test_modes_table_shows_internal_rotation_row_for_ethane(win):
    """A molecule with hindered internal rotors (C2H6) exposes a dedicated 'internal rotation'
    per-mode row; the harmonic 'vibrational' row is shown alongside it."""
    win.gas_combo.setCurrentText("C2H6")
    win.T_spin.setValue(800.0)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    assert win.modes_table.rowCount() == 6  # 5 modes + totals
    labels = [
        win.modes_table.item(r, 0).text()
        for r in range(win.modes_table.rowCount())
        if win.modes_table.item(r, 0) is not None
    ]
    assert "internal rotation" in labels
    assert "vibrational" in labels
    # the internal-rotation row carries a non-zero Cv_m contribution (last column)
    ir_row = labels.index("internal rotation")
    cv_col = _MODE_COLS.index("Cv_m") + 1
    assert float(win.modes_table.item(ir_row, cv_col).text()) > 0.0


def test_compute_mixture(win):
    win.radio_mix.setChecked(True)
    win._on_mode_changed()
    # one row already present; set N2 and O2
    win.mix_table.cellWidget(0, 0).setCurrentText("N2")
    win.mix_table.cellWidget(0, 1).setValue(0.8)
    win._add_mixture_row()
    win.mix_table.cellWidget(1, 0).setCurrentText("O2")
    win.mix_table.cellWidget(1, 1).setValue(0.2)
    win.T_spin.setValue(298.15)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    res = win._last_result
    assert res is not None
    # massic R of an 0.8 N2 / 0.2 O2 mixture ~ 286-287 J/kg/K
    assert 280.0 < res.R_specific < 295.0


def test_air_preset_loads_and_shows_components(win):
    """Selecting the 'Air' preset fills the editable mixture with dry-air species, and computing
    shows the per-component contribution table plus M_avg / R_specific / S_mixing."""
    win.preset_combo.setCurrentIndex(win.preset_combo.findText("Air"))
    win._on_load_preset()
    assert win.radio_mix.isChecked()
    species = [win.mix_table.cellWidget(r, 0).currentText()
               for r in range(win.mix_table.rowCount())]
    assert set(species) == {"N2", "O2", "AR", "CO2"}

    win.T_spin.setValue(298.15)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    res = win._last_result
    # canonical dry-air numbers
    assert res.M_avg * 1e3 == pytest.approx(28.96, abs=0.05)
    assert res.R_specific == pytest.approx(287.0, abs=0.5)
    assert res.S_mixing > 0.0

    # per-component table visible with a species row per component + a totals row
    assert not win.components_box.isHidden()
    assert win.modes_box.isHidden()
    comp_labels = [win.components_table.item(r, 0).text()
                   for r in range(win.components_table.rowCount())]
    assert {"N2", "O2", "Ar", "CO2"}.issubset(set(comp_labels))
    assert "Σ total" in comp_labels

    # results table lists the mixture summary rows
    res_labels = [win.results_table.item(r, 0).text()
                  for r in range(win.results_table.rowCount())]
    assert any("M_avg" in x for x in res_labels)
    assert any("R_specific" in x for x in res_labels)
    assert any("S_mixing" in x for x in res_labels)


def test_mixture_fractions_free_entry(win):
    """Each fraction is free and independent in [0, 1]; editing one never
    changes another. The mixture is normalised at compute time."""
    win.radio_mix.setChecked(True)
    win._on_mode_changed()

    def frac(r):
        return win.mix_table.cellWidget(r, 1)

    def reset_table(n):
        while win.mix_table.rowCount() > 0:
            win.mix_table.removeRow(0)
        for _ in range(n):
            win._add_mixture_row()

    # three independent fractions that do NOT sum to 1
    reset_table(3)
    for r, name in enumerate(["N2", "O2", "CO2"]):
        win.mix_table.cellWidget(r, 0).setCurrentText(name)
    frac(0).setValue(0.5)
    frac(1).setValue(0.5)
    frac(2).setValue(0.5)
    # each row keeps exactly the value the user typed — no coupling
    assert [frac(r).value() for r in range(3)] == pytest.approx([0.5, 0.5, 0.5], abs=1e-4)
    # ... and changing one row leaves the others untouched
    frac(1).setValue(0.2)
    assert frac(0).value() == pytest.approx(0.5, abs=1e-4)
    assert frac(2).value() == pytest.approx(0.5, abs=1e-4)
    assert frac(1).value() == pytest.approx(0.2, abs=1e-4)

    # fractions are bounded to [0, 1]
    frac(0).setValue(5.0)
    assert frac(0).value() == pytest.approx(1.0, abs=1e-4)
    frac(0).setValue(-1.0)
    assert frac(0).value() == pytest.approx(0.0, abs=1e-4)

    # the Σ indicator reports the running sum
    assert "Σ" in win.mix_sum_label.text()

    # compute normalises whatever the user entered (0.0/0.2/0.5 -> 0/0.2857/0.7143)
    reset_table(3)
    for r, name in enumerate(["N2", "O2", "CO2"]):
        win.mix_table.cellWidget(r, 0).setCurrentText(name)
    frac(0).setValue(0.0)
    frac(1).setValue(0.2)
    frac(2).setValue(0.5)
    win.T_spin.setValue(298.15)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    res = win._last_result
    assert res is not None
    # mole fractions normalised: O2 = 0.2/0.7, CO2 = 0.5/0.7; the 0-fraction N2
    # row is dropped from the mixture.
    assert res.x["O2"] == pytest.approx(0.2 / 0.7, abs=1e-6)
    assert res.x["CO2"] == pytest.approx(0.5 / 0.7, abs=1e-6)
    assert "N2" not in res.x


def test_plot_tab_renders(win):
    win.gas_combo.setCurrentText("N2")
    win.plot_prop.setCurrentText("Cp_m")
    win.plot_tmin.setValue(300.0)
    win.plot_tmax.setValue(1000.0)
    win.plot_npts.setValue(20)
    win._on_plot()
    assert len(win.plot_canvas.figure.axes) == 1
    ax = win.plot_canvas.figure.axes[0]
    assert ax.has_data()


def test_thermal_fields_available_in_gui(win):
    """The results table lists the thermal fields, and the Plot tab can draw both at once."""
    win.gas_combo.setCurrentText("CO2")
    win.T_spin.setValue(600.0)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    res = win._last_result
    assert res.T_v == pytest.approx(res.U_m / res.Cv_m)
    assert res.T_p == pytest.approx(res.H_m / res.Cp_m)
    # both thermal-field rows appear in the results table (with K units)
    prop_cells = [
        win.results_table.item(r, 0).text()
        for r in range(win.results_table.rowCount())
        if win.results_table.item(r, 0) is not None
    ]
    assert any("T_v" in c for c in prop_cells) and any("T_p" in c for c in prop_cells)

    # the combined thermal-fields plot draws two curves
    win.plot_prop.setCurrentText("T_v & T_p (thermal fields)")
    win.plot_tmin.setValue(300.0)
    win.plot_tmax.setValue(1500.0)
    win.plot_npts.setValue(15)
    win._on_plot()
    ax = win.plot_canvas.figure.axes[0]
    assert len(ax.lines) == 2
    assert len({ln.get_color() for ln in ax.lines}) == 2


def test_plot_tab_uses_mixture_in_mixture_mode(win):
    """Regression: the Plot tab used to ignore Mixture mode and plot the pure-gas
    combo (default AR, monatomic) -> a flat Cp_m/Cv_m line. It must now plot the
    actual mixture, whose Cp_m rises with temperature."""
    win.radio_mix.setChecked(True)
    win._on_mode_changed()
    win.mix_table.cellWidget(0, 0).setCurrentText("N2")
    win.mix_table.cellWidget(0, 1).setValue(0.8)
    win._add_mixture_row()
    win.mix_table.cellWidget(1, 0).setCurrentText("O2")
    win.mix_table.cellWidget(1, 1).setValue(0.2)
    # partition-function factors are dropped from the property list for a mixture
    props = [win.plot_prop.itemText(i) for i in range(win.plot_prop.count())]
    assert "Cp_m" in props
    assert "Qtotal" not in props
    win.plot_prop.setCurrentText("Cp_m")
    win.plot_tmin.setValue(300.0)
    win.plot_tmax.setValue(1500.0)
    win.plot_npts.setValue(15)
    win._on_plot()
    ax = win.plot_canvas.figure.axes[0]
    ys = ax.lines[0].get_ydata()
    assert ys[0] < ys[-1]  # not flat: real T-dependence of the diatomic mixture
    # switching back to pure restores the partition props
    win.radio_pure.setChecked(True)
    win._on_mode_changed()
    props = [win.plot_prop.itemText(i) for i in range(win.plot_prop.count())]
    assert "Qtotal" in props


def test_humidair_tab_computes_and_plots(win):
    """The Humid Air tab computes the saturation limit + psychrometrics and plots a curve."""
    win.humid_T.setValue(298.15)
    win.humid_P.setValue(101325.0)
    win.humid_mode.setCurrentIndex(0)  # Saturated (max solubility)
    win._on_humidair_compute()
    labels = [win.humid_table.item(r, 0).text() for r in range(win.humid_table.rowCount())]
    assert any("P_sat" in x for x in labels)
    assert any("humidity ratio max" in x for x in labels)
    assert any("wet-bulb" in x for x in labels)
    # vapour partition-function breakdown populated (4 factors)
    factors = [win.humid_modes_table.item(r, 0).text()
               for r in range(win.humid_modes_table.rowCount())]
    assert {"translational", "rotational", "vibrational", "electronic"}.issubset(set(factors))

    # sub-saturated mode enables the value field
    win.humid_mode.setCurrentIndex(1)  # Relative humidity
    win._on_humid_mode_changed()
    assert win.humid_value.isEnabled()
    win.humid_value.setValue(0.5)
    win._on_humidair_compute()

    # plot a saturation-pressure curve
    win.humid_plot_prop.setCurrentIndex(0)
    win.humid_tmin.setValue(280.0)
    win.humid_tmax.setValue(360.0)
    win.humid_npts.setValue(20)
    win._on_humidair_plot()
    assert win.humid_canvas.figure.axes[0].has_data()


def test_thermodynamic_comparisons_tab(win, tmp_path):
    """The dedicated 'Thermodynamic Comparisons' tab plots the water-vapour content and the
    4-curve dry/humid property comparison, and exports the data."""
    win.cmp_P.setValue(101325.0)
    win.cmp_mode.setCurrentIndex(1)  # relative humidity
    win.cmp_value.setValue(0.5)
    win.cmp_tmin.setValue(273.16)
    win.cmp_tmax.setValue(330.0)
    win.cmp_npts.setValue(20)

    # water-vapour content (actual + saturation)
    win.cmp_analysis.setCurrentIndex(0)
    win._on_comparison_analysis_changed()
    assert not win.cmp_prop.isEnabled()  # property controls disabled for this analysis
    win._on_comparison_plot()
    assert win.cmp_canvas.figure.axes[0].has_data()
    assert "actual w [g/kg]" in win._cmp_table.columns

    # dry vs humid: entropy comparison -> 4 curves
    win.cmp_analysis.setCurrentIndex(1)
    win._on_comparison_analysis_changed()
    assert win.cmp_prop.isEnabled()
    win.cmp_prop.setCurrentText("Entropy S")
    win.cmp_isobaric.setChecked(True)
    win.cmp_isochoric.setChecked(True)
    win._on_comparison_plot()
    assert len(win.cmp_canvas.figure.axes[0].get_lines()) == 4
    assert len(win._cmp_table.columns) == 4

    # export the numerical data to CSV
    out = tmp_path / "cmp.csv"
    win._cmp_table.to_csv(out)
    assert out.exists() and out.stat().st_size > 0


def test_validate_tab_runs_and_passes(win):
    win.val_species.setCurrentText("N2")
    win.val_prop.setCurrentText("Cp")
    win._on_validate()
    assert win.val_table.rowCount() > 0
    assert "PASS" in win.val_status.text().upper()
    assert len(win.val_canvas.figure.axes) == 1


class TestStateIsNeverOverDetermined:
    """P/V and n/m are alternatives; supplying both members of a pair over-determines it.

    The old checkboxes let a user tick V while P stayed active, and ``State.resolve`` accepts
    that pair without checking it against the ideal-gas law: with P=101325 Pa, T=298.15 K,
    n=1 mol and V=0.024 m^3 it returned PV/nRT = 0.981 and an entropy 0.16 J/mol/K off. Ticking
    both n and m raised "Inconsistent m and n" instead. Radio buttons remove both cases.
    """

    def test_pressure_and_volume_are_never_both_sent(self, win):
        win.T_spin.setValue(500.0)
        win.P_spin.setValue(1e5)
        win.use_P.setChecked(True)
        st = win._make_state()
        assert st.T == pytest.approx(500.0)
        assert st.P == pytest.approx(1e5)
        assert st.V is None

        win.use_V.setChecked(True)
        win.V_spin.setValue(0.05)
        st = win._make_state()
        assert st.V == pytest.approx(0.05)
        assert st.P is None
        win.use_P.setChecked(True)  # restore

    def test_moles_and_mass_are_never_both_sent(self, win):
        win.use_n.setChecked(True)
        win.n_spin.setValue(2.0)
        st = win._make_state()
        assert st.n == pytest.approx(2.0)
        assert st.m is None

        win.use_m.setChecked(True)
        win.m_spin.setValue(0.05)
        st = win._make_state()
        assert st.m == pytest.approx(0.05)
        assert st.n is None
        win.use_n.setChecked(True)  # restore

    @pytest.mark.parametrize("use_volume", [False, True])
    @pytest.mark.parametrize("use_mass", [False, True])
    def test_every_combination_resolves_to_an_ideal_gas(self, win, use_volume, use_mass):
        """PV = nRT must hold exactly, whichever pair of inputs the user picks."""
        from statthermopy.constants import R
        from statthermopy.database import get

        win.T_spin.setValue(298.15)
        win.P_spin.setValue(101325.0)
        win.V_spin.setValue(0.05)
        win.n_spin.setValue(2.0)
        win.m_spin.setValue(0.05)
        (win.use_V if use_volume else win.use_P).setChecked(True)
        (win.use_m if use_mass else win.use_n).setChecked(True)

        resolved = win._make_state().resolve(get("N2").molar_mass)
        assert resolved.P * resolved.V / (resolved.n * R * resolved.T) == pytest.approx(1.0)
        win.use_P.setChecked(True)
        win.use_n.setChecked(True)

    def test_the_derived_field_is_greyed_out(self, win):
        win.use_P.setChecked(True)
        assert win.P_spin.isEnabled() and not win.V_spin.isEnabled()
        win.use_V.setChecked(True)
        assert win.V_spin.isEnabled() and not win.P_spin.isEnabled()
        win.use_P.setChecked(True)


class TestPpmComponentsSurviveTheGui:
    """A ppm component must reach the mixture, and must appear on the plot that results."""

    def test_the_fraction_spinbox_can_hold_a_ppm_value(self, win):
        """At four decimals it rounded 4.5e-05 to zero on entry, silently dropping H2S."""
        win.radio_mix.setChecked(True)
        win._on_mode_changed()
        spin = win.mix_table.cellWidget(0, 1)
        spin.setValue(0.000045)
        assert spin.value() == pytest.approx(4.5e-05)
        win.radio_pure.setChecked(True)
        win._on_mode_changed()

    def test_a_small_component_is_named_on_the_plot_title(self, win):
        win.radio_mix.setChecked(True)
        win._on_mode_changed()
        while win.mix_table.rowCount() < 3:
            win._add_mixture_row()
        for row, (name, x) in enumerate([("CH4", 0.557), ("CO2", 0.439), ("O2", 0.004)]):
            win.mix_table.cellWidget(row, 0).setCurrentText(name)
            win.mix_table.cellWidget(row, 1).setValue(x)
        win.plot_prop.setCurrentText("Cp_m")
        win.plot_tmin.setValue(300.0)
        win.plot_tmax.setValue(1000.0)
        win._on_plot()
        title = win.plot_canvas.ax.get_title()
        assert "O2 0.0040" in title, title      # used to read "O2 0.00"
        win.radio_pure.setChecked(True)
        win._on_mode_changed()


class TestThreeReportingBases:
    """Results are indexed on three bases: per mol, per kg and per m^3."""

    @staticmethod
    def _rows(win):
        table = win.results_table
        out = {}
        for r in range(table.rowCount()):
            label = table.item(r, 0).text().split()[0]
            out[label] = [table.item(r, c).text() for c in range(1, 4)]
        return out

    def test_the_table_carries_all_three_columns(self, win):
        headers = [win.results_table.horizontalHeaderItem(c).text()
                   for c in range(win.results_table.columnCount())]
        assert headers == ["Property", "Molar", "Massic", "Volumetric"]

    def test_the_three_bases_reconcile(self, win):
        """volumetric = massic x rho, and molar = volumetric x V_m."""
        win.radio_pure.setChecked(True)
        win._on_mode_changed()
        win.gas_combo.setCurrentText("N2")
        win.T_spin.setValue(298.15)
        win.P_spin.setValue(101325.0)
        win._on_compute()
        rows = self._rows(win)
        V_m = float(rows["V_m"][0])
        rho = float(rows["rho"][2])
        for key in ("U_m", "H_m", "S_m", "Cv_m", "Cp_m"):
            molar, massic, volumetric = (float(v) for v in rows[key])
            assert volumetric == pytest.approx(molar / V_m, rel=1e-5), key
            assert volumetric == pytest.approx(massic * rho, rel=1e-4), key

    def test_ratios_have_no_per_volume_form(self, win):
        """gamma and the thermal fields are not extensive, so those cells stay blank."""
        win.gas_combo.setCurrentText("N2")
        win._on_compute()
        rows = self._rows(win)
        for key in ("gamma", "T_v", "T_p"):
            assert rows[key][1] == "—" and rows[key][2] == "—", key

    def test_a_mixture_also_gets_the_volumetric_column(self, win):
        win.radio_mix.setChecked(True)
        win._on_mode_changed()
        win.T_spin.setValue(300.0)
        win.P_spin.setValue(101325.0)
        win._on_compute()
        rows = self._rows(win)
        assert rows["S_m"][2] not in ("", "—")
        win.radio_pure.setChecked(True)
        win._on_mode_changed()


def test_export_writes_file(win, tmp_path):
    win.gas_combo.setCurrentText("N2")
    win.T_spin.setValue(298.15)
    win.P_spin.setValue(101325.0)
    win._on_compute()
    from statthermopy.io import Exporter

    out = tmp_path / "n2.csv"
    Exporter(win._last_result).to_csv(out)
    assert out.exists()
    assert "Cp_m" in out.read_text(encoding="utf-8")


def test_export_menu_path_with_mocked_dialog(win, tmp_path, monkeypatch):
    # Drive the menu-triggered export slot with the file dialog mocked off.
    win.gas_combo.setCurrentText("CO2")
    win.T_spin.setValue(800.0)
    win.P_spin.setValue(1e5)
    win._on_compute()
    target = tmp_path / "co2.json"
    monkeypatch.setattr(
        "statthermopy.gui.mainwindow.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(target), ""),
    )
    win._on_export("json")
    assert target.exists()
    assert "Cp_m" in target.read_text(encoding="utf-8")


def test_mixture_delete_row_and_n_m_state(win):
    # add two rows, delete the selected one
    win.radio_mix.setChecked(True)
    win._on_mode_changed()
    win._add_mixture_row()
    assert win.mix_table.rowCount() >= 2
    win.mix_table.selectRow(0)
    win._del_mixture_row()
    # the amount reaches _make_state as whichever of n/m is selected, never both
    win.use_m.setChecked(True)
    win.m_spin.setValue(0.05)
    st = win._make_state()
    assert st.m == pytest.approx(0.05)
    assert st.n is None
    win.use_n.setChecked(True)


def test_theme_applied_on_construct(win):
    # the light path runs at construction: a stylesheet is set and primary buttons flagged.
    assert win._theme_mode in {"light", "dark"}
    assert QApplication.instance().styleSheet().strip() != ""
    assert win.compute_btn.property("primary") is True
    assert win.plot_btn.property("primary") is True
    assert win.val_btn.property("primary") is True
    # icons assigned to every action button
    assert not win.compute_btn.icon().isNull()
    assert not win.add_row_btn.icon().isNull()


def test_apply_theme_dark_restyles(win):
    win._apply_theme("dark")
    assert win._theme_mode == "dark"
    sheet = QApplication.instance().styleSheet()
    assert "#0b1220" in sheet  # dark bg token
    # verdict badge canvas still valid after restyle
    assert win.val_canvas is not None
    win._apply_theme("light")
    assert win._theme_mode == "light"


def test_theme_toggle_via_menu(win):
    # drive the View -> Theme -> Dark action (radio group)
    dark_act = win._theme_actions["Dark"]
    dark_act.trigger()
    assert win._theme_mode == "dark"
    assert win._theme_choice == "Dark"
    light_act = win._theme_actions["Light"]
    light_act.trigger()
    assert win._theme_mode == "light"
    # canvas facecolor follows the active palette
    facecolor = win.plot_canvas.figure.get_facecolor()
    import matplotlib.colors as mcolors

    assert mcolors.to_hex(facecolor) == "#ffffff"


def test_theme_helpers():
    from PySide6.QtGui import QFont

    from statthermopy.gui import theme

    assert isinstance(theme.default_font(), QFont)
    # detect_dark runs against the live (offscreen) app and returns a plain bool
    assert isinstance(theme.detect_dark(), bool)


# -- Air Transport tab ----------------------------------------------------------


def test_air_transport_compute_populates_tables(win):
    """The Air Transport tab computes a humid point and fills the results + contribution tables."""
    win.air_T.setValue(298.15)
    win.air_P.setValue(101325.0)
    win.air_mode.setCurrentIndex(0)  # Saturated
    win._on_air_transport_compute()
    assert win.air_table.rowCount() > 0
    # the headline transport properties appear in the results table
    labels = [win.air_table.item(r, 0).text() for r in range(win.air_table.rowCount())]
    assert any("viscosity" in lab.lower() for lab in labels)
    assert any("Prandtl" in lab for lab in labels)
    # per-species contributions: 5 rows for saturated humid air (4 dry + H2O)
    assert win.air_contrib_table.rowCount() == 5
    species = [win.air_contrib_table.item(r, 0).text()
               for r in range(win.air_contrib_table.rowCount())]
    assert "H2O" in species
    assert win._air_transport_last is not None
    assert win._air_transport_last.mu > 0


def test_air_transport_plot_renders(win):
    """The dry-vs-humid comparison plot draws two curves and stores the table for export."""
    win.air_plot_prop.setCurrentIndex(0)  # mu
    win.air_which.setCurrentText("Dry vs Humid")
    win.air_tmin.setValue(280.0)
    win.air_tmax.setValue(360.0)
    win.air_npts.setValue(20)
    win.air_plot_p.setValue(101325.0)
    win._on_air_transport_plot()
    ax = win.air_canvas.figure.axes[0]
    assert ax.has_data()
    assert len(ax.lines) == 2  # dry + humid
    assert win._air_transport_table is not None
    assert "Dry air" in win._air_transport_table.columns


def test_air_transport_export_writes_files(win, tmp_path, monkeypatch):
    """The Air Transport export buttons write CSV / Excel / JSON / PDF from the point eval."""
    win.air_T.setValue(298.15)
    win.air_P.setValue(101325.0)
    win.air_mode.setCurrentIndex(1)  # Relative humidity
    win.air_value.setValue(0.5)
    win._on_air_transport_compute()
    monkeypatch.setattr(
        "statthermopy.gui.mainwindow.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(tmp_path / "air.json"), ""),
    )
    win.air_json_btn.click()
    out = tmp_path / "air.json"
    assert out.exists() and out.stat().st_size > 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "properties" in data and "components" in data


# -- Transport tab --------------------------------------------------------------


def test_transport_compute_populates_table(win):
    """Point evaluation fills the results table with all 13 transport properties."""
    from statthermopy.transport import TRANSPORT_PROPS

    win.transport_species.setCurrentText("N2")
    win.transport_T.setValue(300.0)
    win.transport_P.setValue(101325.0)
    win._on_transport_compute()
    assert win.transport_table.rowCount() == len(TRANSPORT_PROPS)
    # viscosity of N2 @ 300 K ~ 1.8e-5 Pa·s (row labelled "mu")
    labels = [win.transport_table.item(r, 0).text() for r in range(win.transport_table.rowCount())]
    assert "mu" in labels
    mu_row = labels.index("mu")
    mu_val = float(win.transport_table.item(mu_row, 1).text())
    assert mu_val == pytest.approx(1.77e-5, rel=0.05)
    assert win._transport_last is not None


def test_transport_plot_vs_t_renders(win):
    """A vs-T curve plots without exception and the canvas carries data."""
    win.transport_species.setCurrentText("N2")
    win.transport_mode.setCurrentText("vs T")
    # select only the first property (mu) — clear then reselect
    for i in range(win.transport_props.count()):
        win.transport_props.item(i).setSelected(i == 0)
    win.transport_tmin.setValue(300.0)
    win.transport_tmax.setValue(1000.0)
    win.transport_pmin.setValue(101325.0)
    win.transport_pmax.setValue(101325.0)
    win.transport_npts.setValue(20)
    win._on_transport_plot()
    ax = win.transport_canvas.figure.axes[0]
    assert ax.has_data()
    assert len(ax.lines) == 1


def test_transport_plot_map_renders(win):
    """A 2-D map builds a pcolormesh and records the map params for Tecplot export."""
    win.transport_species.setCurrentText("N2")
    win.transport_mode.setCurrentText("2-D map")
    for i in range(win.transport_props.count()):
        win.transport_props.item(i).setSelected(i == 0)
    win.transport_tmin.setValue(300.0)
    win.transport_tmax.setValue(800.0)
    win.transport_pmin.setValue(1e3)
    win.transport_pmax.setValue(1e6)
    win.transport_npts.setValue(12)
    win._on_transport_plot()
    assert win._transport_map is not None
    ax = win.transport_canvas.figure.axes[0]
    assert ax.has_data()


def test_transport_export_writes_files(win, tmp_path, monkeypatch):
    """CSV (point eval) and Tecplot (2-D map) export buttons write files."""
    # point eval -> CSV
    win.transport_species.setCurrentText("N2")
    win.transport_T.setValue(300.0)
    win.transport_P.setValue(101325.0)
    win._on_transport_compute()
    csv_target = tmp_path / "transport.csv"
    monkeypatch.setattr(
        "statthermopy.gui.mainwindow.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(csv_target), ""),
    )
    win.transport_csv_btn.click()
    assert csv_target.exists()
    assert "mu" in csv_target.read_text(encoding="utf-8")

    # 2-D map -> Tecplot
    win.transport_mode.setCurrentText("2-D map")
    for i in range(win.transport_props.count()):
        win.transport_props.item(i).setSelected(i == 0)
    win.transport_tmin.setValue(300.0)
    win.transport_tmax.setValue(600.0)
    win.transport_pmin.setValue(1e3)
    win.transport_pmax.setValue(1e5)
    win.transport_npts.setValue(10)
    win._on_transport_plot()
    dat_target = tmp_path / "transport.dat"
    monkeypatch.setattr(
        "statthermopy.gui.mainwindow.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(dat_target), ""),
    )
    win.transport_dat_btn.click()
    assert dat_target.exists()
    assert "VARIABLES" in dat_target.read_text(encoding="utf-8")
