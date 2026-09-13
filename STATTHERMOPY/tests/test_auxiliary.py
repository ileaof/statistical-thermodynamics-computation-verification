"""Tests for backend, validation, equilibrium, exporters, plots and CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from statthermopy import State, Thermodynamics, get


# -- Backend -----------------------------------------------------------------

def test_numpy_backend_basic():
    from statthermopy.backend import NumpyBackend, get_backend, set_backend
    b = get_backend()
    assert b.name == "numpy"
    assert math.isclose(b.sum(b.asarray([1.0, 2.0, 3.0])), 6.0)
    set_backend(NumpyBackend())
    assert get_backend().name == "numpy"


# -- Validation --------------------------------------------------------------

def test_validation_runner_with_fake_source():
    from statthermopy.validation import ReferenceSource, ValidationRunner

    # Build a "reference" whose Cp column equals the engine's own predictions (so the
    # validation machinery must report ~0% error) and a deliberately biased point.
    n2 = get("N2")
    Ts = [300.0, 500.0, 1000.0]
    cps = [Thermodynamics(n2, State(T=T, P=101325.0)).compute().Cp_m for T in Ts]

    class FakeSource:
        def load(self):
            import pandas as pd
            return pd.DataFrame({"T": Ts, "Cp": cps})

    runner = ValidationRunner("N2", FakeSource(), property_name="Cp", pressure=101325.0)
    report = runner.run()
    assert len(report.T) == 3
    assert all(abs(e) < 1e-6 for e in report.errors_percent)
    assert report.mean_abs_error_percent < 1e-6
    assert report.max_abs_error_percent < 1e-6


def test_validation_runner_detects_bias():
    from statthermopy.validation import ValidationRunner

    n2 = get("N2")
    pred = Thermodynamics(n2, State(T=300.0, P=101325.0)).compute().Cp_m
    # reference set 10% above prediction -> error = (pred - 1.1*pred)/(1.1*pred) = -9.09%
    ref = pred * 1.10

    class BiasedSource:
        def load(self):
            import pandas as pd
            return pd.DataFrame({"T": [300.0], "Cp": [ref]})

    runner = ValidationRunner("N2", BiasedSource(), property_name="Cp", pressure=101325.0)
    report = runner.run()
    assert report.errors_percent[0] < -5.0          # clearly negative
    assert abs(report.errors_percent[0] + 9.09) < 0.2  # ~ -9.09%


# -- Equilibrium placeholder -------------------------------------------------

def test_equilibrium_placeholder_raises():
    from statthermopy.equilibrium import Reaction, gibbs_minimisation, EquilibriumNotImplemented
    r = Reaction(reactants={"N2": 1.0, "H2": 3.0}, products={"NH3": 2.0})
    with pytest.raises(EquilibriumNotImplemented):
        r.delta_G(298.15)
    with pytest.raises(EquilibriumNotImplemented):
        r.equilibrium_constant(298.15)
    with pytest.raises(EquilibriumNotImplemented):
        gibbs_minimisation(["N2", "H2", "NH3"], {}, 1000.0, 1e5)


# -- Exporters ---------------------------------------------------------------

@pytest.fixture
def n2_result():
    return Thermodynamics(get("N2"), State(T=298.15, P=101325.0)).compute()


def test_export_csv(tmp_path, n2_result):
    from statthermopy.io import Exporter
    p = Exporter(n2_result).to_csv(tmp_path / "n2.csv")
    assert Path(p).exists()
    text = Path(p).read_text(encoding="utf-8")
    assert "Cp_m" in text


def test_export_json(tmp_path, n2_result):
    from statthermopy.io import Exporter
    p = Exporter(n2_result).to_json(tmp_path / "n2.json")
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    assert "properties" in data
    assert math.isclose(data["properties"]["Cp_m"], n2_result.Cp_m, rel_tol=1e-9)


def test_export_yaml(tmp_path, n2_result):
    from statthermopy.io import Exporter
    p = Exporter(n2_result).to_yaml(tmp_path / "n2.yaml")
    assert "properties" in Path(p).read_text(encoding="utf-8")


def test_export_latex(tmp_path, n2_result):
    from statthermopy.io import Exporter
    p = Exporter(n2_result).to_latex(tmp_path / "n2.tex")
    text = Path(p).read_text(encoding="utf-8")
    assert "tabular" in text and "Enthalpy" in text


def test_export_excel(tmp_path, n2_result):
    from statthermopy.io import Exporter
    p = Exporter(n2_result).to_excel(tmp_path / "n2.xlsx")
    assert Path(p).exists() and Path(p).stat().st_size > 0


def test_exporter_rejects_unknown_format(tmp_path, n2_result):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh._last_result = n2_result
    sh.onecmd(f"export bogus {tmp_path / 'x'}")
    # no exception; just an error message printed (covered by command)


# -- Plots -------------------------------------------------------------------

def test_plot_property_returns_axes(tmp_path):
    from statthermopy.plots import plot_property
    ax = plot_property("N2", "Cp_m", [300, 500, 1000], P=1e5)
    assert ax is not None
    out = tmp_path / "fig.png"
    ax.figure.savefig(out, dpi=80)
    assert out.exists()


def test_plot_all_properties(tmp_path):
    from statthermopy.plots import plot_all_properties
    axes = plot_all_properties("Ar", [300, 500, 1000], P=1e5, save_dir=tmp_path)
    assert len(axes) == 15  # 10 molar (incl. T_v, T_p) + 5 partition
    assert any((tmp_path / f"Ar_{p}.png").exists() for p in ("Cp_m", "T_v", "Qtotal"))


def test_thermal_fields_definitions_and_units():
    """T_v = U_m/Cv_m and T_p = H_m/Cp_m (both in K); for a monatomic gas both equal T."""
    from statthermopy import Thermodynamics, get
    from statthermopy.core.state import State
    from statthermopy.plots import PROP_UNITS

    r = Thermodynamics(get("N2"), State(T=600.0, P=1e5)).compute()
    assert r.T_v == pytest.approx(r.U_m / r.Cv_m)
    assert r.T_p == pytest.approx(r.H_m / r.Cp_m)
    assert PROP_UNITS["T_v"] == "K" and PROP_UNITS["T_p"] == "K"

    # monatomic Ar: U=3/2 RT, Cv=3/2 R -> T_v = T ; H=5/2 RT, Cp=5/2 R -> T_p = T
    ar = Thermodynamics(get("Ar"), State(T=750.0, P=1e5)).compute()
    assert ar.T_v == pytest.approx(750.0, rel=1e-9)
    assert ar.T_p == pytest.approx(750.0, rel=1e-9)


def test_mixture_thermal_fields():
    """Mixtures expose T_v/T_p too, and the combined mixture plot draws both curves."""
    from statthermopy import IdealGasMixture
    from statthermopy.core.state import State
    from statthermopy.plots import plot_mixture_thermal_fields

    mix = IdealGasMixture.from_names({"N2": 0.78, "O2": 0.21, "Ar": 0.01})
    mp = mix.compute(State(T=800.0, P=1e5))
    assert mp.T_v == pytest.approx(mp.U_m / mp.Cv_m)
    assert mp.T_p == pytest.approx(mp.H_m / mp.Cp_m)
    ax = plot_mixture_thermal_fields(mix, [300, 800, 1500], P=1e5)
    assert len(ax.lines) == 2 and len({ln.get_color() for ln in ax.lines}) == 2


def test_plot_thermal_fields_two_distinct_curves():
    """The combined view draws both fields with distinct colours, a legend and K units."""
    from statthermopy.plots import plot_thermal_fields

    ax = plot_thermal_fields("CO2", [300, 600, 1000, 1500], P=1e5)
    assert len(ax.lines) == 2
    colours = {line.get_color() for line in ax.lines}
    assert len(colours) == 2  # distinct colours
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("T_v" in t for t in legend_texts) and any("T_p" in t for t in legend_texts)
    assert "K" in ax.get_ylabel()


def test_thermal_fields_property_vs_T_grid_matches():
    """T_v/T_p are grid-derivable, so the accelerated property_vs_T matches the per-T path."""
    import numpy as np

    from statthermopy import Thermodynamics, get
    from statthermopy.core.state import State

    th = Thermodynamics(get("CO2"), State(T=300.0, P=1e5))
    Ts = np.linspace(300.0, 1500.0, 7)
    for prop in ("T_v", "T_p"):
        _, grid = th.property_vs_T(prop, Ts)
        ref = [getattr(Thermodynamics(get("CO2"), State(T=t, P=1e5)).compute(), prop) for t in Ts]
        assert np.allclose(grid, ref)


# -- Thermal fields: continuous from 0 K (Third-Law / numerical stability) -----


def test_thermal_fields_finite_down_to_zero():
    """T_v/T_p must stay finite (no NaN from the vibrational exp(theta/T) overflow) for
    every T in (0, T_max] and tend to 0 as T -> 0."""
    mol = get("N2")
    for T in (1e-6, 1e-3, 1e-2, 1.0, 5.0, 50.0, 300.0, 2000.0):
        r = Thermodynamics(mol, State(T=T, P=1e5)).compute()
        assert math.isfinite(r.T_v), T
        assert math.isfinite(r.T_p), T
        assert math.isfinite(r.Cv_m), T
    # approaches 0 as T -> 0 (T_v = U/Cv ~ T for the classical model at low T)
    r_lo = Thermodynamics(mol, State(T=1e-3, P=1e5)).compute()
    assert abs(r_lo.T_v) < 1e-2
    assert abs(r_lo.T_p) < 1e-2


def test_thermal_fields_at_zero_kelvin():
    """T = 0 is accepted; the thermal fields collapse to 0 and U_m = H_m = 0."""
    r = Thermodynamics(get("N2"), State(T=0.0, P=1e5)).compute()
    assert math.isfinite(r.T_v) and r.T_v == 0.0
    assert math.isfinite(r.T_p) and r.T_p == 0.0
    assert r.U_m == 0.0 and r.H_m == 0.0
    # classical trans+rot keep their equipartition Cv (known Third-Law limitation);
    # the quantum vibrational/electronic modes freeze.
    assert r.Cv_m == pytest.approx(2.5 * 8.314462618, rel=1e-9)


def test_quantum_modes_freeze_at_low_T():
    """Vibrational and electronic Cv -> 0 as T -> 0 (Third Law for the quantum modes)."""
    from statthermopy.core.state import State
    from statthermopy.partition import PartitionFunction

    mol = get("N2")
    pf = PartitionFunction(mol)
    contribs = pf.contributions(State(T=1e-3, P=1e5))
    assert contribs["vibrational"].Cv_m < 1e-6 * 8.314462618
    assert contribs["electronic"].Cv_m < 1e-6 * 8.314462618


def test_thermal_fields_grid_continuous_from_zero():
    """property_vs_T over a grid starting at 0 K is finite on both the NumPy and the
    accelerated paths, and the two agree at low temperature."""
    import numpy as np

    from statthermopy.backend import set_backend

    mol = get("N2")
    Ts = np.linspace(0.0, 2000.0, 400)
    results = {}
    for b in ("numpy", "numba"):
        set_backend(b)
        th = Thermodynamics(mol, State(T=300.0, P=1e5))
        _, tv = th.property_vs_T("T_v", Ts, P=1e5)
        _, tp = th.property_vs_T("T_p", Ts, P=1e5)
        tv, tp = np.asarray(tv), np.asarray(tp)
        assert np.isfinite(tv).all() and np.isfinite(tp).all(), b
        assert tv[0] == 0.0 and tp[0] == 0.0
        results[b] = (tv, tp)
    set_backend("numpy")
    assert np.allclose(results["numpy"][0], results["numba"][0], atol=1e-9)
    assert np.allclose(results["numpy"][1], results["numba"][1], atol=1e-9)


def test_plot_thermal_fields_from_zero():
    """The combined thermal-field plot draws both curves starting at Tmin = 0 without NaN."""
    import numpy as np

    from statthermopy.plots import plot_thermal_fields

    ax = plot_thermal_fields("N2", np.linspace(0.0, 1500.0, 200), P=1e5)
    for line in ax.lines:
        ys = line.get_ydata()
        assert np.isfinite(np.asarray(ys)).all()
    assert len(ax.lines) == 2


def test_plot_mixture_thermal_fields_from_zero():
    """Mixture thermal-field curves are continuous from 0 K too."""
    import numpy as np

    from statthermopy import IdealGasMixture
    from statthermopy.plots import plot_mixture_thermal_fields

    mix = IdealGasMixture.from_names({"N2": 0.78, "O2": 0.21, "Ar": 0.01})
    ax = plot_mixture_thermal_fields(mix, np.linspace(0.0, 1500.0, 200), P=1e5)
    for line in ax.lines:
        assert np.isfinite(np.asarray(line.get_ydata())).all()
    assert len(ax.lines) == 2


def test_export_at_zero_kelvin_is_safe():
    """Exporting a T = 0 result (where S_m -> -inf) must not break YAML/CSV/JSON."""
    import os
    import tempfile

    from statthermopy.io import Exporter

    res = Thermodynamics(get("N2"), State(T=0.0, P=1e5)).compute()
    d = tempfile.mkdtemp()
    for fmt in ("yaml", "csv", "json"):
        p = os.path.join(d, f"out.{fmt}")
        getattr(Exporter(res), f"to_{fmt}")(p)
        assert os.path.getsize(p) > 0


def test_plot_mixture_property_varies_with_T():
    """A diatomic mixture's Cp_m must rise with T (not the flat line a monatomic
    pure gas would give): rotational and vibrational modes activate."""
    import pytest

    from statthermopy.mixture import IdealGasMixture
    from statthermopy.plots import plot_mixture_property

    mix = IdealGasMixture.from_names({"N2": 0.8, "O2": 0.2})
    ax = plot_mixture_property(mix, "Cp_m", [300, 800, 1500], P=1e5)
    assert ax is not None
    ys = ax.lines[0].get_ydata()
    # strictly increasing Cp_m over the range -> genuinely not a flat line
    assert ys[0] < ys[1] < ys[2]
    assert (ys[-1] - ys[0]) > 1.0  # J/mol/K of real curvature

    # partition-function factors are per-species: not available for a mixture
    with pytest.raises(ValueError, match="not available for a mixture"):
        plot_mixture_property(mix, "Qtotal", [300, 500], P=1e5)


# -- CLI ---------------------------------------------------------------------

def test_cli_gas_and_properties(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("list")
    sh.onecmd("gas N2")
    sh.onecmd("T = 298.15")
    sh.onecmd("P = 101325")
    sh.onecmd("properties")
    out = capsys.readouterr().out
    assert "Cp_m" in out and "29.1" in out


def test_cli_mixture(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("mixture Ar:0.7 N2:0.3")
    sh.onecmd("T = 298.15")
    sh.onecmd("P = 101325")
    sh.onecmd("properties")
    out = capsys.readouterr().out
    assert "M_avg" in out and "gamma" in out


def test_cli_fluid_air(capsys):
    """The `fluid Air` preset builds humid/dry air and prints the mixture report with the
    per-component breakdown, M_avg, R_specific and the entropy of mixing."""
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("fluids")
    assert "Air" in capsys.readouterr().out
    sh.onecmd("fluid Air h2o=0.01")
    sh.onecmd("T = 298.15")
    sh.onecmd("P = 101325")
    sh.onecmd("properties")
    out = capsys.readouterr().out
    assert "M_avg" in out and "R_specific" in out and "S_mixing" in out
    assert "Per-component" in out
    assert "H2O" in out and "N2" in out


def test_cli_run_fluid_air(capsys):
    """One-shot `run --fluid Air` reports the air mixture."""
    from statthermopy.cli.app import main
    rc = main(["run", "--fluid", "Air", "--T", "300", "--P", "101325"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Mixture" in out and "M_avg" in out


def test_cli_humidair(capsys):
    """`humidair` prints the saturation limit, psychrometrics and vapour breakdown."""
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("T = 298.15")
    sh.onecmd("P = 101325")
    sh.onecmd("humidair")
    out = capsys.readouterr().out
    assert "SATURATED" in out and "P_sat" in out
    assert "humidity_ratio,max" in out and "dew_point" in out and "wet_bulb" in out
    assert "partition-function contributions" in out
    # relative-humidity form
    sh.onecmd("humidair rh=0.5")
    out2 = capsys.readouterr().out
    assert "relative_humidity" in out2


def test_cli_run_humidair(capsys):
    """One-shot `humidair` subcommand runs and reports the saturated state."""
    from statthermopy.cli.app import main
    rc = main(["humidair", "--T", "293.15", "--P", "101325"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Humid Air" in out and "P_sat" in out


def test_cli_modes(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas H2O")
    sh.onecmd("T = 300")
    sh.onecmd("P = 1e5")
    sh.onecmd("modes")
    out = capsys.readouterr().out
    assert "Qt" in out and "rotational" in out


def test_cli_export(capsys, tmp_path):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas N2")
    sh.onecmd("T = 298.15")
    sh.onecmd("P = 101325")
    sh.onecmd("properties")
    p = tmp_path / "n2.json"
    sh.onecmd(f"export json {p}")
    assert p.exists()


def test_cli_no_gas_error(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("T = 300")
    sh.onecmd("properties")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_plot(tmp_path, capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas N2")
    out = tmp_path / "cp.png"
    sh.onecmd(f"plot Cp_m 300 1500 20 {out}")
    assert out.exists()


def test_cli_one_shot_run(capsys):
    from statthermopy.cli.app import main
    main(["run", "--gas", "CH4", "--T", "800", "--P", "5e5"])
    out = capsys.readouterr().out
    assert "Cp_m" in out


def test_cli_quit():
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    assert sh.onecmd("quit") is True

# -- mole-fraction display ----------------------------------------------------

class TestMoleFractionsAreNeverShownAsZero:
    """A component that is in the calculation must never be displayed as zero.

    Plot titles formatted the composition with two decimals, so a five-component biogas came
    out as ``CH4 0.56, CO2 0.44, O2 0.00, H2S 0.00, CO 0.00`` -- three components apparently
    absent, printed on top of a curve that was computed with all five.
    """

    BIOGAS = {"CH4": 0.557, "CO2": 0.439, "O2": 0.004, "H2S": 0.000045, "CO": 0.000009}

    @pytest.mark.parametrize(
        "x,decimals,expected",
        [
            (0.7808, 4, "0.7808"),       # ordinary fractions keep fixed point
            (0.0093, 4, "0.0093"),
            (0.0004, 4, "0.0004"),       # dry air's CO2: legible, so left alone
            (0.0040, 2, "4.00e-03"),     # two decimals would erase it
            (0.0040, 4, "0.0040"),       # four would not
            (4.5e-05, 4, "4.50e-05"),
            (9.0e-06, 4, "9.00e-06"),
            (0.0, 4, "0.0000"),          # a real zero may print as zero
        ],
    )
    def test_the_switch_happens_exactly_at_the_erasure_boundary(self, x, decimals, expected):
        from statthermopy.mixture import format_mole_fraction

        assert format_mole_fraction(x, decimals=decimals) == expected

    def _title(self, fn):
        import numpy as np

        from statthermopy.mixture import IdealGasMixture

        mix = IdealGasMixture.from_names(self.BIOGAS)
        return fn(mix, np.linspace(300.0, 1000.0, 10)).get_title()

    def test_property_plot_title_shows_every_component(self):
        from statthermopy.plots import plot_mixture_property

        title = self._title(lambda m, T: plot_mixture_property(m, "Cp_m", T))
        assert "O2 0.0040" in title
        assert "H2S 4.50e-05" in title and "CO 9.00e-06" in title
        assert " 0.00," not in title and not title.endswith(" 0.00")

    def test_thermal_fields_plot_title_shows_every_component(self):
        from statthermopy.plots import plot_mixture_thermal_fields

        title = self._title(plot_mixture_thermal_fields)
        assert "O2 0.0040" in title
        assert " 0.00," not in title

    def test_air_title_is_unchanged_where_it_was_already_legible(self):
        import numpy as np

        from statthermopy.mixture import IdealGasMixture
        from statthermopy.plots import plot_mixture_property

        mix = IdealGasMixture.from_names(
            {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0093, "CO2": 0.0004}
        )
        title = plot_mixture_property(mix, "Cp_m", np.linspace(300.0, 1000.0, 10)).get_title()
        assert "Ar 0.0093" in title and "CO2 0.0004" in title
        assert "e-0" not in title  # nothing needed scientific notation here


# -- HTML report --------------------------------------------------------------

class TestHtmlExport:
    """A standalone report: openable, printable and mailable on its own."""

    def _report(self, tmp_path, result, table=None):
        from statthermopy.io import Exporter

        path = Exporter(result, table=table).to_html(tmp_path / "report.html")
        return path, path.read_text(encoding="utf-8")

    def test_it_writes_a_complete_document(self, tmp_path, n2_result):
        path, text = self._report(tmp_path, n2_result)
        assert path.exists() and path.suffix == ".html"
        assert text.lstrip().startswith("<!doctype html>")
        assert text.rstrip().endswith("</body>")

    def test_it_is_self_contained(self, tmp_path, n2_result):
        """No external stylesheet, font, image or script: it must render offline."""
        _, text = self._report(tmp_path, n2_result)
        assert "<script" not in text
        assert "<link" not in text
        assert "http://" not in text and "https://" not in text

    def test_it_carries_the_identity(self, tmp_path, n2_result):
        _, text = self._report(tmp_path, n2_result)
        assert "StatThermoPy" in text
        assert "Statistical Thermodynamics in Python" in text

    def test_it_reports_state_and_properties(self, tmp_path, n2_result):
        _, text = self._report(tmp_path, n2_result)
        for section in ("<h2>State</h2>", "<h2>Properties</h2>", "<h2>All values</h2>"):
            assert section in text
        for label in ("Temperature", "Enthalpy", "Heat capacity (P)"):
            assert label in text

    def test_the_numbers_are_the_engine_numbers(self, tmp_path, n2_result):
        _, text = self._report(tmp_path, n2_result)
        assert f"{n2_result.Cp_m:.6g}" in text
        assert f"{n2_result.S_m:.6g}" in text

    def test_an_optional_table_is_appended(self, tmp_path, n2_result):
        _, text = self._report(
            tmp_path, n2_result, table={"T": [300.0, 400.0], "Cp_m": [29.1, 29.3]}
        )
        assert "<h2>Table</h2>" in text and "29.1" in text

    def test_a_mixture_names_its_species(self, tmp_path):
        from statthermopy import State
        from statthermopy.mixture import IdealGasMixture

        result = IdealGasMixture.from_names({"N2": 0.79, "O2": 0.21}).compute(
            State(T=300.0, P=101325.0)
        )
        _, text = self._report(tmp_path, result)
        assert "N2" in text and "O2" in text

    def test_species_names_are_escaped_not_injected(self, tmp_path, n2_result):
        """Keys and names are data; they must not be able to close a tag."""
        from statthermopy.io.exporters import _esc

        assert _esc("<b>x</b>") == "&lt;b&gt;x&lt;/b&gt;"

    def test_the_cli_writes_html(self, tmp_path, capsys):
        from statthermopy.cli.app import main

        out_file = tmp_path / "n2.html"
        assert main(["run", "--gas", "N2", "--T", "300", "--export", "html", str(out_file)]) == 0
        assert "exported ->" in capsys.readouterr().out
        assert out_file.exists() and out_file.stat().st_size > 0
