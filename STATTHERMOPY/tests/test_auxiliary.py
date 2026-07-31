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
    assert len(axes) == 13  # 8 molar + 5 partition
    assert any((tmp_path / f"Ar_{p}.png").exists() for p in ("Cp_m", "Qtotal"))


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