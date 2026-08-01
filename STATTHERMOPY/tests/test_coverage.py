"""Targeted tests to exercise error branches and rarely-used paths for coverage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from statthermopy import State, Thermodynamics, get
from statthermopy.constants import R


# -- State: all resolve branches --------------------------------------------

def test_state_both_P_and_V_given_consistent():
    # P and V both given -> used as-is.
    s = State(T=300.0, P=1e5, V=0.024943, n=1.0)
    rs = s.resolve(0.028)
    assert math.isclose(rs.P, 1e5)
    assert math.isclose(rs.V, 0.024943)


def test_state_V_only_derives_P():
    s = State(T=300.0, V=0.024943, n=1.0)
    rs = s.resolve(0.028)
    assert math.isclose(rs.P, R * 300.0 / 0.024943, rel_tol=1e-3)


def test_state_m_and_n_both_consistent():
    s = State(T=300.0, P=1e5, n=2.0, m=0.056)  # 2 mol * 0.028 = 0.056
    rs = s.resolve(0.028)
    assert math.isclose(rs.n, 2.0)


def test_state_m_and_n_inconsistent_raises():
    s = State(T=300.0, P=1e5, n=2.0, m=0.999)  # m != n*M
    with pytest.raises(ValueError):
        s.resolve(0.028)


def test_state_bad_molar_mass():
    s = State(T=300.0, P=1e5)
    with pytest.raises(ValueError):
        s.resolve(0.0)


def test_state_validation_negatives():
    with pytest.raises(ValueError):
        State(T=300.0, P=-1.0)
    with pytest.raises(ValueError):
        State(T=300.0, V=-1.0)
    with pytest.raises(ValueError):
        State(T=300.0, n=-1.0)
    with pytest.raises(ValueError):
        State(T=300.0, m=-1.0)


def test_state_repr():
    s = State(T=300.0, P=1e5, n=2.0, m=0.056)
    r = repr(s)
    assert "T=300" in r and "P=" in r and "n=2" in r


# -- Molecule: validation branches -------------------------------------------

def test_molecule_linear_wrong_atom_count():
    from statthermopy import Geometry, Molecule
    with pytest.raises(ValueError):
        Molecule(name="X", formula="X", molar_mass_gmol=10.0,
                 geometry=Geometry.LINEAR, n_atoms=1)


def test_molecule_nonlinear_too_few_atoms():
    from statthermopy import Geometry, Molecule
    with pytest.raises(ValueError):
        Molecule(name="X", formula="X", molar_mass_gmol=10.0,
                 geometry=Geometry.NONLINEAR, n_atoms=2)


def test_molecule_nonlinear_wrong_moments_count():
    from statthermopy import Geometry, Molecule, VibrationalMode
    with pytest.raises(ValueError):
        Molecule(name="X", formula="X", molar_mass_gmol=10.0,
                 geometry=Geometry.NONLINEAR, n_atoms=3,
                 moments_of_inertia=(1.0, 2.0))


def test_molecule_bad_n_atoms():
    from statthermopy import Geometry, Molecule
    with pytest.raises(ValueError):
        Molecule(name="X", formula="X", molar_mass_gmol=10.0,
                 geometry=Geometry.MONOATOMIC, n_atoms=0)


def test_molecule_bad_symmetry():
    from statthermopy import Geometry, Molecule
    with pytest.raises(ValueError):
        Molecule(name="X", formula="X", molar_mass_gmol=10.0,
                 geometry=Geometry.MONOATOMIC, n_atoms=1, symmetry_number=0)


def test_molecule_linear_wrong_oscillator_count():
    from statthermopy import Geometry, Molecule, VibrationalMode
    with pytest.raises(ValueError):
        Molecule(name="X", formula="X", molar_mass_gmol=10.0,
                 geometry=Geometry.LINEAR, n_atoms=3,
                 moments_of_inertia=(1.0,),
                 vibrational_modes=(VibrationalMode(100.0, 1),))  # 1 != 4


# -- Mode base default contribution ------------------------------------------

def test_base_mode_default_contribution():
    from statthermopy.modes.base import Mode
    from statthermopy import State

    class Dummy(Mode):
        name = "dummy"
        def ln_q(self, state):
            return 0.5 * math.log(state.T)
        def d_ln_q_dT(self, state):
            return 0.5 / state.T
        def cv_m(self, state):
            return 0.5 * R

    rs = State(T=400.0, P=1e5, n=1.0).resolve(0.028)
    c = Dummy().contribution(rs)
    assert math.isclose(c.U_m, R * 400.0 * 400.0 * (0.5 / 400.0))
    assert math.isclose(c.S_m, R * (0.5 * math.log(400.0) + 0.5))
    assert math.isclose(c.A_m, -R * 400.0 * 0.5 * math.log(400.0))


# -- Vibrational standalone methods -----------------------------------------

def test_vibrational_standalone_methods():
    from statthermopy.core.molecule import VibrationalMode
    from statthermopy.modes import Vibrational
    vib = Vibrational((VibrationalMode(1500.0, 1),))
    rs = State(T=800.0, P=1e5, n=1.0).resolve(0.028)
    assert vib.ln_q(rs) == vib.contribution(rs).ln_q
    assert math.isclose(vib.d_ln_q_dT(rs), vib.contribution(rs).d_ln_q_dT, rel_tol=1e-9)
    assert math.isclose(vib.cv_m(rs), vib.contribution(rs).Cv_m, rel_tol=1e-9)


def test_vibrational_empty_standalone():
    from statthermopy.modes import Vibrational
    vib = Vibrational(())
    rs = State(T=800.0, P=1e5, n=1.0).resolve(0.028)
    assert vib.ln_q(rs) == 0.0
    assert vib.d_ln_q_dT(rs) == 0.0
    assert vib.cv_m(rs) == 0.0


# -- Rotational quantum fallback / standalone -------------------------------

def test_rotational_linear_quantum_standalone():
    from statthermopy import Geometry
    from statthermopy.modes import Rotational
    from statthermopy.constants import h, k_B
    theta = 2.878
    I = h * h / (8 * math.pi**2 * theta * k_B)
    rot = Rotational(Geometry.LINEAR, 2, (I,), use_quantum=True)
    rs = State(T=300.0, P=1e5, n=1.0).resolve(0.028)
    # ln_q via quantum sum matches the standalone _q_linear_quantum
    assert math.isclose(rot.ln_q(rs), math.log(rot._q_linear_quantum(300.0)), rel_tol=1e-9)


# -- Exporters: table paths, nested flatten, native conversion ---------------

@pytest.fixture
def n2_result_with_table():
    res = Thermodynamics(get("N2"), State(T=298.15, P=101325.0)).compute()
    table = {"T": [300.0, 500.0, 1000.0], "Cp_m": [29.12, 29.26, 32.5]}
    return res, table


def test_export_csv_with_table(tmp_path, n2_result_with_table):
    from statthermopy.io import Exporter
    res, table = n2_result_with_table
    p = Exporter(res, table=table).to_csv(tmp_path / "n2.csv")
    text = Path(p).read_text(encoding="utf-8")
    assert "T" in text and "29.12" in text


def test_export_json_with_table(tmp_path, n2_result_with_table):
    from statthermopy.io import Exporter
    res, table = n2_result_with_table
    p = Exporter(res, table=table).to_json(tmp_path / "n2.json")
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    assert "table" in data and len(data["table"]["T"]) == 3


def test_export_yaml_with_table(tmp_path, n2_result_with_table):
    from statthermopy.io import Exporter
    res, table = n2_result_with_table
    p = Exporter(res, table=table).to_yaml(tmp_path / "n2.yaml")
    text = Path(p).read_text(encoding="utf-8")
    assert "table" in text


def test_export_excel_with_table(tmp_path, n2_result_with_table):
    from statthermopy.io import Exporter
    res, table = n2_result_with_table
    p = Exporter(res, table=table).to_excel(tmp_path / "n2.xlsx")
    assert Path(p).stat().st_size > 0


def test_exporter_as_dict_fallback():
    from statthermopy.io.exporters import _as_dict
    with pytest.raises(TypeError):
        _as_dict(123)


def test_native_conversion_numpy():
    from statthermopy.io.exporters import _native
    assert _native(np.float64(1.5)) == 1.5
    assert _native(np.array([1.0, 2.0])) == [1.0, 2.0]
    assert _native({"a": np.float64(2.0)}) == {"a": 2.0}
    assert _native([np.int64(3)]) == [3]


# -- CLI: error branches and remaining commands -----------------------------

def test_cli_list(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("list")
    out = capsys.readouterr().out
    assert "gases" in out


def test_cli_gas_no_arg(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas")
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_cli_gas_unknown(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas notarealthing")
    out = capsys.readouterr().out
    assert "error" in out.lower() or "unknown" in out.lower()


def test_cli_set_all_state_vars(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas Ar")
    sh.onecmd("T = 300")
    sh.onecmd("P = 1e5")
    sh.onecmd("V = 0.024")
    sh.onecmd("n = 1")
    sh.onecmd("m = 0.04")
    sh.onecmd("state")
    out = capsys.readouterr().out
    assert "T = 300" in out and "gas:" in out


def test_cli_set_bad_number(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("T = abc")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_properties_no_T(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas N2")
    sh.onecmd("P = 1e5")
    sh.onecmd("properties")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_modes_no_gas(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("modes")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_mixture_no_args(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("mixture")
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_cli_mixture_bad_token(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("mixture Ar-0.7 N2:0.3")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_mixture_bad_fraction(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("mixture Ar:xx N2:0.3")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_mixture_unknown_species(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("mixture Unknownx:0.5 N2:0.5")
    out = capsys.readouterr().out
    assert "error" in out.lower() or "unknown" in out.lower()


def test_cli_plot_no_args(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas N2")
    sh.onecmd("plot")
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_cli_plot_no_gas(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("plot Cp_m 300 1500")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_export_no_args(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("export")
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_cli_export_nothing_to_export(capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("export csv out.csv")
    out = capsys.readouterr().out
    assert "error" in out.lower() or "nothing" in out.lower()


def test_cli_export_unknown_format(capsys, tmp_path):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas N2")
    sh.onecmd("T = 298.15")
    sh.onecmd("P = 101325")
    sh.onecmd("properties")
    sh.onecmd(f"export zzz {tmp_path / 'x'}")
    out = capsys.readouterr().out
    assert "error" in out.lower()


def test_cli_export_all_formats(tmp_path, capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas N2")
    sh.onecmd("T = 298.15")
    sh.onecmd("P = 101325")
    sh.onecmd("properties")
    for fmt, ext in [("csv", "csv"), ("json", "json"), ("yaml", "yaml"),
                     ("latex", "tex")]:
        p = tmp_path / f"n2.{ext}"
        sh.onecmd(f"export {fmt} {p}")
        assert p.exists()


def test_cli_plot_with_windows_path(tmp_path, capsys):
    from statthermopy.cli.app import StatThermoPyShell
    sh = StatThermoPyShell()
    sh.onecmd("gas N2")
    out = tmp_path / "cp.png"
    sh.onecmd(f"plot Cv_m 300 1500 10 {out}")
    assert out.exists()


def test_cli_split_fallback(monkeypatch):
    from statthermopy.cli.app import _split
    # Force shlex.split to raise to exercise the except branch.
    import statthermopy.cli.app as appmod
    monkeypatch.setattr(appmod.shlex, "split", lambda *a, **k: (_ for _ in ()).throw(ValueError()))
    assert _split("a b c") == ["a", "b", "c"]


# -- Backend: exercise numpy backend methods directly -----------------------

def test_numpy_backend_all_methods():
    from statthermopy.backend import get_backend
    b = get_backend()
    arr = b.asarray([1.0, 2.0, 3.0])
    assert math.isclose(b.sum(arr), 6.0)
    assert math.isclose(float(b.exp(b.asarray([0.0]))[0]), 1.0)
    assert math.isclose(float(b.expm1(b.asarray([0.0]))[0]), 0.0)
    assert math.isclose(float(b.log(b.asarray([math.e]))[0]), 1.0)
    assert math.isclose(float(b.log1p(b.asarray([0.0]))[0]), 0.0)


# -- Validation: molar-attribute resolution & repr --------------------------

def test_validation_molar_attr_resolution():
    from statthermopy.validation import ValidationRunner

    class Src:
        def load(self):
            import pandas as pd
            return pd.DataFrame({"T": [300.0], "Cp_m": [
                Thermodynamics(get("N2"), State(T=300.0, P=1e5)).compute().Cp_m]})

    r = ValidationRunner("N2", Src(), property_name="Cp_m").run()
    assert abs(r.errors_percent[0]) < 1e-6
    assert "ValidationReport" in repr(r)


# -- PartitionFunction modes dict & repr ------------------------------------

def test_partition_modes_dict():
    n2 = get("N2")
    pf = Thermodynamics(n2, State(T=300.0, P=1e5)).partition
    md = pf.modes
    assert set(md.keys()) == {
        "translational", "rotational", "vibrational", "internal_rotation", "electronic",
    }


# -- argparse one-shot mode -------------------------------------------------

def test_cli_main_run_mixture(capsys):
    from statthermopy.cli.app import main
    rc = main(["run", "--mixture", "Ar:0.7", "N2:0.3", "--T", "300", "--P", "1e5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "M_avg" in out


def test_cli_main_run_mixture_mass_basis(capsys):
    from statthermopy.cli.app import main
    main(["run", "--mixture", "Ar:0.5", "H2:0.5", "--basis", "mass", "--T", "300"])
    out = capsys.readouterr().out
    assert "M_avg" in out


def test_cli_main_run_gas_with_export(capsys, tmp_path):
    from statthermopy.cli.app import main
    p = tmp_path / "n2.json"
    main(["run", "--gas", "N2", "--T", "300", "--P", "1e5",
          "--export", "json", str(p)])
    assert p.exists()


def test_cli_main_run_mixture_with_export(capsys, tmp_path):
    from statthermopy.cli.app import main
    p = tmp_path / "mix.json"
    main(["run", "--mixture", "Ar:0.7", "N2:0.3", "--T", "300",
          "--export", "json", str(p)])
    assert p.exists()


def test_cli_main_default_repl_exits_cleanly(monkeypatch):
    import io as _io
    from statthermopy.cli.app import main
    monkeypatch.setattr("sys.stdin", _io.StringIO("quit\n"))
    assert main([]) == 0