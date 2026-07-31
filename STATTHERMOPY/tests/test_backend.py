"""Tests for the pluggable numerical backend (NumPy + accelerated Numba/OpenMP/CUDA).

The accelerated-backend tests are gated on ``numba`` being importable. The NumPy path and the
backend registry are always exercised. An autouse fixture restores the active backend to NumPy
after every test so global state never leaks between tests.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from statthermopy import State, Thermodynamics
from statthermopy.backend import (
    Backend,
    NumpyBackend,
    available_backends,
    get_backend,
    list_backends,
    set_backend,
)
from statthermopy.database import get

# Tight tolerance: the @njit kernels mirror the NumPy path's closed forms and agree to machine
# precision.
_REL = 1e-9


@pytest.fixture(autouse=True)
def _restore_numpy_backend():
    set_backend("numpy")
    yield
    set_backend("numpy")


# ---------------------------------------------------------------------------
# Always-on: NumPy backend + registry (no numba required)
# ---------------------------------------------------------------------------

def test_default_backend_is_numpy():
    assert get_backend().name == "numpy"
    assert isinstance(get_backend(), NumpyBackend)


def test_rewire_preserves_values():
    # After routing Vibrational/Electronic through the backend, the N2 reference values are
    # unchanged from Phase 1.
    p = Thermodynamics(get("N2"), State(T=298.15, P=101325.0)).compute()
    assert p.Cp_m == pytest.approx(29.1129, abs=1e-4)
    assert p.gamma == pytest.approx(1.3998, abs=1e-4)
    assert p.S_m == pytest.approx(191.4458, abs=1e-3)


def test_get_backend_by_name_numpy():
    assert get_backend("numpy").name == "numpy"
    assert isinstance(get_backend("numpy"), NumpyBackend)


def test_set_backend_by_instance_then_restore():
    nb = NumpyBackend()
    set_backend(nb)
    assert get_backend() is nb
    set_backend("numpy")
    assert get_backend().name == "numpy"


def test_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("definitely_not_a_backend")


def test_get_backend_rejects_non_string_name():
    with pytest.raises(TypeError, match="must be a string"):
        get_backend(123)  # type: ignore[arg-type]


def test_set_backend_rejects_invalid_type():
    with pytest.raises(TypeError, match="expects a backend name"):
        set_backend(123)  # type: ignore[arg-type]


def test_list_backends_declared():
    assert list_backends() == ["numpy", "numba", "openmp", "cuda"]


def test_available_backends_includes_numpy():
    av = available_backends()
    assert av[0] == "numpy"
    assert "numpy" in av


def test_available_backends_with_numba():
    av = available_backends()
    # numba is importable in this test environment (dev extras); numba/openmp then present.
    assert "numba" in av
    assert "openmp" in av
    # cuda only if a GPU is actually detected.
    assert ("cuda" in av) == bool(numba.cuda.is_available())


def test_default_kernels_are_none_on_numpy():
    be = get_backend("numpy")
    assert be.linear_quantum_moments(1.0, 300.0, 150) is None
    assert be.molar_property_grid(get("N2"), [300.0, 500.0], 101325.0, False) is None


def test_property_vs_T_numpy_fallback_matches_per_T():
    # With the NumPy backend (molar_property_grid -> None), property_vs_T uses the per-T loop and
    # matches Thermodynamics.compute() at each point.
    mol = get("N2")
    th = Thermodynamics(mol, State(T=298.15, P=101325.0))
    Ts = [298.15, 600.0, 1000.0, 2000.0]
    _, cps = th.property_vs_T("Cp_m", Ts, P=101325.0)
    for T, c in zip(Ts, cps):
        assert c == pytest.approx(
            Thermodynamics(mol, State(T=T, P=101325.0)).compute().Cp_m, rel=_REL
        )


def test_property_vs_T_partition_factors_numpy():
    mol = get("CO2")
    th = Thermodynamics(mol, State(T=298.15, P=101325.0))
    _, qtot = th.property_vs_T("Qtotal", [298.15, 1000.0], P=101325.0)
    for T, q in zip([298.15, 1000.0], qtot):
        assert q == pytest.approx(
            Thermodynamics(mol, State(T=T, P=101325.0)).compute().Qtotal, rel=_REL
        )


def test_import_statthermopy_does_not_load_numba():
    # `import statthermopy` must stay light: numba (heavy) is imported lazily only on backend
    # selection.
    code = (
        "import sys, statthermopy; "
        "assert 'numba' not in sys.modules, sys.modules.get('numba'); "
        "assert 'statthermopy.gui' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# Accelerated backends (require numba)
# ---------------------------------------------------------------------------

numba = pytest.importorskip("numba")

_CLASSICAL = ["N2", "O2", "CO2", "CH4", "AR", "H2O", "H2", "NO"]
_TGRID = [298.15, 400.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0]
_PROPS = ["Cp_m", "Cv_m", "H_m", "S_m", "G_m", "A_m", "U_m", "gamma", "Qtotal", "ln_Qtotal"]


def _numpy_grid(species, prop, use_quantum=False):
    set_backend("numpy")
    mol = get(species)
    th = Thermodynamics(mol, State(T=298.15, P=101325.0), use_quantum_rotation=use_quantum)
    return th.property_vs_T(prop, _TGRID, P=101325.0)[1]


@pytest.mark.parametrize("species", _CLASSICAL)
def test_numba_property_grid_matches_numpy(species):
    # Compute every numpy reference first (while the backend is numpy), then switch to numba
    # and compare — avoiding flipping the global backend mid-loop.
    refs = {prop: _numpy_grid(species, prop) for prop in _PROPS}
    set_backend("numba")
    mol = get(species)
    th = Thermodynamics(mol, State(T=298.15, P=101325.0))
    for prop in _PROPS:
        _, vals = th.property_vs_T(prop, _TGRID, P=101325.0)
        for a, b in zip(refs[prop], vals):
            assert a == pytest.approx(b, rel=_REL, abs=_REL), f"{species}/{prop}"


def test_numba_single_T_compute_matches_numpy():
    set_backend("numba")
    p = Thermodynamics(get("CO2"), State(T=1000.0, P=101325.0)).compute()
    set_backend("numpy")
    ref = Thermodynamics(get("CO2"), State(T=1000.0, P=101325.0)).compute()
    for attr in ("Cp_m", "S_m", "H_m", "G_m", "Cv_m", "gamma", "Qtotal"):
        assert getattr(p, attr) == pytest.approx(getattr(ref, attr), rel=_REL)


@pytest.mark.parametrize("species", ["H2", "NO", "N2"])
def test_numba_quantum_rotation_matches_numpy(species):
    qgrid = [50.0, 100.0, 150.0, 200.0, 298.15, 500.0, 1000.0]
    set_backend("numba")
    mol = get(species)
    th = Thermodynamics(mol, State(T=100.0, P=101325.0), use_quantum_rotation=True)
    _, nb = th.property_vs_T("Cp_m", qgrid, P=101325.0)
    set_backend("numpy")
    _, ref = Thermodynamics(
        mol, State(T=100.0, P=101325.0), use_quantum_rotation=True
    ).property_vs_T("Cp_m", qgrid, P=101325.0)
    for a, b in zip(ref, nb):
        assert a == pytest.approx(b, rel=1e-7, abs=1e-7), f"{species}"


def test_numba_linear_quantum_moments_matches_python():
    set_backend("numba")
    be = get_backend()
    # H2 theta_rot ~ 85.3 K; compare the kernel against the pure-Python loop.
    from statthermopy.modes.rotational import rotational_temperature

    mol = get("H2")
    theta = rotational_temperature(mol.moments_of_inertia[0])
    for T in (50.0, 100.0, 298.15, 1000.0):
        q_nb, m1_nb, m2_nb = be.linear_quantum_moments(theta, T, 150)
        # pure-python reference loop
        q = s1 = s2 = 0.0
        beta = theta / T
        import math
        for J in range(151):
            y = J * (J + 1) * beta
            term = (2 * J + 1) * math.exp(-y)
            q += term
            s1 += term * y
            s2 += term * y * y
            if term < 1e-15 * q and J > 5:
                break
        assert q_nb == pytest.approx(q, rel=1e-9)
        assert m1_nb == pytest.approx(s1 / q, rel=1e-9)
        assert m2_nb == pytest.approx(s2 / q, rel=1e-9)


def test_openmp_property_grid_matches_numpy():
    refs = {prop: _numpy_grid("CO2", prop) for prop in _PROPS}
    set_backend("openmp")
    mol = get("CO2")
    th = Thermodynamics(mol, State(T=298.15, P=101325.0))
    for prop in _PROPS:
        _, vals = th.property_vs_T(prop, _TGRID, P=101325.0)
        for a, b in zip(refs[prop], vals):
            assert a == pytest.approx(b, rel=_REL), f"openmp/{prop}"


def test_cuda_backend_matches_numpy():
    # On a GPU-less machine this exercises the fallback (warn + delegate to Numba); on a
    # GPU machine it exercises the CUDA kernel. Both must match the NumPy path.
    ref = _numpy_grid("N2", "Cp_m")
    set_backend("cuda")
    mol = get("N2")
    th = Thermodynamics(mol, State(T=298.15, P=101325.0))
    _, vals = th.property_vs_T("Cp_m", _TGRID, P=101325.0)
    for a, b in zip(ref, vals):
        assert a == pytest.approx(b, rel=1e-7), "cuda"


def test_cuda_fallback_warns_without_gpu():
    if numba.cuda.is_available():
        pytest.skip("GPU present: fallback path not exercised")
    with pytest.warns(RuntimeWarning, match="no NVIDIA GPU"):
        from statthermopy.backend.cuda_backend import CudaBackend
        CudaBackend()


def test_cuda_compute_matches_numpy():
    # A full compute() with the CUDA backend drives its array methods (delegated to NumPy) via the
    # modes, and the quantum-rotation path drives its linear_quantum_moments (-> Numba).
    set_backend("cuda")
    p = Thermodynamics(
        get("H2"), State(T=100.0, P=101325.0), use_quantum_rotation=True
    ).compute()
    set_backend("numpy")
    ref = Thermodynamics(
        get("H2"), State(T=100.0, P=101325.0), use_quantum_rotation=True
    ).compute()
    for attr in ("Cp_m", "Cv_m", "S_m", "H_m", "G_m", "gamma", "Qtotal", "ln_Qtotal"):
        assert getattr(p, attr) == pytest.approx(getattr(ref, attr), rel=1e-7), attr


def test_validate_with_numba_backend():
    from statthermopy.validation import validate

    set_backend("numba")
    rep = validate("N2", "Cp")
    assert rep.mean_abs_error_percent < 5.0
    # the accelerated path agrees with the numpy validation path
    set_backend("numpy")
    rep_np = validate("N2", "Cp")
    assert rep.predicted == pytest.approx(rep_np.predicted, rel=_REL)


def test_property_vs_T_large_grid_completes():  # pragma: no cover - smoke / timing
    set_backend("numba")
    mol = get("N2")
    th = Thermodynamics(mol, State(T=298.15, P=101325.0))
    Ts = np.linspace(300.0, 3000.0, 2000)
    _, cps = th.property_vs_T("Cp_m", Ts, P=101325.0)
    assert len(cps) == 2000
    assert cps[0] == pytest.approx(29.11, abs=0.1)