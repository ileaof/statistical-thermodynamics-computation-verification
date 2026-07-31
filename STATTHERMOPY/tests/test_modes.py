"""Tests for the four partition-function modes."""

from __future__ import annotations

import math

import pytest

from statthermopy import Geometry, State
from statthermopy.constants import R
from statthermopy.modes import Electronic, Rotational, Translational, Vibrational
from statthermopy.modes.rotational import rotational_temperature


def _rs(T=300.0, P=1e5, n=1.0, M=0.028):
    return State(T=T, P=P, n=n).resolve(M)


# -- Translational -----------------------------------------------------------

def test_translational_internal_energy_and_cv():
    tr = Translational(molecular_mass=4.65e-26)  # ~ N2 molecule mass
    rs = _rs()
    c = tr.contribution(rs)
    assert math.isclose(c.U_m, 1.5 * R * rs.T, rel_tol=1e-12)
    assert math.isclose(c.Cv_m, 1.5 * R, rel_tol=1e-12)


def test_translational_entropy_sackur_tetrode():
    # N2: m = 28.0134 amu, S_m(298.15, 1 atm) ~ 150.3 J/mol/K for translation alone.
    from statthermopy.constants import amu
    m = 28.0134 * amu
    tr = Translational(molecular_mass=m)
    rs = State(T=298.15, P=101325.0, n=1.0).resolve(0.0280134)
    c = tr.contribution(rs)
    assert math.isclose(c.S_m, 150.30, abs_tol=0.05)


def test_translational_d_ln_q_dT():
    tr = Translational(molecular_mass=1e-26)
    rs = _rs(T=500.0)
    assert math.isclose(tr.d_ln_q_dT(rs), 1.5 / 500.0, rel_tol=1e-12)


def test_translational_rejects_bad_mass():
    with pytest.raises(ValueError):
        Translational(molecular_mass=-1.0)


# -- Rotational --------------------------------------------------------------

def test_rotational_temperature_formula():
    # theta_rot = h^2/(8 pi^2 I k)
    I = 1e-46
    theta = rotational_temperature(I)
    from statthermopy.constants import h, k_B
    assert math.isclose(theta, h * h / (8 * math.pi**2 * I * k_B), rel_tol=1e-12)


def test_rotational_monoatomic_zero():
    rot = Rotational(Geometry.MONOATOMIC, 1, ())
    rs = _rs()
    c = rot.contribution(rs)
    assert c.ln_q == 0.0 and c.U_m == 0.0 and c.S_m == 0.0 and c.Cv_m == 0.0


def test_rotational_linear_classical():
    # N2-like: theta_rot=2.878 K, sigma=2 -> Qr = T/(sigma*theta)
    from statthermopy.constants import h, k_B
    theta = 2.878
    I = h * h / (8 * math.pi**2 * theta * k_B)
    rot = Rotational(Geometry.LINEAR, 2, (I,))
    rs = _rs(T=298.15)
    c = rot.contribution(rs)
    assert math.isclose(c.U_m, R * rs.T, rel_tol=1e-9)
    assert math.isclose(c.Cv_m, R, rel_tol=1e-9)
    expected_lnqr = math.log(rs.T / (2 * theta))
    assert math.isclose(c.ln_q, expected_lnqr, rel_tol=1e-9)


def test_rotational_nonlinear_classical():
    # nonlinear -> U_m = 1.5 R T, Cv = 1.5 R
    from statthermopy.constants import h, k_B
    thetas = [40.13, 20.87, 13.36]  # H2O-like
    moments = tuple(h * h / (8 * math.pi**2 * t * k_B) for t in thetas)
    rot = Rotational(Geometry.NONLINEAR, 2, moments)
    rs = _rs(T=300.0)
    c = rot.contribution(rs)
    assert math.isclose(c.U_m, 1.5 * R * rs.T, rel_tol=1e-9)
    assert math.isclose(c.Cv_m, 1.5 * R, rel_tol=1e-9)


def test_rotational_linear_quantum_matches_classical_at_high_T():
    # At high T the quantum sum converges to the classical limit U -> R T.
    from statthermopy.constants import h, k_B
    theta = 2.878
    I = h * h / (8 * math.pi**2 * theta * k_B)
    rot_q = Rotational(Geometry.LINEAR, 2, (I,), use_quantum=True)
    rs = _rs(T=1000.0)
    c = rot_q.contribution(rs)
    assert math.isclose(c.U_m, R * rs.T, rel_tol=1e-3)  # ~ RT, within 0.1%
    assert c.Cv_m < R + 0.2  # approaches R from below


def test_rotational_linear_quantum_low_T_freezing():
    # At very low T (T << theta), rotation freezes: U -> 0, Cv -> 0.
    from statthermopy.constants import h, k_B
    theta = 85.3  # H2
    I = h * h / (8 * math.pi**2 * theta * k_B)
    rot_q = Rotational(Geometry.LINEAR, 2, (I,), use_quantum=True)
    rs = _rs(T=10.0)
    c = rot_q.contribution(rs)
    assert c.U_m < 0.05 * R * rs.T  # far below RT
    assert c.Cv_m < 0.1 * R


# -- Vibrational -------------------------------------------------------------

def test_vibrational_zero_modes():
    vib = Vibrational(())
    rs = _rs()
    c = vib.contribution(rs)
    assert c.ln_q == 0.0 and c.U_m == 0.0 and c.Cv_m == 0.0


def test_vibrational_closed_form_formula():
    # Verify U_m and Cv_m against the independent quantum HO closed forms.
    from statthermopy.core.molecule import VibrationalMode
    from statthermopy.units import CM1_TO_K
    theta = 1500.0 * CM1_TO_K  # K  (1500 cm-1, exact conversion)
    vib = Vibrational((VibrationalMode(1500.0, 1),))
    rs = _rs(T=800.0)
    c = vib.contribution(rs)
    x = theta / rs.T
    ex = math.exp(x)
    U_ref = R * theta / (ex - 1.0)
    Cv_ref = R * x * x * ex / ((ex - 1.0) ** 2)
    assert math.isclose(c.U_m, U_ref, rel_tol=1e-9)
    assert math.isclose(c.Cv_m, Cv_ref, rel_tol=1e-9)
    # S_m closed form
    S_ref = R * (x / (ex - 1.0) - math.log1p(-math.exp(-x)))
    assert math.isclose(c.S_m, S_ref, rel_tol=1e-9)


def test_vibrational_high_T_classical_limit():
    # At very high T the harmonic oscillator approaches equipartition: Cv -> R, U -> R T.
    from statthermopy.core.molecule import VibrationalMode
    theta = 300.0  # K
    wn = theta / 1.438777  # cm-1
    vib = Vibrational((VibrationalMode(wn, 1),))
    rs = _rs(T=20000.0)  # theta/T = 0.015 -> within ~1% of classical limit
    c = vib.contribution(rs)
    assert math.isclose(c.Cv_m, R, rel_tol=1e-2)
    assert math.isclose(c.U_m, R * rs.T, rel_tol=1e-2)


def test_vibrational_low_T_freezing():
    # At T << theta_v, vibrational mode frozen: U -> 0, Cv -> 0.
    from statthermopy.core.molecule import VibrationalMode
    theta = 3000.0
    wn = theta / 1.438777
    vib = Vibrational((VibrationalMode(wn, 1),))
    rs = _rs(T=50.0)
    c = vib.contribution(rs)
    assert c.U_m < 0.01 * R * rs.T
    assert c.Cv_m < 1e-3 * R


def test_vibrational_degeneracy_doubles_contribution():
    from statthermopy.core.molecule import VibrationalMode
    wn = 1000.0
    one = Vibrational((VibrationalMode(wn, 1),))
    two = Vibrational((VibrationalMode(wn, 2),))
    rs = _rs(T=1000.0)
    c1 = one.contribution(rs)
    c2 = two.contribution(rs)
    assert math.isclose(c2.U_m, 2 * c1.U_m, rel_tol=1e-9)
    assert math.isclose(c2.S_m, 2 * c1.S_m, rel_tol=1e-9)


def test_vibrational_derivative_consistency():
    # d ln Q / dT should equal numerical derivative of ln_q.
    from statthermopy.core.molecule import VibrationalMode
    vib = Vibrational((VibrationalMode(1500.0, 1),))
    rs = _rs(T=800.0)
    dT = 1e-3
    ln0 = vib.ln_q(_rs(T=800.0 - dT))
    ln1 = vib.ln_q(_rs(T=800.0 + dT))
    num = (ln1 - ln0) / (2 * dT)
    assert math.isclose(vib.d_ln_q_dT(rs), num, rel_tol=1e-4)


# -- Electronic --------------------------------------------------------------

def test_electronic_ground_state_only():
    from statthermopy.core.molecule import ElectronicLevel
    el = Electronic((ElectronicLevel(0.0, 1),))
    rs = _rs()
    c = el.contribution(rs)
    assert c.ln_q == 0.0
    assert c.U_m == 0.0 and c.Cv_m == 0.0
    assert math.isclose(el.populations(rs.T)[0], 1.0)


def test_electronic_degenerate_ground_state():
    # O2-like: ground g=3, no excited states populated at low T
    from statthermopy.core.molecule import ElectronicLevel
    el = Electronic((ElectronicLevel(0.0, 3),))
    rs = _rs(T=298.15)
    c = el.contribution(rs)
    assert math.isclose(c.ln_q, math.log(3.0), rel_tol=1e-12)
    assert math.isclose(c.U_m, 0.0, abs_tol=1e-9)


def test_electronic_excited_state_population():
    from statthermopy.core.molecule import ElectronicLevel
    # NO-like: ground g=2 at 0, excited g=2 at 121.1 cm-1
    el = Electronic((ElectronicLevel(0.0, 2), ElectronicLevel(121.1, 2)))
    rs = _rs(T=300.0)
    pops = el.populations(rs.T)
    assert math.isclose(sum(pops), 1.0, rel_tol=1e-9)
    assert pops[1] > 0.0  # excited state populated
    assert pops[0] > pops[1]


def test_electronic_requires_at_least_one_level():
    with pytest.raises(ValueError):
        Electronic(())


def test_electronic_derivative_consistency():
    from statthermopy.core.molecule import ElectronicLevel
    el = Electronic((ElectronicLevel(0.0, 2), ElectronicLevel(121.1, 2)))
    rs = _rs(T=300.0)
    dT = 1e-3
    ln0 = el.ln_q(_rs(T=300.0 - dT))
    ln1 = el.ln_q(_rs(T=300.0 + dT))
    num = (ln1 - ln0) / (2 * dT)
    assert math.isclose(el.d_ln_q_dT(rs), num, rel_tol=1e-4)