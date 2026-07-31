"""Tests for core data structures: Molecule, State, Contribution."""

from __future__ import annotations

import math

import pytest

from statthermopy import Geometry, Molecule, State, VibrationalMode, ElectronicLevel
from statthermopy import units as u


# -- Molecule ----------------------------------------------------------------

def _h2():
    return Molecule(
        name="H2", formula="H2", molar_mass_gmol=2.01588, geometry=Geometry.LINEAR,
        n_atoms=2, symmetry_number=2,
        moments_of_inertia=(4.6e-48,),
        vibrational_modes=(VibrationalMode(4401.0, 1),),
        electronic_levels=(ElectronicLevel(0.0, 1),),
    )


def test_molecule_molar_mass_si():
    m = _h2()
    assert math.isclose(m.molar_mass, 0.00201588)
    assert math.isclose(m.molar_mass_gmol, 2.01588)


def test_molecule_geometry_flags():
    m = _h2()
    assert m.is_linear and m.is_diatomic
    assert not m.is_monoatomic and not m.is_nonlinear


def test_molecule_molecular_mass():
    m = _h2()
    assert math.isclose(m.molecular_mass, m.molar_mass / 6.02214076e23)


def test_molecule_monoatomic_no_rotation():
    ar = Molecule(name="Ar", formula="Ar", molar_mass_gmol=39.948,
                  geometry=Geometry.MONOATOMIC, n_atoms=1)
    assert ar.is_monoatomic
    assert ar.moments_of_inertia == ()
    assert ar.vibrational_modes == ()
    assert ar.ground_state_degeneracy == 1  # default ground state


def test_molecule_default_electronic_ground_state():
    m = Molecule(name="X", formula="X", molar_mass_gmol=10.0,
                 geometry=Geometry.MONOATOMIC, n_atoms=1)
    assert len(m.electronic_levels) == 1
    assert m.electronic_levels[0].degeneracy == 1


def test_molecule_validation_errors():
    # monoatomic with rotation/vibration -> error
    with pytest.raises(ValueError):
        Molecule(name="Bad", formula="Bad", molar_mass_gmol=10.0,
                 geometry=Geometry.MONOATOMIC, n_atoms=1,
                 moments_of_inertia=(1.0,))
    # linear with wrong number of moments
    with pytest.raises(ValueError):
        Molecule(name="Bad2", formula="Bad2", molar_mass_gmol=10.0,
                 geometry=Geometry.LINEAR, n_atoms=2,
                 moments_of_inertia=(1.0, 2.0))
    # nonlinear with wrong oscillator count
    with pytest.raises(ValueError):
        Molecule(name="Bad3", formula="Bad3", molar_mass_gmol=10.0,
                 geometry=Geometry.NONLINEAR, n_atoms=3,
                 moments_of_inertia=(1.0, 2.0, 3.0),
                 vibrational_modes=(VibrationalMode(100.0, 1),))  # 1 != 3


def test_vibrational_mode_validation():
    with pytest.raises(ValueError):
        VibrationalMode(100.0, 0)
    with pytest.raises(ValueError):
        VibrationalMode(-1.0, 1)


def test_electronic_level_validation():
    with pytest.raises(ValueError):
        ElectronicLevel(-1.0, 1)
    with pytest.raises(ValueError):
        ElectronicLevel(0.0, 0)


# -- State -------------------------------------------------------------------

def test_state_resolve_TP():
    s = State(T=300.0, P=1e5, n=1.0)
    rs = s.resolve(0.028)
    assert math.isclose(rs.T, 300.0)
    assert math.isclose(rs.P, 1e5)
    assert math.isclose(rs.V, 8.314462618 * 300.0 / 1e5)
    assert math.isclose(rs.m, 0.028)


def test_state_resolve_TV():
    s = State(T=300.0, V=0.024, n=1.0)
    rs = s.resolve(0.028)
    assert math.isclose(rs.P, 8.314462618 * 300.0 / 0.024)


def test_state_default_n_is_one_mole():
    s = State(T=300.0, P=1e5)
    rs = s.resolve(0.028)
    assert math.isclose(rs.n, 1.0)


def test_state_from_mass():
    s = State(T=300.0, P=1e5, m=0.028)
    rs = s.resolve(0.028)
    assert math.isclose(rs.n, 1.0)
    assert math.isclose(rs.m, 0.028)


def test_state_inconsistent_m_n():
    s = State(T=300.0, P=1e5, n=2.0, m=0.001)  # m != n*M for M=0.028
    with pytest.raises(ValueError):
        s.resolve(0.028)


def test_state_default_pressure_when_none():
    s = State(T=300.0)
    rs = s.resolve(0.028)
    assert math.isclose(rs.P, 101325.0)


def test_state_validation_negative_T():
    with pytest.raises(ValueError):
        State(T=-1.0)


def test_state_is_intensive():
    assert State(T=300.0).is_intensive
    assert not State(T=300.0, n=2.0).is_intensive


# -- Contribution ------------------------------------------------------------

def test_contribution_addition():
    from statthermopy import Contribution
    a = Contribution(name="a", ln_q=1.0, d_ln_q_dT=0.1, U_m=10.0, S_m=1.0, A_m=-5.0, Cv_m=0.5)
    b = Contribution(name="b", ln_q=2.0, d_ln_q_dT=0.2, U_m=20.0, S_m=2.0, A_m=-10.0, Cv_m=1.0)
    s = a + b
    assert math.isclose(s.ln_q, 3.0)
    assert math.isclose(s.U_m, 30.0)
    assert math.isclose(s.Cv_m, 1.5)