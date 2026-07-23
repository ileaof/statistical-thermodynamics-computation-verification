"""Tests for chemical equilibrium and imperfect-gas equations of state."""

import numpy as np

from statistical_thermodynamics import equilibrium as eq


def test_second_virial_negative_at_low_T():
    # Below the Boyle temperature attraction dominates, so B* < 0.
    assert eq.second_virial_lj(1.0) < 0.0


def test_boyle_temperature_near_accepted_value():
    T_B = eq.boyle_temperature_lj()
    assert np.isclose(T_B, 3.418, atol=0.02)
    assert np.isclose(eq.second_virial_lj(T_B), 0.0, atol=1e-6)


def test_carnahan_starling_ideal_limit():
    # As eta -> 0 the compressibility factor approaches 1.
    assert np.isclose(eq.carnahan_starling(0.0), 1.0, rtol=1e-12)


def test_carnahan_starling_low_density_virial():
    # Leading expansion: Z ~ 1 + 4 eta + ...
    eta = 1e-3
    assert np.isclose((eq.carnahan_starling(eta) - 1) / eta, 4.0, rtol=1e-2)


def test_degree_of_dissociation_bounds():
    # Large Kp -> alpha -> 1; small Kp -> alpha -> 0.
    assert np.isclose(eq.degree_of_dissociation(1e12, 1e5), 1.0, atol=1e-3)
    assert eq.degree_of_dissociation(1e-6, 1e5) < 1e-2
    assert 0.0 < eq.degree_of_dissociation(1.0, 1e5) < 1.0


def test_hard_sphere_virial_constants():
    assert np.isclose(eq.B2_HARD_SPHERE, 2 / 3 * np.pi, rtol=1e-12)
    assert np.isclose(eq.B3_HARD_SPHERE, 5 / 18 * np.pi ** 2, rtol=1e-12)
