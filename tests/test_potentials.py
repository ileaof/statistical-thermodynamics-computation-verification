"""Tests for intermolecular potentials."""

import numpy as np

from statistical_thermodynamics import potentials as pot


def test_lennard_jones_zero_at_sigma():
    assert np.isclose(pot.lennard_jones(1.0), 0.0, atol=1e-12)


def test_lennard_jones_minimum_location_and_depth():
    r_min, V_min = pot.lennard_jones_minimum()
    assert np.isclose(r_min, 2.0 ** (1 / 6), rtol=1e-12)
    assert np.isclose(pot.lennard_jones(r_min), -1.0, rtol=1e-10)
    assert V_min == -1.0


def test_lennard_jones_force_vanishes_at_minimum():
    r_min, _ = pot.lennard_jones_minimum()
    assert np.isclose(pot.lennard_jones_force(r_min), 0.0, atol=1e-10)


def test_force_is_negative_gradient():
    r = 1.4
    dr = 1e-6
    numeric = -(pot.lennard_jones(r + dr) - pot.lennard_jones(r - dr)) / (2 * dr)
    assert np.isclose(pot.lennard_jones_force(r), numeric, rtol=1e-4)


def test_mayer_function_limits():
    # Deep overlap (r << sigma): huge repulsion -> f -> -1.
    assert np.isclose(pot.mayer_f(0.5, T=1.0), -1.0, atol=1e-6)
    # Large separation: f -> 0.
    assert np.isclose(pot.mayer_f(5.0, T=1.0), 0.0, atol=1e-3)
