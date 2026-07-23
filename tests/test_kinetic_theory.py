"""Tests for kinetic theory."""

import numpy as np
from scipy.integrate import quad

from statistical_thermodynamics import constants as C
from statistical_thermodynamics import kinetic_theory as kt


def test_speed_distribution_normalised():
    m, T = 28.0134 * C.u, 300.0
    norm, _ = quad(kt.maxwell_speed_pdf, 0, np.inf, args=(m, T))
    assert np.isclose(norm, 1.0, rtol=1e-6)


def test_characteristic_speed_ratios():
    v_p, v_avg, v_rms = kt.characteristic_speeds(28.0134 * C.u, 300.0)
    assert np.isclose(v_avg / v_p, np.sqrt(4 / np.pi), rtol=1e-10)
    assert np.isclose(v_rms / v_p, np.sqrt(1.5), rtol=1e-10)
    assert v_p < v_avg < v_rms


def test_mean_speed_matches_moment_integral():
    m, T = 28.0134 * C.u, 300.0
    v1, _ = quad(lambda v: v * kt.maxwell_speed_pdf(v, m, T), 0, np.inf)
    assert np.isclose(v1, kt.mean_speed(m, T), rtol=1e-6)


def test_mean_free_path_inverse_density():
    d = 3.7e-10
    n1 = kt.number_density(1.0e5, 300.0)
    n2 = kt.number_density(2.0e5, 300.0)
    lam1 = kt.mean_free_path(n1, d)
    lam2 = kt.mean_free_path(n2, d)
    # lambda ~ 1/n ~ 1/P: doubling pressure halves the mean free path.
    assert np.isclose(lam1 / lam2, 2.0, rtol=1e-10)


def test_air_mean_free_path_order_of_magnitude():
    # Air at 1 atm, 300 K: mean free path is ~ 6-7 x 10^-8 m.
    n = kt.number_density(1.01325e5, 300.0)
    lam = kt.mean_free_path(n, 3.7e-10)
    assert 3e-8 < lam < 1e-7
