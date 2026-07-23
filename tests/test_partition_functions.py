"""Tests for partition functions and their thermodynamic limits."""

import numpy as np

from statistical_thermodynamics import partition_functions as pf


def test_two_level_high_T_energy_saturates():
    # As T -> infinity the mean energy approaches eps/2.
    assert np.isclose(pf.two_level_energy(1e6, eps=1.0), 0.5, atol=1e-3)


def test_two_level_heat_capacity_matches_fluctuation_form():
    T = 0.5
    # Closed-form C should equal the analytic fluctuation expression.
    x = 1.0 / T
    p1 = np.exp(-x) / pf.two_level_partition(T)
    var = 1.0 ** 2 * p1 * (1 - p1)
    C_fluct = var / T ** 2
    assert np.isclose(pf.two_level_heat_capacity(T), C_fluct, rtol=1e-10)


def test_harmonic_closed_vs_sum():
    T = 1.3
    Z_closed = pf.harmonic_partition(T)
    Z_sum, U_sum, _ = pf.harmonic_partition_sum(T, n_max=400)
    assert np.isclose(Z_closed, Z_sum, rtol=1e-8)
    assert np.isclose(pf.harmonic_energy(T), U_sum, rtol=1e-6)


def test_harmonic_heat_capacity_dulong_petit_limit():
    # High-T Einstein mode -> k_B (reduced units).
    assert np.isclose(pf.harmonic_heat_capacity(1e4, theta=1.0), 1.0, atol=1e-3)


def test_rotational_sum_matches_high_T_expansion():
    # At T = 20 * theta the Euler-Maclaurin series is essentially exact.
    theta, sigma = 5.0, 1
    T = 20 * theta
    z_sum = pf.rotational_partition_sum(T, theta, sigma)[0]
    z_em = pf.rotational_partition_high_T(T, theta, sigma)
    assert np.isclose(z_sum, z_em, rtol=1e-5)


def test_translational_sum_approaches_continuum():
    alpha = 1e-4
    zs = pf.translational_z1_sum(alpha)
    zc = pf.translational_z1_continuum(alpha)
    # Continuum overestimates the sum by ~1/2; relative error ~ sqrt(alpha/pi).
    assert abs(zc - zs) < 1.0
    assert np.isclose((zc - zs) / zc, np.sqrt(alpha / np.pi), rtol=0.05)
