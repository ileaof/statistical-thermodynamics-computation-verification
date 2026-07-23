"""Tests for ideal-gas thermodynamics."""

import numpy as np

from statistical_thermodynamics import constants as C
from statistical_thermodynamics import thermodynamics as th


def test_thermal_wavelength_scaling():
    m = 40 * C.u
    lam300 = th.thermal_wavelength(m, 300.0)
    lam1200 = th.thermal_wavelength(m, 1200.0)
    # Lambda ~ T^{-1/2}: quadrupling T halves Lambda.
    assert np.isclose(lam300 / lam1200, 2.0, rtol=1e-10)


def test_sackur_tetrode_argon_matches_experiment():
    # Standard molar entropy of argon at 298.15 K, 1 bar is ~154.8 J/mol/K.
    S = th.sackur_tetrode_molar(39.948 * C.u, 298.15, 1.0e5)
    assert abs(S - 154.85) < 1.0


def test_sackur_tetrode_mass_scaling():
    T, p = 298.15, 1.0e5
    dS = (th.sackur_tetrode_molar(39.948 * C.u, T, p)
          - th.sackur_tetrode_molar(20.1797 * C.u, T, p))
    predicted = 1.5 * C.R * np.log(39.948 / 20.1797)
    assert np.isclose(dS, predicted, rtol=1e-10)


def test_chemical_potential_matches_helmholtz_derivative():
    m, T = 40 * C.u, 300.0
    N, V = 1e23, 1e23 / 2.5e25
    dN = N * 1e-6
    mu_fd = (th.helmholtz_ideal_gas(N + dN, V, T, m)
             - th.helmholtz_ideal_gas(N - dN, V, T, m)) / (2 * dN)
    mu_closed = th.chemical_potential_ideal_gas(N, V, T, m)
    assert np.isclose(mu_fd, mu_closed, rtol=1e-6)


def test_heat_capacity_from_fluctuation_two_level():
    # For a two-level system with eps=1, reduced units.
    T = 0.5
    x = 1.0 / T
    p1 = np.exp(-x) / (1 + np.exp(-x))
    U = 1.0 * p1
    E2 = 1.0 * p1  # E^2 = eps^2 only in the excited level
    C_val = th.heat_capacity_from_fluctuation(U, E2, T, k_B_=1.0)
    assert np.isclose(C_val, (E2 - U ** 2) / T ** 2)
