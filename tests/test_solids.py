"""Tests for the Einstein and Debye heat-capacity models."""

import numpy as np

from statistical_thermodynamics import constants as C
from statistical_thermodynamics import solids as sol


def test_einstein_dulong_petit_limit():
    assert np.isclose(sol.einstein_heat_capacity(1e5, 300.0, molar=False), 1.0,
                      atol=1e-3)


def test_einstein_freezes_out_at_low_T():
    # Deep in the quantum regime the heat capacity is exponentially small.
    assert sol.einstein_heat_capacity(30.0, 1320.0, molar=False) < 1e-3


def test_debye_dulong_petit_limit():
    # Deep in the classical regime (T = 50 theta_D) C_V -> 3R.
    C_hi = sol.debye_heat_capacity(50 * 343.0, 343.0)[0]
    assert np.isclose(C_hi, sol.dulong_petit(), rtol=1e-3)


def test_debye_T3_law_coefficient():
    # Low-T law is for C_V / R; molar=False returns C_V / 3R, so multiply by 3.
    theta_D = 343.0
    T = 5.0
    C_over_R = 3.0 * sol.debye_heat_capacity(T, theta_D, molar=False)[0]
    coeff = C_over_R / (T / theta_D) ** 3
    assert np.isclose(coeff, 12 * np.pi ** 4 / 5, rtol=1e-2)


def test_debye_below_einstein_at_low_T():
    # With the same characteristic temperature, Debye exceeds Einstein at low T
    # (acoustic modes stay active); check Debye C_V is the larger there.
    theta = 300.0
    T = 30.0
    cd = sol.debye_heat_capacity(T, theta, molar=False)[0]
    ce = sol.einstein_heat_capacity(T, theta, molar=False)
    assert cd > ce


def test_dulong_petit_value():
    assert np.isclose(sol.dulong_petit(), 3 * C.R)
