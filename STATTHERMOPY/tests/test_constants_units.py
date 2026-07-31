"""Tests for physical constants and unit conversions."""

from __future__ import annotations

import math

import pytest

from statthermopy import constants as c
from statthermopy import units as u


def test_gas_constant_is_navogadro_times_boltzmann():
    assert math.isclose(c.R, c.N_A * c.k_B, rel_tol=1e-12)


def test_hc_matches_h_times_c():
    assert math.isclose(c.hc, c.h * c.c, rel_tol=1e-12)


def test_wavenumber_to_kelvin_round_trip():
    # 1 cm-1 ~ 1.4388 K
    assert math.isclose(u.CM1_TO_K, 1.438777, rel_tol=1e-4)
    assert math.isclose(u.wavenumber_to_kelvin(1000.0), 1000.0 * u.CM1_TO_K)


def test_wavenumber_to_joule():
    # E = h c wavenumber; 100 cm-1
    assert math.isclose(u.wavenumber_to_joule(100.0), c.h * c.c * 100.0 * 100.0)


def test_molar_mass_conversion():
    assert math.isclose(u.molar_mass_gmol_to_kgmol(28.014), 0.028014)


def test_inertia_conversion():
    # 1 amu*A^2 in kg m^2
    assert math.isclose(u.inertia_amuA2_to_kgm2(1.0), c.amu * 1e-20)


def test_pressure_conversions():
    assert math.isclose(u.BAR_TO_PA, 1e5)
    assert math.isclose(u.ATM_TO_PA, 101325.0)


def test_energy_conversions():
    assert math.isclose(u.J_MOL_TO_KJ_MOL * 1000.0, 1.0)
    assert math.isclose(u.J_MOL_TO_CAL_MOL * 4184.0, 1000.0, rel_tol=1e-3)
    assert math.isclose(u.KJ_MOL_TO_KCAL_MOL * 4.184, 1.0, rel_tol=1e-3)