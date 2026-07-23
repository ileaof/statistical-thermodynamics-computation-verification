"""Tests for the physical constants."""

import numpy as np

from statistical_thermodynamics import constants as C


def test_gas_constant_consistency():
    assert np.isclose(C.N_A * C.k_B, C.R, rtol=1e-12)


def test_hbar_definition():
    assert np.isclose(C.hbar, C.h / (2 * np.pi), rtol=1e-15)


def test_stefan_boltzmann_matches_codata():
    assert np.isclose(C.sigma_SB, C.SIGMA_SB_CODATA, rtol=1e-6)


def test_radiation_constant_relation():
    assert np.isclose(C.a_rad, 4 * C.sigma_SB / C.c, rtol=1e-12)


def test_ev_is_elementary_charge():
    assert C.eV == C.e_charge
