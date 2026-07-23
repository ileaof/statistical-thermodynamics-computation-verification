"""Tests for quantum statistics and blackbody radiation."""

import numpy as np

from statistical_thermodynamics import constants as C
from statistical_thermodynamics import quantum_statistics as qs


def test_occupation_ordering():
    # For x = (eps-mu)/kT > 0:  n_FD <= n_MB <= n_BE.
    eps, mu, T = 2.0, 0.0, 1.0
    fd = qs.fermi_dirac(eps, mu, T)
    mb = qs.maxwell_boltzmann(eps, mu, T)
    be = qs.bose_einstein(eps, mu, T)
    assert fd <= mb <= be


def test_fermi_dirac_half_at_mu():
    assert np.isclose(qs.fermi_dirac(1.0, 1.0, 0.3), 0.5, rtol=1e-12)


def test_quantum_to_classical_limit():
    # For large x, both quantum statistics approach Maxwell-Boltzmann.
    eps, mu, T = 10.0, 0.0, 1.0
    mb = qs.maxwell_boltzmann(eps, mu, T)
    assert np.isclose(qs.bose_einstein(eps, mu, T), mb, rtol=1e-3)
    assert np.isclose(qs.fermi_dirac(eps, mu, T), mb, rtol=1e-3)


def test_bose_function_reproduces_zeta():
    assert np.isclose(qs.bose_function(1.5, 1.0 - 1e-14), C.ZETA_3_2, rtol=1e-3)
    assert np.isclose(qs.bose_function(2.5, 1.0 - 1e-14), C.ZETA_5_2, rtol=1e-3)


def test_planck_integral_gives_stefan_boltzmann():
    # int u(omega) domega = a T^4.
    T = 300.0
    x = np.linspace(1e-4, 60, 5000)
    omega = x * C.k_B * T / C.hbar
    u_num = np.trapz(qs.planck_u_omega(omega, T), omega)
    assert np.isclose(u_num, C.a_rad * T ** 4, rtol=1e-3)


def test_stefan_boltzmann_flux_T4():
    assert np.isclose(qs.stefan_boltzmann_flux(200.0) / qs.stefan_boltzmann_flux(100.0),
                      16.0, rtol=1e-12)
