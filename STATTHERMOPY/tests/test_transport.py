"""Tests for the statistical transport properties module.

Validates the Chapman–Enskog / Lennard–Jones transport engine, the ideal-gas thermophysical
coefficients, the derived dimensionless groups, continuity down to T = 0, the binary-diffusion
combining rules and symmetry, the Lennard–Jones schema migration of all 22 species, and the
vectorised ``property_vs_T`` helper against per-point evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from statthermopy import State, get, list_molecules
from statthermopy.constants import R, k_B
from statthermopy.thermodynamics import Thermodynamics
from statthermopy.transport import (
    TRANSPORT_PROPS,
    TRANSPORT_UNITS,
    TransportCalculator,
    binary_diffusion,
    collision_integral,
    combine_epsilon_over_k,
    combine_sigma,
    omega_11,
    omega_22,
    pair_epsilon_over_k,
    pair_sigma_m,
    reduced_mass,
    self_diffusion,
    t_star,
)

# -- collision integrals --------------------------------------------------------


def test_omega_22_at_unit_t_star():
    # Neufeld fit: Ω^(2,2)*(1.0) ≈ 1.600 (standard tabulation value)
    assert omega_22(1.0) == pytest.approx(1.600, abs=0.02)


def test_omega_22_at_high_t_star():
    # Ω^(2,2)*(10) ≈ 0.824 (standard tabulation value)
    assert omega_22(10.0) == pytest.approx(0.824, abs=0.02)


def test_collision_integrals_monotonic_decreasing():
    Ts = np.linspace(0.3, 100.0, 50)
    o11 = [omega_11(t) for t in Ts]
    o22 = [omega_22(t) for t in Ts]
    # both collision integrals fall as T* rises (the LJ potential becomes weaker relative to kT)
    assert all(b <= a + 1e-9 for a, b in zip(o11, o11[1:], strict=False))
    assert all(b <= a + 1e-9 for a, b in zip(o22, o22[1:], strict=False))


def test_collision_integral_dispatcher():
    assert collision_integral(1, 1, 2.0) == pytest.approx(omega_11(2.0))
    assert collision_integral(2, 2, 2.0) == pytest.approx(omega_22(2.0))


def test_low_t_star_is_finite():
    # below the Neufeld lower bound (T*=0.3) the low-T* branch is used; must stay finite & > 0
    assert math.isfinite(omega_11(0.1)) and omega_11(0.1) > 0.0
    assert math.isfinite(omega_22(0.1)) and omega_22(0.1) > 0.0


# -- viscosity vs literature ----------------------------------------------------


@pytest.mark.parametrize("name, T, expected, tol", [
    ("Ar", 300.0, 2.27e-5, 0.05),
    ("N2", 300.0, 1.78e-5, 0.05),
    ("O2", 300.0, 2.08e-5, 0.05),
    ("CO2", 300.0, 1.50e-5, 0.05),
])
def test_viscosity_vs_literature(name, T, expected, tol):
    mol = get(name)
    mu = TransportCalculator(mol, State(T=T, P=101325.0)).viscosity(T)
    assert mu == pytest.approx(expected, rel=tol)


def test_methane_viscosity_reasonable():
    # CH4 is ~6% low with CE/LJ (known) — keep a wider tolerance
    mol = get("CH4")
    mu = TransportCalculator(mol, State(T=300.0, P=101325.0)).viscosity(300.0)
    assert mu == pytest.approx(1.12e-5, rel=0.12)


def test_viscosity_pressure_independent():
    """Dilute-gas viscosity is a function of T only — independent of P."""
    mol = get("N2")
    calc = TransportCalculator(mol, State(T=400.0, P=101325.0))
    mu1 = calc.viscosity(400.0)
    mu2 = TransportCalculator(mol, State(T=400.0, P=1e6)).viscosity(400.0)
    assert mu1 == pytest.approx(mu2)


def test_viscosity_rises_with_temperature():
    mol = get("N2")
    calc = TransportCalculator(mol, State(T=300.0, P=101325.0))
    assert calc.viscosity(500.0) > calc.viscosity(300.0)


# -- thermal conductivity vs literature -----------------------------------------


@pytest.mark.parametrize("name, T, expected, tol", [
    ("Ar", 300.0, 1.77e-2, 0.10),
    ("N2", 300.0, 2.60e-2, 0.10),
])
def test_conductivity_vs_literature(name, T, expected, tol):
    mol = get(name)
    k = TransportCalculator(mol, State(T=T, P=101325.0)).conductivity(T)
    assert k == pytest.approx(expected, rel=tol)


def test_eucken_monatomic_recovery():
    """For a monatomic gas (γ=5/3) the Eucken factor (9γ−5)/4 = 5/2, which equals the
    Chapman–Enskog monatomic multiplier on c_v, so Eucken recovers the exact CE result
    k = (5/2)·c_v·μ = (15/4)(k_B/m)·μ (c_v per mass = (3/2)R/M for a monatomic gas)."""
    mol = get("Ar")
    calc = TransportCalculator(mol, State(T=300.0, P=101325.0))
    mu = calc.viscosity(300.0)
    k = calc.conductivity(300.0, mu=mu)
    th = Thermodynamics(mol, State(T=300.0, P=101325.0)).compute()
    assert th.gamma == pytest.approx(5.0 / 3.0, abs=0.02)
    expected = (15.0 / 4.0) * (k_B / mol.molecular_mass) * mu
    assert k == pytest.approx(expected, rel=0.02)


# -- diffusion ------------------------------------------------------------------


def test_self_diffusion_order_of_magnitude():
    mol = get("N2")
    D = TransportCalculator(mol, State(T=300.0, P=101325.0)).self_diffusion_coeff(300.0, 101325.0)
    # N2 self-diffusion at 300 K, 1 atm is ~2e-5 m²/s
    assert pytest.approx(2.0e-5, rel=0.3) == D


def test_self_diffusion_pressure_inverse():
    """D_self ∝ 1/P for an ideal gas."""
    mol = get("N2")
    D1 = TransportCalculator(mol, State(T=300.0, P=101325.0)).self_diffusion_coeff(300.0, 101325.0)
    D2 = TransportCalculator(mol, State(T=300.0, P=2 * 101325.0)).self_diffusion_coeff(
        300.0, 2 * 101325.0
    )
    assert pytest.approx(0.5 * D1) == D2


def test_binary_diffusion_symmetry():
    """D_ij == D_ji by construction."""
    n2, o2 = get("N2"), get("O2")
    D_ij = binary_diffusion(n2, o2, 300.0, 101325.0)
    D_ji = binary_diffusion(o2, n2, 300.0, 101325.0)
    assert D_ij == pytest.approx(D_ji)
    assert D_ij == pytest.approx(2.06e-5, rel=0.15)


def test_self_diffusion_matches_calculator():
    mol = get("N2")
    via_calc = TransportCalculator(mol, State(T=300.0, P=101325.0)).self_diffusion_coeff(
        300.0, 101325.0
    )
    via_func = self_diffusion(mol, 300.0, 101325.0)
    assert via_func == pytest.approx(via_calc)


# -- combining rules ------------------------------------------------------------


def test_lorentz_berthelot_combining():
    n2, o2 = get("N2"), get("O2")
    lj_n2, lj_o2 = n2.lennard_jones, o2.lennard_jones
    assert combine_sigma(lj_n2, lj_o2) == pytest.approx(
        0.5 * (lj_n2.sigma_angstrom + lj_o2.sigma_angstrom)
    )
    assert combine_epsilon_over_k(lj_n2, lj_o2) == pytest.approx(
        math.sqrt(lj_n2.epsilon_over_k * lj_o2.epsilon_over_k)
    )
    # reduced mass m_ij = m_i m_j / (m_i + m_j)
    m_i, m_j = n2.molecular_mass, o2.molecular_mass
    assert reduced_mass(n2, o2) == pytest.approx(m_i * m_j / (m_i + m_j))


def test_pair_sigma_epsilon_meters():
    n2, o2 = get("N2"), get("O2")
    sigma_m = pair_sigma_m(n2.lennard_jones, o2.lennard_jones)
    eps_k = pair_epsilon_over_k(n2.lennard_jones, o2.lennard_jones)
    assert sigma_m == pytest.approx(combine_sigma(n2.lennard_jones, o2.lennard_jones) * 1e-10)
    assert eps_k > 0.0


# -- full report / derived properties -------------------------------------------


def test_full_report_has_all_properties():
    mol = get("N2")
    res = TransportCalculator(mol, State(T=300.0, P=101325.0)).compute()
    for prop in TRANSPORT_PROPS:
        assert hasattr(res, prop)
        assert math.isfinite(getattr(res, prop))


def test_units_cover_all_props():
    for prop in TRANSPORT_PROPS:
        assert prop in TRANSPORT_UNITS


def test_ideal_gas_derived_properties():
    mol = get("N2")
    res = TransportCalculator(mol, State(T=300.0, P=101325.0)).compute()
    assert res.Z == 1.0
    assert res.beta == pytest.approx(1.0 / 300.0)
    assert res.kappa_T == pytest.approx(1.0 / 101325.0)
    assert res.mu_JT == 0.0
    th = Thermodynamics(mol, State(T=300.0, P=101325.0)).compute()
    R_specific = R / mol.molar_mass
    assert res.a == pytest.approx(math.sqrt(th.gamma * R_specific * 300.0))


def test_density_and_kinematic_viscosity():
    mol = get("N2")
    res = TransportCalculator(mol, State(T=300.0, P=101325.0)).compute()
    rho = 101325.0 * mol.molar_mass / (R * 300.0)
    assert res.rho == pytest.approx(rho)
    assert res.nu == pytest.approx(res.mu / rho)


def test_prandtl_eucken_form():
    """Pr = 4γ/(9γ−5), independent of μ (Eucken) and finite at every T."""
    mol = get("N2")
    res = TransportCalculator(mol, State(T=300.0, P=101325.0)).compute()
    th = Thermodynamics(mol, State(T=300.0, P=101325.0)).compute()
    expected = 4.0 * th.gamma / (9.0 * th.gamma - 5.0)
    assert res.Pr == pytest.approx(expected)
    # also equal to ν/α
    assert res.Pr == pytest.approx(res.nu / res.alpha)


def test_prandtl_value_n2():
    """N2 Pr ≈ 0.71 (literature); the Eucken closed form 4γ/(9γ−5) gives 0.737 for γ=1.4,
    a known ~3 % overprediction of the Eucken correlation — keep a wider tolerance."""
    mol = get("N2")
    res = TransportCalculator(mol, State(T=300.0, P=101325.0)).compute()
    assert res.Pr == pytest.approx(0.71, abs=0.04)


def test_schmidt_lewis_relation():
    mol = get("N2")
    res = TransportCalculator(mol, State(T=300.0, P=101325.0)).compute()
    # Sc = ν/D_self
    assert res.Sc == pytest.approx(res.nu / res.D_self)
    assert res.Le == pytest.approx(res.Sc / res.Pr)


def test_gamma_consistent_with_thermodynamics():
    mol = get("N2")
    res = TransportCalculator(mol, State(T=800.0, P=101325.0)).compute()
    th = Thermodynamics(mol, State(T=800.0, P=101325.0)).compute()
    assert res.gamma == pytest.approx(th.gamma)


# -- continuity at T = 0 --------------------------------------------------------


def test_t_zero_continuity():
    """At T = 0: μ, k, D → 0; Pr/Sc/Le finite; no NaN/inf anywhere."""
    mol = get("N2")
    res = TransportCalculator(mol, State(T=0.0, P=101325.0)).compute()
    assert res.mu == 0.0
    assert res.k == 0.0
    assert res.D_self == 0.0
    assert res.a == 0.0
    assert math.isfinite(res.Pr)
    assert math.isfinite(res.Sc)
    assert math.isfinite(res.Le)
    assert res.Z == 1.0


def test_t_star_zero_is_finite():
    assert t_star(0.0, 71.4) == 0.0
    assert math.isfinite(omega_11(0.0))
    assert math.isfinite(omega_22(0.0))


# -- all 22 species carry LJ params ---------------------------------------------


def test_all_species_have_lennard_jones():
    names = list_molecules()
    assert len(names) == 22
    for name in names:
        mol = get(name)
        assert mol.has_lennard_jones, f"{name} is missing lennard_jones parameters"
        assert mol.lennard_jones.sigma_angstrom > 0
        assert mol.lennard_jones.epsilon_over_k > 0
        assert mol.lennard_jones.sigma_m > 0
        assert mol.lennard_jones.epsilon > 0


def test_no_critical_attribute_remains():
    """CriticalConstants was removed with the phase-diagrams module."""
    mol = get("N2")
    assert not hasattr(mol, "critical")
    assert not hasattr(mol, "has_critical")


def test_all_species_compute_transport():
    for name in list_molecules():
        mol = get(name)
        res = TransportCalculator(mol, State(T=300.0, P=101325.0)).compute()
        for prop in TRANSPORT_PROPS:
            assert math.isfinite(getattr(res, prop)), f"{name}.{prop} not finite"


# -- vectorised helpers ---------------------------------------------------------


def test_property_vs_T_matches_per_point():
    mol = get("N2")
    calc = TransportCalculator(mol, State(T=300.0, P=101325.0))
    Ts = np.linspace(300.0, 1000.0, 12)
    Ts_calc, vals = calc.property_vs_T("mu", Ts, P=101325.0)
    per_point = [
        TransportCalculator(mol, State(T=float(T), P=101325.0)).viscosity(float(T))
        for T in Ts
    ]
    assert list(Ts_calc) == list(Ts)
    assert vals == pytest.approx(per_point)


def test_property_vs_P_pressure_inverse_for_nu():
    """ν ∝ 1/P."""
    mol = get("N2")
    calc = TransportCalculator(mol, State(T=300.0, P=101325.0))
    Ps = np.array([101325.0, 2 * 101325.0, 4 * 101325.0])
    Ps_calc, vals = calc.property_vs_P("nu", Ps, T=300.0)
    assert list(Ps_calc) == list(Ps)
    # halves as P doubles
    assert vals[1] == pytest.approx(0.5 * vals[0])
    assert vals[2] == pytest.approx(0.25 * vals[0])


# -- error paths ----------------------------------------------------------------


def _bare_molecule(name="X"):
    """A minimal Molecule with no Lennard–Jones parameters."""
    from statthermopy.core.molecule import Geometry, Molecule

    return Molecule(
        name=name, formula=name, molar_mass_gmol=40.0,
        geometry=Geometry.MONOATOMIC, n_atoms=1,
    )


def test_calculator_requires_lennard_jones():
    """A molecule without LJ parameters raises."""
    bare = _bare_molecule()
    with pytest.raises(ValueError, match="Lennard–Jones"):
        TransportCalculator(bare, State(T=300.0, P=101325.0))


def test_binary_diffusion_requires_both_lj():
    with pytest.raises(ValueError, match="LJ parameters"):
        binary_diffusion(get("N2"), _bare_molecule(), 300.0, 101325.0)
