"""Gas-phase transport property correlations.

ThermoPack is a *thermodynamic* library and does **not** provide transport
properties (viscosity, thermal conductivity). ThermoLab fills this gap with
standard engineering correlations valid for the **dilute / gas phase**:

* Viscosity:        Sutherland's law (per component) + Wilke mixture rule.
* Thermal conductivity: Eucken relation (uses cp and viscosity) + Wassiljeva
  mixture rule.
* Thermal diffusivity: ``alpha = k / (rho * cp)``.
* Prandtl number:    ``Pr  = mu * cp / k``.

These are estimates accurate to typically 5-15% for low-pressure gases; they are
**not** valid for liquids, dense/supercritical states, or two-phase mixtures.
States flagged as liquid/two-phase by the backend return transport properties
computed from the same correlations with a warning attached to the ``State``
(see :mod:`thermolab.state`).
"""

from __future__ import annotations

import warnings

import numpy as np

from ._fluid_db import SUTHERLAND, get_sutherland
from .units import R_GAS

# Chung's reduced dipole moment & collision factor are folded into the
# fallback estimate below. mu_ref scaling for the fallback (Pa.s) at T_ref=300K.
_CHUNG_REF_T: float = 300.0


def _sutherland_viscosity(T: float, coeffs) -> float:
    """Sutherland's law: mu = mu0 * (T0+S)/(T+S) * (T/T0)^1.5."""
    T = float(T)
    return coeffs.mu0 * (coeffs.T0 + coeffs.S) / (T + coeffs.S) * (T / coeffs.T0) ** 1.5


def _fallback_viscosity(T: float, molar_mass_kg_per_mol: float, Tc: float) -> float:
    """Chung-style low-pressure gas viscosity estimate [Pa.s].

    Uses the corresponding-states correlation
    ``mu ~ 4.0785e-7 * sqrt(M*T) / (sigma^2 * Omega^(2,2))`` with a simplified
    sigma from critical properties. Good to within ~20% when no Sutherland data
    are available.
    """
    M = molar_mass_kg_per_mol * 1000.0  # g/mol
    # characteristic length [Angstrom] from Tc (rough Chung estimate)
    sigma = 0.809 * (Tc * 1e6 / 101325.0) ** (1.0 / 3.0) if Tc else 3.7
    T_star = T / 1.2593 / max(Tc, 1.0) * 10.0 if Tc else 1.0
    T_star = max(T_star, 0.3)
    # Neufeld collision integral (2,2)
    A, B, C, D, E, F = 1.16145, 0.14874, 0.52487, 0.77320, 2.16178, 2.43787
    omega = A * T_star ** -B + C * np.exp(-D * T_star) + E * np.exp(-F * T_star)
    mu = 2.6693e-6 * np.sqrt(M * T) / (sigma ** 2 * omega)  # Pa.s
    return mu


def pure_viscosity(T: float, molar_mass_kg_per_mol: float, name: str, Tc: float | None = None) -> float:
    """Gas-phase dynamic viscosity [Pa.s] of a pure component."""
    coeffs = get_sutherland(name)
    if coeffs is not None:
        return _sutherland_viscosity(T, coeffs)
    if Tc is None:
        # last-resort: ideal-gas scaling around a 1.7e-5 anchor at 300K
        mu_ref = 1.7e-5
        return mu_ref * (T / _CHUNG_REF_T) ** 0.7
    return _fallback_viscosity(T, molar_mass_kg_per_mol, Tc)


def mixture_viscosity(
    T: float,
    z: np.ndarray,
    molar_masses_kg_per_mol: np.ndarray,
    component_names: list[str],
    critical_temps: list[float | None],
) -> float:
    """Gas mixture viscosity [Pa.s] via Wilke's combining rule."""
    z = np.asarray(z, dtype=float)
    mus = np.array(
        [
            pure_viscosity(T, molar_masses_kg_per_mol[i], component_names[i], critical_temps[i])
            for i in range(len(z))
        ]
    )
    denom = np.zeros_like(z)
    for i in range(len(z)):
        s = 0.0
        for j in range(len(z)):
            phi = (1.0 + np.sqrt(mus[i] / mus[j]) * (molar_masses_kg_per_mol[j] / molar_masses_kg_per_mol[i]) ** 0.25) ** 2 / np.sqrt(
                8.0 * (1.0 + molar_masses_kg_per_mol[i] / molar_masses_kg_per_mol[j])
            )
            s += z[j] * phi
        denom[i] = s
    return float(np.sum(z * mus / denom))


def _eucken_conductivity(mu: float, cp_molar: float, molar_mass_kg_per_mol: float) -> float:
    """Eucken gas thermal conductivity [W/(m.K)].

    ``k = mu * (cp_molar + 1.25 * R) / M``  with cp_molar in J/(mol.K),
    M in kg/mol, mu in Pa.s. Reproduces monatomic Pr = 2/3 exactly.
    """
    return mu * (cp_molar + 1.25 * R_GAS) / molar_mass_kg_per_mol


def pure_conductivity(
    T: float, cp_molar: float, mu: float, molar_mass_kg_per_mol: float
) -> float:
    """Gas-phase thermal conductivity [W/(m.K)] of a pure component (Eucken)."""
    return _eucken_conductivity(mu, cp_molar, molar_mass_kg_per_mol)


def mixture_conductivity(
    T: float,
    z: np.ndarray,
    molar_masses_kg_per_mol: np.ndarray,
    cp_molars: np.ndarray,
    mus: np.ndarray,
) -> float:
    """Gas mixture thermal conductivity [W/(m.K)] via Wassiljeva's rule.

    Approximated with the Mason-Saxena combination formula.
    """
    z = np.asarray(z, dtype=float)
    ks = np.array(
        [_eucken_conductivity(mus[i], cp_molars[i], molar_masses_kg_per_mol[i]) for i in range(len(z))]
    )
    denom = np.zeros_like(z)
    for i in range(len(z)):
        s = 0.0
        for j in range(len(z)):
            ratio = (molar_masses_kg_per_mol[j] / molar_masses_kg_per_mol[i]) ** 0.25
            phi = (1.0 + (ks[i] / ks[j]) ** 0.5 * ratio) ** 2 / np.sqrt(
                8.0 * (1.0 + molar_masses_kg_per_mol[i] / molar_masses_kg_per_mol[j])
            )
            s += z[j] * phi
        denom[i] = s
    return float(np.sum(z * ks / denom))


def warn_transport_approx(phase: str) -> None:
    """Emit a one-time warning when transport is approximated off gas-phase."""
    warnings.warn(
        f"Transport properties are gas-phase correlations and are not accurate "
        f"for the {phase!r} phase.",
        stacklevel=2,
    )