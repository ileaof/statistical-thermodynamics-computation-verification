"""Physical constants and unit helpers used throughout ThermoLab.

All ThermoLab properties are reported in **SI mass-based** units:

* Temperature            K
* Pressure               Pa
* Density                kg/m^3
* Specific volume        m^3/kg
* Energy (u, h, g, a)    J/kg
* Entropy                J/(kg.K)
* Heat capacities        J/(kg.K)
* Speed of sound         m/s
* Viscosity              Pa.s
* Thermal conductivity   W/(m.K)
* Thermal diffusivity    m^2/s
* Joule-Thomson          K/Pa
* Thermal expansion      1/K
* Compressibility        1/Pa

ThermoPack returns *molar* quantities; the helpers below convert to mass-based
units using the mixture molar mass.
"""

from __future__ import annotations

import numpy as np

# Universal gas constant (J/(mol.K))
R_GAS: float = 8.31446261815324

# Standard gravity (m/s^2)
G_STD: float = 9.80665

# Absolute zero in Celsius (K)
T_CELSIUS_ZERO: float = 273.15

# Common conversion helpers
PA_TO_BAR: float = 1e-5
PA_TO_ATM: float = 1.0 / 101325.0
KPA_TO_PA: float = 1e3
MPA_TO_PA: float = 1e6
BAR_TO_PA: float = 1e5
KJ_TO_J: float = 1e3
KJ_PER_KG_TO_J_PER_KG: float = 1e3


def specific_gas_constant(molar_mass_kg_per_mol: float) -> float:
    """Return the specific gas constant R_s = R / M (J/(kg.K))."""
    return R_GAS / molar_mass_kg_per_mol


def molar_to_mass(molar_value: float, molar_mass_kg_per_mol: float) -> float:
    """Convert a molar extensive quantity (per mol) to mass-based (per kg).

    ``molar_value`` is in ``X/mol`` and the result is in ``X/kg`` where X is any
    energy-like or entropy-like unit. This works because dividing by kg/mol
    multiplies by mol/kg.
    """
    return float(molar_value) / molar_mass_kg_per_mol


def molar_to_mass_array(molar_values: np.ndarray, molar_mass_kg_per_mol: float) -> np.ndarray:
    """Vectorized :func:`molar_to_mass`."""
    return np.asarray(molar_values, dtype=float) / molar_mass_kg_per_mol


def mole_to_mass_fractions(z: np.ndarray, molar_masses_kg_per_mol: np.ndarray) -> np.ndarray:
    """Convert mole fractions ``z`` (sum to 1) to mass fractions."""
    z = np.asarray(z, dtype=float)
    masses = z * molar_masses_kg_per_mol
    total = masses.sum()
    if total <= 0.0:
        raise ValueError("Total mass is non-positive; check mole fractions.")
    return masses / total


def mixture_molar_mass(z: np.ndarray, molar_masses_kg_per_mol: np.ndarray) -> float:
    """Mixture molar mass (kg/mol): M_mix = sum(z_i * M_i)."""
    z = np.asarray(z, dtype=float)
    return float(np.dot(z, molar_masses_kg_per_mol))


def k_to_c(t_kelvin: float) -> float:
    """Kelvin to Celsius."""
    return t_kelvin - T_CELSIUS_ZERO


def c_to_k(t_celsius: float) -> float:
    """Celsius to Kelvin."""
    return t_celsius + T_CELSIUS_ZERO