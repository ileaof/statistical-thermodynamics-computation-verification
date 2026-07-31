"""Unit conversions for spectroscopic and thermodynamic quantities.

Conversions are pure multiplicative factors derived from the CODATA constants in
:mod:`statthermopy.constants`. They let the molecular database store human-friendly
quantities (wavenumbers in cm^-1, molar masses in g/mol, moments of inertia in amu*Angstrom^2)
while the engine works in SI units internally.
"""

from __future__ import annotations

from .constants import N_A, amu, c, h, k_B

# --- Wavenumber / energy / temperature triangle ------------------------------

#: cm^-1 -> J.  E = h * c * wavenumber, with wavenumber converted cm^-1 -> m^-1.
CM1_TO_J: float = h * c * 100.0
#: cm^-1 -> K.  theta = h*c*100/k_B.
CM1_TO_K: float = h * c * 100.0 / k_B
#: cm^-1 -> kJ/mol.
CM1_TO_KJ_MOL: float = h * c * 100.0 * N_A / 1000.0

# --- Mass / molar mass -------------------------------------------------------

#: g/mol -> kg/mol.
G_MOL_TO_KG_MOL: float = 1.0e-3
#: kg/mol -> g/mol.
KG_MOL_TO_G_MOL: float = 1.0e3

# --- Moments of inertia ------------------------------------------------------

#: amu * Angstrom^2 -> kg m^2.
AMU_A2_TO_KG_M2: float = amu * 1.0e-20

# --- Pressure ----------------------------------------------------------------

#: bar -> Pa.
BAR_TO_PA: float = 1.0e5
#: atm -> Pa.
ATM_TO_PA: float = 101325.0

# --- Energy per mole ---------------------------------------------------------

#: J/mol -> kJ/mol.
J_MOL_TO_KJ_MOL: float = 1.0e-3
#: J/mol -> cal/mol (thermochemical).
J_MOL_TO_CAL_MOL: float = 1.0 / 4.184
#: kJ/mol -> kcal/mol.
KJ_MOL_TO_KCAL_MOL: float = 1.0 / 4.184


def wavenumber_to_kelvin(wavenumber_cm1: float) -> float:
    """Convert a wavenumber (cm^-1) to an equivalent temperature (K), ``theta = h c ṽ / k``."""
    return wavenumber_cm1 * CM1_TO_K


def wavenumber_to_joule(wavenumber_cm1: float) -> float:
    """Convert a wavenumber (cm^-1) to energy per molecule (J)."""
    return wavenumber_cm1 * CM1_TO_J


def molar_mass_gmol_to_kgmol(m_gmol: float) -> float:
    """Convert a molar mass from g/mol to kg/mol."""
    return m_gmol * G_MOL_TO_KG_MOL


def inertia_amuA2_to_kgm2(i_amu_a2: float) -> float:
    """Convert a moment of inertia from amu*Å^2 to kg*m^2."""
    return i_amu_a2 * AMU_A2_TO_KG_M2


__all__ = [
    "CM1_TO_J", "CM1_TO_K", "CM1_TO_KJ_MOL",
    "G_MOL_TO_KG_MOL", "KG_MOL_TO_G_MOL",
    "AMU_A2_TO_KG_M2",
    "BAR_TO_PA", "ATM_TO_PA",
    "J_MOL_TO_KJ_MOL", "J_MOL_TO_CAL_MOL", "KJ_MOL_TO_KCAL_MOL",
    "wavenumber_to_kelvin", "wavenumber_to_joule",
    "molar_mass_gmol_to_kgmol", "inertia_amuA2_to_kgm2",
]