"""Fluid database: aliases, name normalization, and transport correlation data.

ThermoPack is a *thermodynamic* library only; it does **not** supply transport
properties (viscosity, thermal conductivity). ThermoLab computes gas-phase
transport via engineering correlations (:mod:`thermolab.transport`). The
Sutherland viscosity constants for the most common gases are stored here; any
fluid without an explicit entry falls back to a Chung-style estimate that uses
the critical temperature obtained from the backend at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SutherlandCoeffs:
    """Sutherland viscosity parameters for a pure gas.

    ``mu = mu0 * (T0 + S) / (T + S) * (T / T0) ** 1.5``  [Pa.s]
    """

    mu0: float  # reference viscosity at T0  [Pa.s]
    T0: float   # reference temperature       [K]
    S: float    # Sutherland temperature       [K]


# Reference T = 273.15 K for all entries (consistent basis).
_T0 = 273.15

SUTHERLAND: dict[str, SutherlandCoeffs] = {
    # common inorganic / diatomic gases
    "N2":  SutherlandCoeffs(1.663e-5, _T0, 107.0),
    "O2":  SutherlandCoeffs(1.919e-5, _T0, 139.0),
    "AR":  SutherlandCoeffs(2.117e-5, _T0, 144.0),
    "H2":  SutherlandCoeffs(0.840e-5, _T0, 72.0),
    "HE":  SutherlandCoeffs(1.870e-5, _T0, 78.0),
    "CO":  SutherlandCoeffs(1.660e-5, _T0, 102.0),
    "CO2": SutherlandCoeffs(1.370e-5, _T0, 240.0),
    "H2O": SutherlandCoeffs(1.120e-5, _T0, 600.0),  # steam; polar, approximate
    "NH3": SutherlandCoeffs(0.920e-5, _T0, 370.0),
    "SO2": SutherlandCoeffs(1.170e-5, _T0, 416.0),
    "H2S": SutherlandCoeffs(1.170e-5, _T0, 331.0),
    "NO":  SutherlandCoeffs(1.780e-5, _T0, 133.0),
    "N2O": SutherlandCoeffs(1.350e-5, _T0, 274.0),
    "KR":  SutherlandCoeffs(2.320e-5, _T0, 188.0),
    "XE":  SutherlandCoeffs(2.120e-5, _T0, 252.0),
    "NE":  SutherlandCoeffs(2.97e-5,  _T0, 56.0),
    # representative refrigerant vapors (approximate; Chung fallback used otherwise)
    "R134A":  SutherlandCoeffs(1.02e-5, _T0, 320.0),
    "R32":    SutherlandCoeffs(1.08e-5, _T0, 280.0),
    "R143A":  SutherlandCoeffs(0.95e-5, _T0, 300.0),
    "R1234YF": SutherlandCoeffs(1.05e-5, _T0, 300.0),
    "R1234ZE": SutherlandCoeffs(1.00e-5, _T0, 310.0),
    "R12":    SutherlandCoeffs(1.10e-5, _T0, 280.0),
    "BENZENE": SutherlandCoeffs(0.70e-5, _T0, 360.0),
    "DME":    SutherlandCoeffs(0.85e-5, _T0, 290.0),
}

# Named pseudo-fluids that map to a composition. The fractions are *mole*
# fractions and must sum to 1 (normalization is applied automatically).
#
# "Air" is the most important one: ThermoPack has no "Air" component in this
# build, so dry air is represented as its dominant constituents, which are all
# GERG2008 components and yield accurate air properties.
FLUID_ALIASES: dict[str, dict] = {
    "AIR": {
        "components": ["N2", "O2", "Ar", "CO2"],
        "fractions": [0.7884, 0.2095, 0.0093, 0.0004],
        "description": "Dry air (US standard atmosphere composition)",
    },
    "DRYAIR": {
        "components": ["N2", "O2", "Ar", "CO2"],
        "fractions": [0.7884, 0.2095, 0.0093, 0.0004],
        "description": "Dry air (alias of Air)",
    },
}


def normalize_name(name: str) -> str:
    """Normalize a fluid/component name for case-insensitive lookup."""
    return str(name).strip().upper().replace("_", "")


def is_alias(name: str) -> bool:
    return normalize_name(name) in FLUID_ALIASES


def resolve_alias(name: str) -> tuple[list[str], list[float]]:
    """Return ``(components, mole_fractions)`` for a named alias.

    Raises ``KeyError`` if the name is not an alias.
    """
    entry = FLUID_ALIASES[normalize_name(name)]
    return list(entry["components"]), list(entry["fractions"])


def get_sutherland(name: str) -> SutherlandCoeffs | None:
    """Return Sutherland coefficients for a normalized fluid name, or None."""
    return SUTHERLAND.get(normalize_name(name))


# Curated list of fluids known to be supported by the installed ThermoPack
# build (used for documentation / ``list_fluids``). Runtime availability is
# always verified by attempting construction in the backend.
KNOWN_SUPPORTED: tuple[str, ...] = (
    "H2O", "CO2", "N2", "O2", "Ar", "H2", "He", "NH3", "CH4",
    "R134a", "R32", "R143a", "R1234yf", "R1234ze", "R12", "R14", "R23",
    "R116", "R125", "Benzene", "CO", "N2O", "SO2", "H2S", "Kr", "Xe", "Ne",
    "DME",
)