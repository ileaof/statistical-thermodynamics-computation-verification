"""Statistical Transport Properties — first-principles transport & thermophysical properties.

Transport coefficients (viscosity, thermal conductivity, binary/self diffusion) of a dilute gas
from the Chapman–Enskog first-order solution of the Boltzmann equation with the Lennard–Jones
pair potential, plus the ideal-gas thermophysical coefficients (Z, speed of sound, expansion
coefficient, compressibility, Joule–Thomson) and the derived dimensionless groups
(Prandtl/Schmidt/Lewis). The heat capacities and γ are taken from the statistical-mechanics
engine (:mod:`statthermopy.thermodynamics`), so every result descends from the molecular
partition function ``Q = Q_t Q_r Q_v Q_e`` and the LJ molecular potential — no external property
database (REFPROP/CoolProp) is used.

Public API
----------
- :class:`TransportCalculator`, :class:`TransportProperties`
- :func:`binary_diffusion`, :func:`self_diffusion`
- :data:`TRANSPORT_PROPS`, :data:`TRANSPORT_UNITS`
- :mod:`statthermopy.transport.plots` — vs-T / vs-P / 2-D map / multi-property plots
- :mod:`statthermopy.transport.export` — CSV and Tecplot export

The layer is **extensible by design**: future dense-gas (Enskog / corresponding-states) models,
mixture diffusion, plasma transport and combustion/CFD coupling all slot in behind the
:class:`TransportCalculator` / :class:`TransportProperties` interface without changing the
public API.
"""

from __future__ import annotations

from . import export, plots
from .collision import collision_integral, omega_11, omega_22, t_star
from .lennard_jones import (
    combine_epsilon_over_k,
    combine_sigma,
    pair_epsilon_over_k,
    pair_sigma_m,
    reduced_mass,
)
from .transport import (
    TRANSPORT_PROPS,
    TRANSPORT_UNITS,
    TransportCalculator,
    TransportProperties,
    binary_diffusion,
    self_diffusion,
)

__all__ = [
    "TransportCalculator",
    "TransportProperties",
    "binary_diffusion",
    "self_diffusion",
    "TRANSPORT_PROPS",
    "TRANSPORT_UNITS",
    "collision_integral",
    "omega_11",
    "omega_22",
    "t_star",
    "combine_sigma",
    "combine_epsilon_over_k",
    "pair_sigma_m",
    "pair_epsilon_over_k",
    "reduced_mass",
    "plots",
    "export",
]
