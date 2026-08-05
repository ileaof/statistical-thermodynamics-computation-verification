"""Air Transport Properties Database — dry and humid air transport from kinetic-theory mixing rules.

This subpackage adds **mixture transport** to the existing pure-species Chapman–Enskog engine
(:mod:`statthermopy.transport`). The dry-air model is the standard N₂/O₂/Ar/CO₂ background; the
humid-air model accounts for water vapour through the statistical :class:`~statthermopy.humidair.
HumidAir` composition and updates every transport property with temperature, pressure and
humidity. Mixture transport is computed with the well-established dilute-gas mixing rules —
**Wilke** (viscosity), **Mason–Saxena** (thermal conductivity) and **Blanc** (mass diffusivity of
water vapour in air) — on top of the per-species values from
:class:`~statthermopy.transport.TransportCalculator` and
:func:`~statthermopy.transport.binary_diffusion`. A structured, extensible per-species transport
database (:mod:`.species_data`) holds the Lennard–Jones parameters, critical properties, acentric
factor and reference coefficients for N₂, O₂, Ar, CO₂, H₂O — the latter fields stored as inputs and
as the documented hook for a future non-ideal / high-pressure (corresponding-states / Enskog)
extension; they do not enter today's dilute-gas calculation path.

Public API
----------
- :class:`AirTransport`, :class:`AirTransportTable`, :class:`AirTransportAnalysis`
- :class:`MixtureTransportCalculator`, :class:`MixtureTransportProperties`,
  :class:`SpeciesTransportContribution`
- :func:`wilke_viscosity`, :func:`mason_saxena_conductivity`, :func:`blanc_diffusion`
- :data:`AIR_TRANSPORT_PROPS`, :data:`AIR_TRANSPORT_UNITS`, :data:`AIR_TRANSPORT_LABELS`
- :class:`SpeciesTransportData`, :class:`CriticalProperties`,
  :func:`get_species_transport`, :func:`list_species_transport`
- :mod:`.plots` — vs-T plots and dry-vs-humid comparison; :mod:`.export` — CSV/Excel/JSON/PDF
"""

from __future__ import annotations

from . import analysis, export, plots
from .air_transport import AirTransport, AirTransportTable
from .analysis import AirTransportAnalysis
from .export import AirTransportExporter
from .mixture_transport import (
    AIR_TRANSPORT_LABELS,
    AIR_TRANSPORT_PROPS,
    AIR_TRANSPORT_UNITS,
    MixtureTransportCalculator,
    MixtureTransportProperties,
    SpeciesTransportContribution,
    blanc_diffusion,
    mason_saxena_conductivity,
    wilke_viscosity,
)
from .species_data import (
    CriticalProperties,
    SpeciesTransportData,
    get_species_transport,
    list_species_transport,
    register_species_transport,
)

__all__ = [
    "AirTransport",
    "AirTransportTable",
    "AirTransportAnalysis",
    "AirTransportExporter",
    "MixtureTransportCalculator",
    "MixtureTransportProperties",
    "SpeciesTransportContribution",
    "wilke_viscosity",
    "mason_saxena_conductivity",
    "blanc_diffusion",
    "AIR_TRANSPORT_PROPS",
    "AIR_TRANSPORT_UNITS",
    "AIR_TRANSPORT_LABELS",
    "SpeciesTransportData",
    "CriticalProperties",
    "get_species_transport",
    "list_species_transport",
    "register_species_transport",
    "plots",
    "export",
    "analysis",
]