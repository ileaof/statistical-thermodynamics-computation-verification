"""Extended per-species transport-property database records for the air-relevant species.

This registry *extends* the molecular database (:mod:`statthermopy.database`) with the
transport-database constants requested for a complete air transport-property reference:

* the **Lennard–Jones** collision diameter ``σ`` and well depth ``ε/k`` (mirrored from the
  per-species molecular YAML so a record is self-contained for display/export),
* the **critical properties** (``Tc``, ``Pc``, ``Vc``, ``Zc``) and the **acentric factor** ``ω``,
* **viscosity reference coefficients** (a Sutherland-form reference point, informational),
* **thermal-conductivity reference coefficients** (a reference point, informational),
* a **binary-diffusion provenance note**.

Design / philosophy
-------------------
The dilute-gas calculation path in :mod:`statthermopy.transport` is **first principles**:
Chapman–Enskog kinetic theory with the Lennard–Jones potential plus the Wilke / Mason–Saxena /
Blanc mixing rules. The critical properties, acentric factor and reference coefficients stored
here are therefore **not consumed by today's calculation** — they are structured inputs kept on
the same footing as the spectroscopic constants, and the documented hook for a future non-ideal
/ high-pressure (corresponding-states / Enskog) transport extension. No empirical property
correlation enters the calculation path.

The records live **outside** the :class:`~statthermopy.core.molecule.Molecule` class (which stays
focused on spectroscopic constants); a dedicated test guards that ``Molecule`` never gains a
``critical`` attribute. The registry is a plain dict keyed by the species name (uppercase) and is
open: :func:`register_species_transport` adds a new record, and additional species can be added by
dropping entries into ``database/data/air_transport.yaml``.

The extended fields are loaded from ``database/data/air_transport.yaml``; the molar mass and LJ
parameters are merged from the molecular database (:func:`statthermopy.database.get`) so the LJ
values remain a single source of truth.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CriticalProperties",
    "ViscosityCoeffs",
    "ConductivityCoeffs",
    "SpeciesTransportData",
    "get_species_transport",
    "list_species_transport",
    "register_species_transport",
]


@dataclass(frozen=True)
class CriticalProperties:
    """Critical-point constants of a species (future dense-gas / corresponding-states input).

    Attributes
    ----------
    Tc : float
        Critical temperature (K).
    Pc : float
        Critical pressure (Pa).
    Vc : float
        Critical molar volume (m³/mol).
    Zc : float
        Critical compressibility factor (``Pc Vc / (R Tc)``), dimensionless.
    """

    Tc: float
    Pc: float
    Vc: float
    Zc: float


@dataclass(frozen=True)
class ViscosityCoeffs:
    """Sutherland-form viscosity reference coefficients (informational; not used by the dilute path).

    The Sutherland form ``μ(T) = μ_ref (T_ref + S) / (T + S) (T / T_ref)^{3/2}`` reproduces the
    dilute-gas viscosity of simple gases to a few percent; stored here as a *reference* for
    cross-checks and for a future empirical-correlation fallback, not as a calculation input.
    """

    mu_ref: float
    T_ref: float
    Sutherland_S: float


@dataclass(frozen=True)
class ConductivityCoeffs:
    """Thermal-conductivity reference point (informational; not used by the dilute path)."""

    k_ref: float
    T_ref: float


@dataclass(frozen=True)
class SpeciesTransportData:
    """A complete transport-property database record for one species.

    The Lennard–Jones parameters and molar mass are mirrored from the molecular database for
    self-containment (display / export); the critical properties, acentric factor and reference
    coefficients are the *extended* transport-database fields loaded from ``air_transport.yaml``.
    """

    name: str
    formula: str
    molar_mass_gmol: float
    sigma_angstrom: float
    epsilon_over_k: float
    critical: CriticalProperties | None
    acentric_factor: float | None
    viscosity_coeffs: ViscosityCoeffs | None
    conductivity_coeffs: ConductivityCoeffs | None
    binary_diffusion_note: str
    note: str
    source: str

    @property
    def has_critical(self) -> bool:
        """``True`` if critical properties are available."""
        return self.critical is not None

    def as_dict(self) -> dict[str, Any]:
        """Flat dictionary view suitable for export / display."""
        return {
            "name": self.name,
            "formula": self.formula,
            "molar_mass_gmol": self.molar_mass_gmol,
            "sigma_angstrom": self.sigma_angstrom,
            "epsilon_over_k": self.epsilon_over_k,
            "critical": (
                {"Tc": self.critical.Tc, "Pc": self.critical.Pc,
                 "Vc": self.critical.Vc, "Zc": self.critical.Zc}
                if self.critical is not None else None
            ),
            "acentric_factor": self.acentric_factor,
            "viscosity_coeffs": (
                {"mu_ref": self.viscosity_coeffs.mu_ref,
                 "T_ref": self.viscosity_coeffs.T_ref,
                 "Sutherland_S": self.viscosity_coeffs.Sutherland_S}
                if self.viscosity_coeffs is not None else None
            ),
            "conductivity_coeffs": (
                {"k_ref": self.conductivity_coeffs.k_ref,
                 "T_ref": self.conductivity_coeffs.T_ref}
                if self.conductivity_coeffs is not None else None
            ),
            "binary_diffusion_note": self.binary_diffusion_note,
            "note": self.note,
            "source": self.source,
        }


# -- loader -------------------------------------------------------------------

def _data_path() -> Path:
    # Lives under the air subpackage's own data dir — NOT under statthermopy.database.data, whose
    # registry globs every *.y*ml and would otherwise treat this file as a pseudo-molecule.
    return Path(resources.files("statthermopy.transport.air").joinpath("data/air_transport.yaml"))


def _load_yaml() -> dict:
    with open(_data_path(), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_record(name: str, raw: dict) -> SpeciesTransportData:
    """Merge the molecular database (mass + LJ) with the extended YAML fields."""
    from ...database import get  # local import: avoid a circular import at module import time

    mol = get(name)
    lj = mol.lennard_jones
    if lj is None:  # pragma: no cover - the air species all carry LJ
        raise ValueError(f"{name} has no Lennard–Jones parameters in the molecular database.")

    crit_raw = raw.get("critical")
    critical = (
        CriticalProperties(
            Tc=float(crit_raw["Tc"]), Pc=float(crit_raw["Pc"]),
            Vc=float(crit_raw["Vc"]), Zc=float(crit_raw["Zc"]),
        )
        if crit_raw else None
    )

    visc_raw = raw.get("viscosity_coeffs")
    viscosity_coeffs = (
        ViscosityCoeffs(
            mu_ref=float(visc_raw["mu_ref"]),
            T_ref=float(visc_raw["T_ref"]),
            Sutherland_S=float(visc_raw["Sutherland_S"]),
        )
        if visc_raw else None
    )

    cond_raw = raw.get("conductivity_coeffs")
    conductivity_coeffs = (
        ConductivityCoeffs(k_ref=float(cond_raw["k_ref"]), T_ref=float(cond_raw["T_ref"]))
        if cond_raw else None
    )

    return SpeciesTransportData(
        name=mol.name,
        formula=mol.formula,
        molar_mass_gmol=mol.molar_mass_gmol,
        sigma_angstrom=lj.sigma_angstrom,
        epsilon_over_k=lj.epsilon_over_k,
        critical=critical,
        acentric_factor=(float(raw["acentric_factor"]) if raw.get("acentric_factor") is not None else None),
        viscosity_coeffs=viscosity_coeffs,
        conductivity_coeffs=conductivity_coeffs,
        binary_diffusion_note=str(raw.get("binary_diffusion_note", "")),
        note=str(raw.get("note", "")),
        source=str(raw.get("source", "")),
    )


class _SpeciesTransportRegistry:
    """Lazy registry of :class:`SpeciesTransportData`, keyed by uppercase species name."""

    def __init__(self) -> None:
        self._cache: dict[str, SpeciesTransportData] = {}
        self._extra: dict[str, SpeciesTransportData] = {}

    def _index(self) -> dict[str, dict]:
        return {k.upper(): v for k, v in _load_yaml().items()}

    def list_names(self) -> list[str]:
        names = set(self._index().keys()) | set(self._extra.keys())
        return sorted(names)

    def get(self, name: str) -> SpeciesTransportData:
        key = name.upper()
        if key in self._extra:
            return self._extra[key]
        if key in self._cache:
            return self._cache[key]
        index = self._index()
        if key not in index:
            available = ", ".join(sorted(index.keys()))
            raise KeyError(
                f"No extended transport record for {name!r}. Available: {available}."
            )
        rec = _build_record(key, index[key])
        self._cache[key] = rec
        return rec

    def register(self, record: SpeciesTransportData) -> None:
        """Register (or replace) a record, keyed by its name (case-insensitive)."""
        self._extra[record.name.upper()] = record
        self._cache.pop(record.name.upper(), None)


_REGISTRY = _SpeciesTransportRegistry()


def get_species_transport(name: str) -> SpeciesTransportData:
    """Return the :class:`SpeciesTransportData` for ``name`` (case-insensitive)."""
    return _REGISTRY.get(name)


def list_species_transport() -> list[str]:
    """Return the species names that have an extended transport record (sorted)."""
    return _REGISTRY.list_names()


def register_species_transport(record: SpeciesTransportData) -> None:
    """Register (or replace) a :class:`SpeciesTransportData` record (extensibility hook)."""
    _REGISTRY.register(record)