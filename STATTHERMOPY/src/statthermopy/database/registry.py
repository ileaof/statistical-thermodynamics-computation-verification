"""Molecular database: load molecules from YAML.

Each molecule lives in its own YAML file under :mod:`statthermopy.database.data`, making the
database trivially extensible — drop a new YAML in and it is immediately available via
:func:`get`. No empirical *property* data is stored here, only *spectroscopic* constants
(moments of inertia / rotational temperatures, vibrational wavenumbers, electronic term
energies) from which all thermodynamic properties are derived.

YAML schema (optional fields in *italics*)::

    name: N2
    formula: N2
    molar_mass_gmol: 28.0134
    geometry: linear            # monoatomic | linear | nonlinear
    n_atoms: 2
    symmetry_number: 2
    rotational_temperatures: [2.878]      # K  (alternative: moments_of_inertia in kg m^2)
    vibrational_modes:                    # list of {wavenumber_cm1, degeneracy}
      - {wavenumber_cm1: 2358.57, degeneracy: 1}
    internal_rotors:                      # optional; each replaces one 3N-6/3N-5 oscillator
      - {rotation_constant_cm1: 10.7, barrier_cm1: 1024.0, symmetry: 3, n_minima: 3, degeneracy: 1}
    electronic_levels:                    # list of {energy_cm1, degeneracy} (ground at 0)
      - {energy_cm1: 0.0, degeneracy: 1}
    lennard_jones:                         # optional; molecular potential params (NOT property data).
      {sigma_angstrom: 3.798, epsilon_over_k: 71.4}   # used by Chapman-Enskog transport module.
    references: "McQuarrie, Statistical Mechanics; NIST CCCDB"
"""

from __future__ import annotations

import functools
import math
from importlib import resources
from pathlib import Path

import yaml

from ..constants import h, k_B
from ..core.molecule import (
    ElectronicLevel,
    Geometry,
    InternalRotor,
    LennardJones,
    Molecule,
    VibrationalMode,
)

__all__ = ["get", "list_molecules", "load_molecule", "MoleculeRegistry"]


def _rotational_temperatures_to_moments(theta: list[float]) -> tuple[float, ...]:
    """Convert rotational temperatures (K) to principal moments of inertia (kg m^2).

    ``θ_rot = h^2 / (8 π^2 I k_B)``  ->  ``I = h^2 / (8 π^2 θ k_B)``.
    """
    return tuple((h * h) / (8.0 * math.pi * math.pi * th * k_B) for th in theta)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_molecule(data: dict) -> Molecule:
    """Build a :class:`Molecule` from a parsed YAML mapping."""
    geometry = Geometry(data["geometry"])

    # Moments of inertia: accept rotational_temperatures (K) or moments_of_inertia (kg m^2).
    if "moments_of_inertia" in data and data["moments_of_inertia"] is not None:
        moments = tuple(float(m) for m in data["moments_of_inertia"])
    elif "rotational_temperatures" in data and data["rotational_temperatures"] is not None:
        moments = _rotational_temperatures_to_moments([float(t) for t in data["rotational_temperatures"]])
    else:
        moments = tuple()

    vib_modes = tuple(
        VibrationalMode(float(m["wavenumber_cm1"]), int(m.get("degeneracy", 1)))
        for m in (data.get("vibrational_modes") or [])
    )
    rotors = tuple(
        InternalRotor(
            rotation_constant_cm1=float(r["rotation_constant_cm1"]),
            barrier_cm1=float(r["barrier_cm1"]),
            symmetry=int(r.get("symmetry", 3)),
            n_minima=int(r.get("n_minima", 3)),
            degeneracy=int(r.get("degeneracy", 1)),
        )
        for r in (data.get("internal_rotors") or [])
    )
    elec_levels = tuple(
        ElectronicLevel(float(l["energy_cm1"]), int(l.get("degeneracy", 1)))
        for l in (data.get("electronic_levels") or [])
    )

    lj_data = data.get("lennard_jones")
    lennard_jones = None
    if lj_data:
        lennard_jones = LennardJones(
            sigma_angstrom=float(lj_data["sigma_angstrom"]),
            epsilon_over_k=float(lj_data["epsilon_over_k"]),
            note=str(lj_data.get("note", "")),
        )

    return Molecule(
        name=str(data["name"]),
        formula=str(data.get("formula", data["name"])),
        molar_mass_gmol=float(data["molar_mass_gmol"]),
        geometry=geometry,
        n_atoms=int(data["n_atoms"]),
        symmetry_number=int(data.get("symmetry_number", 1)),
        moments_of_inertia=moments,
        vibrational_modes=vib_modes,
        internal_rotors=rotors,
        electronic_levels=elec_levels,
        lennard_jones=lennard_jones,
    )


class MoleculeRegistry:
    """Lazy-loading registry of molecules keyed by name."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir
        self._cache: dict[str, Molecule] = {}

    def _resolve_dir(self) -> Path:
        if self.data_dir is not None:
            return self.data_dir
        return Path(resources.files("statthermopy.database").joinpath("data"))

    def _index(self) -> dict[str, Path]:
        d = self._resolve_dir()
        return {p.stem.upper(): p for p in d.glob("*.y*ml")}

    def list_molecules(self) -> list[str]:
        """Return the available molecule names (sorted)."""
        return sorted(self._index().keys())

    @functools.cache
    def _get_cached(self, name: str) -> Molecule:
        index = self._index()
        key = name.upper()
        if key not in index:
            available = ", ".join(sorted(index.keys()))
            raise KeyError(f"Unknown molecule {name!r}. Available: {available}.")
        return load_molecule(_load_yaml(index[key]))

    def get(self, name: str) -> Molecule:
        """Return the :class:`Molecule` with the given name (case-insensitive)."""
        return self._get_cached(name.upper())


# Module-level default registry.
_REGISTRY = MoleculeRegistry()


def get(name: str) -> Molecule:
    """Retrieve a molecule by name from the default registry (case-insensitive)."""
    return _REGISTRY.get(name)


def list_molecules() -> list[str]:
    """List all molecule names available in the default registry."""
    return _REGISTRY.list_molecules()
