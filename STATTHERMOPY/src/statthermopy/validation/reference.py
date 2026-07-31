"""Embedded NIST/JANAF reference data for automatic validation.

This module ships **curated reference tables** (molar Cp° and absolute molar S° at a
temperature grid, standard state 1 bar) for a core set of well-characterised species, so that
:mod:`statthermopy.validation` can run automatic cross-checks of the first-principles engine
against NIST/JANAF values *out of the box*.

Important — what is and is not shipped
-------------------------------------
Only the *reference values* (numbers) are embedded — **no empirical correlation coefficients**
(NASA/Shomate/JANAF polynomial coefficients) live in the package. The calculation core in
:mod:`statthermopy.thermodynamics` remains pure statistical mechanics. The tables here are used
solely by the validation layer (:class:`~statthermopy.validation.base.ValidationRunner`) for
optional cross-checking and error reporting.

The values were produced by evaluating the NIST Chemistry WebBook Shomate equations for each
species at the tabulated temperatures (the Shomate coefficients themselves are *not* shipped).
Each YAML file under ``statthermopy.validation.data`` cites its source. The data is curated and
refinable; the rigid-rotor / harmonic-oscillator model departs from experiment by up to a few
percent at high temperature (anharmonicity), which the validation tolerance accommodates.

Schema (one YAML per species)::

    species: N2
    source: "NIST Chemistry WebBook, Shomate equation evaluated at T grid; standard state 1 bar"
    pressure: 100000.0          # Pa, standard-state pressure of the reference
    notes: "..."
    T:  [298.15, 400.0, ...]     # K
    Cp: [29.12, 29.25, ...]     # J/mol/K  (column indexed by property_name "Cp")
    S:  [191.61, 200.18, ...]   # J/mol/K  (column indexed by property_name "S")
"""

from __future__ import annotations

import functools
from importlib import resources
from pathlib import Path

import pandas as pd
import yaml

from .base import ReferenceSource, ValidationReport, ValidationRunner

__all__ = [
    "EmbeddedReferenceSource",
    "NistJanafReference",
    "ReferenceRegistry",
    "list_references",
    "validate",
]

#: property columns currently shipped in the reference tables.
REFERENCE_PROPERTIES: tuple[str, ...] = ("Cp", "S")


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class EmbeddedReferenceSource(ReferenceSource):
    """A :class:`ReferenceSource` backed by an embedded YAML reference table.

    Parameters
    ----------
    species : str
        Species name (case-insensitive); must have a YAML file in
        ``statthermopy.validation.data``.
    data_dir : Path, optional
        Override the data directory (mainly for testing).
    """

    def __init__(self, species: str, *, data_dir: Path | None = None) -> None:
        self.species = species
        self._data_dir = data_dir
        self._data: dict | None = None

    # -- loading -----------------------------------------------------------
    def _resolve_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        return Path(resources.files("statthermopy.validation").joinpath("data"))

    def _index(self) -> dict[str, Path]:
        return {p.stem.upper(): p for p in self._resolve_dir().glob("*.y*ml")}

    def _load(self) -> dict:
        if self._data is None:
            index = self._index()
            key = self.species.upper()
            if key not in index:
                available = ", ".join(sorted(index.keys()))
                raise KeyError(
                    f"No embedded reference data for species {self.species!r}. "
                    f"Available: {available}."
                )
            self._data = _load_yaml(index[key])
        return self._data

    # -- public API --------------------------------------------------------
    @property
    def pressure(self) -> float:
        """Standard-state pressure (Pa) declared by the reference (default 1 bar)."""
        return float(self._load().get("pressure", 100000.0))

    @property
    def source(self) -> str:
        """Provenance string for the reference table."""
        return str(self._load().get("source", "embedded reference data"))

    def available_properties(self) -> tuple[str, ...]:
        """Property columns present in the reference table (e.g. ``("Cp", "S")``)."""
        data = self._load()
        return tuple(p for p in REFERENCE_PROPERTIES if p in data)

    def load(self) -> pd.DataFrame:
        """Return the reference data as a :class:`pandas.DataFrame` with a ``T`` column plus
        one column per available property (``Cp``, ``S``).

        This is the single method consumed by
        :class:`~statthermopy.validation.base.ValidationRunner`, which indexes the result by
        ``"T"`` and by the requested ``property_name``.
        """
        data = self._load()
        cols: dict[str, list[float]] = {"T": [float(t) for t in data["T"]]}
        for prop in REFERENCE_PROPERTIES:
            if prop in data:
                cols[prop] = [float(v) for v in data[prop]]
        return pd.DataFrame(cols)


class NistJanafReference(EmbeddedReferenceSource):
    """Embedded reference source backed by the curated NIST/JANAF tables.

    A thin named subclass of :class:`EmbeddedReferenceSource` for explicit, self-documenting
    usage (``NistJanafReference("N2")``) and for user code that wants to pin the provenance by
    name. Behaviour is identical to :class:`EmbeddedReferenceSource`.
    """


class ReferenceRegistry:
    """Lazy-loading registry of embedded reference sources keyed by species name.

    Mirrors :class:`statthermopy.database.registry.MoleculeRegistry`.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir

    def _resolve_dir(self) -> Path:
        if self.data_dir is not None:
            return self.data_dir
        return Path(resources.files("statthermopy.validation").joinpath("data"))

    def _index(self) -> dict[str, Path]:
        return {p.stem.upper(): p for p in self._resolve_dir().glob("*.y*ml")}

    def list_references(self) -> list[str]:
        """Return the species names with embedded reference data (sorted)."""
        return sorted(self._index().keys())

    @functools.lru_cache(maxsize=None)
    def _get_cached(self, name: str) -> EmbeddedReferenceSource:
        index = self._index()
        key = name.upper()
        if key not in index:
            available = ", ".join(sorted(index.keys()))
            raise KeyError(
                f"No embedded reference data for species {name!r}. Available: {available}."
            )
        return EmbeddedReferenceSource(key, data_dir=self._resolve_dir())

    def get(self, name: str) -> EmbeddedReferenceSource:
        """Return the :class:`EmbeddedReferenceSource` for *name* (case-insensitive)."""
        return self._get_cached(name.upper())


# Module-level default registry.
_REGISTRY = ReferenceRegistry()


def list_references() -> list[str]:
    """List the species names that ship embedded NIST/JANAF reference data."""
    return _REGISTRY.list_references()


def validate(
    species: str,
    property_name: str = "Cp",
    *,
    pressure: float | None = None,
) -> ValidationReport:
    """Run automatic validation of the engine against the embedded NIST/JANAF reference.

    Parameters
    ----------
    species : str
        Species name with embedded reference data (see :func:`list_references`).
    property_name : str, default "Cp"
        Property to compare; must be a column of the reference table (``"Cp"`` or ``"S"``).
    pressure : float, optional
        Pressure (Pa) at which to evaluate predictions. Defaults to the standard-state pressure
        declared by the reference (1 bar), which is the correct choice for S (entropy depends on
        pressure via the translational term); Cp is pressure-independent for an ideal gas.

    Returns
    -------
    ValidationReport
        Per-point predicted/reference values and percent errors.

    Raises
    ------
    KeyError
        If *species* has no embedded reference data, or *property_name* is not a column.
    """
    source = _REGISTRY.get(species)
    if pressure is None:
        pressure = source.pressure
    runner = ValidationRunner(species, source, property_name=property_name, pressure=pressure)
    return runner.run()