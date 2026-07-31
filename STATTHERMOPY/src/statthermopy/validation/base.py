"""Validation framework.

Validation compares StatThermoPy's *first-principles* predictions against an external reference
(NIST Webbook, JANAF tables, CEA/NASA). Crucially, the reference data is supplied by the user via
a :class:`ReferenceSource` — StatThermoPy ships **no** embedded empirical property tables, so the
calculation core remains pure statistical mechanics. The reference data is consumed only here, for
optional cross-checking and error reporting.

Example
-------
::

    import pandas as pd
    from statthermopy.validation import ReferenceSource, ValidationRunner

    class NistN2(ReferenceSource):
        def load(self):
            return pd.DataFrame({
                "T": [300, 500, 1000, 2000],
                "Cp": [29.12, 29.26, 30.12, 32.34],   # J/mol/K, from NIST
            })

    runner = ValidationRunner("N2", NistN2())
    print(runner.run().errors_percent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..core.state import State
from ..database import get
from ..thermodynamics import Thermodynamics

__all__ = ["ReferenceSource", "ValidationRunner", "ValidationReport"]


class ReferenceSource(Protocol):
    """A supplier of external reference thermodynamic data.

    Implementations load a table (e.g. from a NIST/JANAF/CEA file the user points to) and return
    it as a structure mapping temperature to reference property values. StatThermoPy does not ship
    any reference data.
    """

    def load(self):  # pragma: no cover - protocol
        """Return the reference data, typically a :class:`pandas.DataFrame` with a ``T`` column."""


@dataclass
class ValidationReport:
    """Outcome of a validation run: per-point predictions, references and percent errors."""

    species: str
    property_name: str
    T: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    reference: list[float] = field(default_factory=list)
    errors_percent: list[float] = field(default_factory=list)

    @property
    def mean_abs_error_percent(self) -> float:
        """Mean absolute percent error over all points."""
        if not self.errors_percent:
            return 0.0
        return sum(abs(e) for e in self.errors_percent) / len(self.errors_percent)

    @property
    def max_abs_error_percent(self) -> float:
        """Worst-case absolute percent error."""
        return max((abs(e) for e in self.errors_percent), default=0.0)

    def __repr__(self) -> str:
        return (
            f"ValidationReport(species={self.species!r}, prop={self.property_name!r}, "
            f"n={len(self.T)}, MAE={self.mean_abs_error_percent:.3f}%, "
            f"max={self.max_abs_error_percent:.3f}%)"
        )


class ValidationRunner:
    """Compare StatThermoPy predictions against a :class:`ReferenceSource`.

    Parameters
    ----------
    species : str
        Molecule name (looked up in the database).
    source : ReferenceSource
        External reference data source (user-supplied; no embedded data).
    property_name : str, default "Cp"
        Attribute of :class:`~statthermopy.thermodynamics.ThermoProperties` to compare, e.g.
        ``"Cp"``, ``"Cp_m"``, ``"H_m"``, ``"S_m"``. If the reference column is a molar quantity,
        use the corresponding ``_m`` attribute.
    pressure : float, default 101325.0
        Pressure (Pa) at which to evaluate predictions.
    """

    #: maps a "simple" property name to the molar ThermoProperties attribute to compare.
    _MOLAR_ATTRS = {
        "Cp": "Cp_m", "Cv": "Cv_m", "H": "H_m", "S": "S_m",
        "U": "U_m", "G": "G_m", "A": "A_m",
    }

    def __init__(
        self,
        species: str,
        source: ReferenceSource,
        *,
        property_name: str = "Cp",
        pressure: float = 101325.0,
    ) -> None:
        self.species = species
        self.source = source
        self.property_name = property_name
        self.pressure = pressure

    def _resolve_attr(self) -> str:
        return self._MOLAR_ATTRS.get(self.property_name, self.property_name)

    def run(self) -> ValidationReport:
        """Run the comparison and return a :class:`ValidationReport`."""
        attr = self._resolve_attr()
        df = self.source.load()
        mol = get(self.species)
        Ts = list(df["T"])
        refs = list(df[self.property_name])
        preds = []
        for T in Ts:
            th = Thermodynamics(mol, State(T=float(T), P=self.pressure)).compute()
            preds.append(getattr(th, attr))
        errs = []
        for p, r in zip(preds, refs):
            if r == 0:
                errs.append(0.0 if p == 0 else float("inf"))
            else:
                errs.append(100.0 * (p - r) / r)
        return ValidationReport(
            species=self.species, property_name=self.property_name,
            T=Ts, predicted=preds, reference=list(refs), errors_percent=errs,
        )