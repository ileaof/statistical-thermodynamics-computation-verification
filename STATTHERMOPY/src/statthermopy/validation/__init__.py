"""Validation of the statistical-mechanics engine against reference data.

Two layers:

* :mod:`statthermopy.validation.base` — the protocol-driven framework
  (:class:`ReferenceSource`, :class:`ValidationRunner`, :class:`ValidationReport`). The
  *calculation core* never touches this; it is used only for optional cross-checks.
* :mod:`statthermopy.validation.reference` — **embedded** NIST/JANAF reference tables for a
  core set of species, plus the :func:`validate` convenience that runs automatic validation out
  of the box.

Only *reference values* (numbers) are embedded — no empirical correlation coefficients
(NASA/Shomate/JANAF polynomials) ship in the package, so the calculation core remains pure
statistical mechanics.
"""

from .base import ReferenceSource, ValidationReport, ValidationRunner
from .reference import (
    EmbeddedReferenceSource,
    NistJanafReference,
    ReferenceRegistry,
    list_references,
    validate,
)

__all__ = [
    "ReferenceSource",
    "ValidationRunner",
    "ValidationReport",
    "EmbeddedReferenceSource",
    "NistJanafReference",
    "ReferenceRegistry",
    "list_references",
    "validate",
]