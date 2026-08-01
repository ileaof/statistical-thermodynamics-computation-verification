"""StatThermoPy — statistical thermodynamics in Python.

Compute thermodynamic properties of ideal gases exclusively from the molecular partition
function ``Q = Q_t Q_r Q_v Q_e``, with no empirical property correlations.

Public API
----------
Classes
~~~~~~~
- :class:`Molecule`, :class:`Geometry`, :class:`VibrationalMode`, :class:`ElectronicLevel`
- :class:`State`
- :class:`PartitionFunction`
- :class:`Thermodynamics`, :class:`ThermoProperties`
- :class:`IdealGasMixture`, :class:`MixtureProperties`

Database
~~~~~~~~
- :func:`get` (retrieve a molecule by name), :func:`list_molecules` (all available names).
"""

from __future__ import annotations

from .constants import R
from .core import (
    Contribution,
    ElectronicLevel,
    Geometry,
    InternalRotor,
    Molecule,
    ResolvedState,
    State,
    VibrationalMode,
)
from .mixture import IdealGasMixture, MixtureProperties
from .modes import Electronic, HinderedRotor, Rotational, Translational, Vibrational
from .partition import PartitionFunction, PartitionValues
from .thermodynamics import ThermoProperties, Thermodynamics

__version__ = "0.1.0"

# Database access (imported lazily-friendly: re-exported here for convenience).
from .database import get, list_molecules  # noqa: E402

__all__ = [
    "R",
    "Contribution",
    "ElectronicLevel",
    "Geometry",
    "Molecule",
    "VibrationalMode",
    "InternalRotor",
    "State",
    "ResolvedState",
    "PartitionFunction",
    "PartitionValues",
    "Thermodynamics",
    "ThermoProperties",
    "IdealGasMixture",
    "MixtureProperties",
    "Translational",
    "Rotational",
    "Vibrational",
    "HinderedRotor",
    "Electronic",
    "get",
    "list_molecules",
    "__version__",
]