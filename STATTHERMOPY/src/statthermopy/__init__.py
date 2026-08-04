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

Statistical transport properties (first-principles, optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- :mod:`statthermopy.transport` — transport and thermophysical properties of pure gases derived
  from Chapman–Enskog kinetic theory with the Lennard–Jones potential, atop this ideal-gas
  engine: viscosity ``μ(T,P)``, kinematic viscosity ``ν``, thermal conductivity ``k``, thermal
  diffusivity ``α``, binary diffusion ``D_ij``, Prandtl/Schmidt/Lewis numbers, compressibility
  factor ``Z``, speed of sound, thermal-expansion coefficient ``β``, isothermal compressibility
  ``κ_T`` and the Joule–Thomson coefficient. Uses molecular LJ parameters
  (:class:`LennardJones`), not empirical property data.
"""

from __future__ import annotations

from .constants import R
from .core import (
    Contribution,
    ElectronicLevel,
    Geometry,
    InternalRotor,
    LennardJones,
    Molecule,
    ResolvedState,
    State,
    VibrationalMode,
)
from .fluids import (
    STANDARD_DRY_AIR,
    PredefinedFluid,
    air,
    available_fluids,
    get_fluid,
    register_fluid,
)
from .mixture import ComponentContribution, IdealGasMixture, MixtureProperties
from .modes import Electronic, HinderedRotor, Rotational, Translational, Vibrational
from .partition import PartitionFunction, PartitionValues
from .thermodynamics import ThermoProperties, Thermodynamics

__version__ = "0.1.0"

# Database access (imported lazily-friendly: re-exported here for convenience).
from .database import get, list_molecules  # noqa: E402

__all__ = [
    "R",
    "Contribution",
    "LennardJones",
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
    "ComponentContribution",
    "air",
    "PredefinedFluid",
    "available_fluids",
    "get_fluid",
    "register_fluid",
    "STANDARD_DRY_AIR",
    "Translational",
    "Rotational",
    "Vibrational",
    "HinderedRotor",
    "Electronic",
    "get",
    "list_molecules",
    "__version__",
]