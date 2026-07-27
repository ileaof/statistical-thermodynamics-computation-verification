"""ThermoLab — a unified thermodynamic property & cycle-analysis framework.

ThermoLab wraps mature thermodynamic engines (ThermoPack by default) behind a
single, modern, object-oriented API for computing thermodynamic functions and
thermophysical properties of pure fluids and gas mixtures.

Quickstart
----------
>>> from thermolab import Gas
>>> air = Gas("Air", backend="thermopack")
>>> st = air.state(T=800.0, P=5e5)
>>> st.rho, st.cp, st.gamma, st.sound_speed

Public API
----------
* :class:`Gas`       — pure fluid or named pseudo-fluid (e.g. ``"Air"``).
* :class:`Mixture`   — multicomponent gas mixture.
* :class:`State`     — resolved thermodynamic state with all properties.
* :func:`list_fluids` — fluids supported by the default backend.
* :data:`__version__` — package version.

Submodules: :mod:`thermolab.cfd`, :mod:`thermolab.tables`,
:mod:`thermolab.plotting`, :mod:`thermolab.cycles`, :mod:`thermolab.optimization`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .exceptions import (
    ThermoLabError,
    UnsupportedFluidError,
    BackendError,
    BackendNotAvailableError,
    ConvergenceError,
    TwoPhaseError,
    FlashSpecificationError,
    UnsupportedPropertyError,
    FluidAliasError,
)
from .fluid import Gas, list_fluids
from .mixture import Mixture
from .state import State
from .properties import PropertyBundle
from .backends import (
    BaseBackend,
    Phase,
    ThermoPackBackend,
    available_backends,
    get_backend,
)

__all__ = [
    "Gas",
    "Mixture",
    "State",
    "PropertyBundle",
    "Phase",
    "BaseBackend",
    "ThermoPackBackend",
    "get_backend",
    "available_backends",
    "list_fluids",
    "__version__",
    # exceptions
    "ThermoLabError",
    "UnsupportedFluidError",
    "BackendError",
    "BackendNotAvailableError",
    "ConvergenceError",
    "TwoPhaseError",
    "FlashSpecificationError",
    "UnsupportedPropertyError",
    "FluidAliasError",
]

# Submodules are imported lazily by users (cfd, tables, plotting, cycles,
# optimization) to avoid eager imports of matplotlib/pandas where unwanted.