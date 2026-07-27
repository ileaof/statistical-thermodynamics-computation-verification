"""Thermodynamic cycle analyses.

All cycles return a :class:`CycleResult` whose ``points`` are real-fluid
:class:`~thermolab.state.State` objects (where the working fluid is supported),
so each point's full property set is available for plotting on any diagram.

Cycles
------
* :func:`rankine`       — vapor power cycle (water).
* :func:`brayton` / :func:`joule` — gas-turbine cycle (air).
* :func:`refrigeration` — vapor-compression refrigeration (R134a).
* :func:`otto`          — spark-ignition air-standard cycle (air).
* :func:`diesel`        — compression-ignition air-standard cycle (air).
"""

from __future__ import annotations

from .base import CyclePoint, CycleResult, CycleError
from .rankine import rankine
from .brayton import brayton, joule
from .refrigeration import refrigeration
from .otto import otto
from .diesel import diesel

__all__ = [
    "CyclePoint",
    "CycleResult",
    "CycleError",
    "rankine",
    "brayton",
    "joule",
    "refrigeration",
    "otto",
    "diesel",
]