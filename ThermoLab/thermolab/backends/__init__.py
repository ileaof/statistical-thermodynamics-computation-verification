"""Backend abstraction for ThermoLab.

A *backend* is the thermodynamic engine that actually evaluates equations of
state. ThermoLab ships a ThermoPack backend; the :class:`BaseBackend` interface
lets other engines (CoolProp, Cantera, REFPROP, pycalphad) be plugged in later
without changing the public API.
"""

from __future__ import annotations

from .base import BaseBackend, Phase
from .registry import get_backend, register_backend, available_backends
from .thermopack_backend import ThermoPackBackend

__all__ = [
    "BaseBackend",
    "Phase",
    "ThermoPackBackend",
    "get_backend",
    "register_backend",
    "available_backends",
]