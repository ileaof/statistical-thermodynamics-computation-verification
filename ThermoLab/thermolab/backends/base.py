"""Abstract backend interface.

All quantities exchanged with a backend are **molar** SI (per mole). Conversion
to the mass-based units that ThermoLab exposes publicly happens in
:class:`thermolab.state.State`.

Implementations should be **stateless w.r.t. the thermodynamic state**: a single
backend instance is bound to a fluid/composition definition and reused across
many state evaluations (important for CFD performance). Composition ``z`` is
passed explicitly to every call so the same backend can serve multiple states.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Phase(str, Enum):
    """Phase labels used across the public API (backend-agnostic)."""

    VAPOR = "vapor"
    LIQUID = "liquid"
    TWOPHASE = "twophase"
    UNKNOWN = "unknown"


@dataclass
class MolarProperties:
    """Container for the molar property set returned by a backend.

    All fields are in molar SI units:

    * ``v``  molar volume        [m^3/mol]
    * ``h``  molar enthalpy      [J/mol]
    * ``s``  molar entropy       [J/(mol.K)]
    * ``u``  molar internal energy [J/mol]
    * ``a``  molar Helmholtz energy [J/mol]
    * ``g``  molar Gibbs energy  [J/mol]
    * ``cp`` molar isobaric heat capacity [J/(mol.K)]
    * ``cv`` molar isochoric heat capacity [J/(mol.K)]
    * ``Z``  compressibility factor [-]
    * ``w``  speed of sound      [m/s]
    * ``beta``  isobaric thermal expansion   [1/K]
    * ``kappa_t`` isothermal compressibility [1/Pa]
    * ``jt``  Joule-Thomson coefficient       [K/Pa]
    """

    v: float
    h: float
    s: float
    u: float
    a: float
    g: float
    cp: float
    cv: float
    Z: float
    w: float
    beta: float
    kappa_t: float
    jt: float


@dataclass
class SaturationState:
    """Saturated liquid/vapor properties for a pure fluid at (T, P)."""

    T: float
    P: float
    # liquid ("f") and vapor ("g") saturated values
    rho_f: float
    rho_g: float
    h_f: float
    h_g: float
    s_f: float
    s_g: float


class BaseBackend(ABC):
    """Abstract thermodynamic backend.

    Subclasses are constructed with a fluid definition (component list and
    reference composition) and expose property/flash/saturation/curve methods.
    """

    #: Short identifier ("thermopack", "coolprop", ...).
    name: str = "base"

    def __init__(self, components: list[str], *, reference_state: str = "DEFAULT", eos: str | None = None):
        self.components = [str(c) for c in components]
        self.nc = len(self.components)
        self.reference_state = reference_state
        self.eos = eos

    # ------------------------------------------------------------------
    # Composition / identity
    # ------------------------------------------------------------------
    @abstractmethod
    def molar_masses(self) -> np.ndarray:
        """Return per-component molar masses in kg/mol, shape ``(nc,)``."""

    def mixture_molar_mass(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        return float(np.dot(z, self.molar_masses()))

    # ------------------------------------------------------------------
    # Phase & single-phase properties
    # ------------------------------------------------------------------
    @abstractmethod
    def guess_phase(self, T: float, P: float, z: np.ndarray) -> Phase:
        """Return the stable single-phase label at (T, P, z)."""

    @abstractmethod
    def molar_properties(self, T: float, P: float, z: np.ndarray, phase: Phase | str) -> MolarProperties:
        """Return the full molar property set for the given single phase."""

    @abstractmethod
    def specific_volume(self, T: float, P: float, z: np.ndarray, phase: Phase | str) -> float:
        """Molar specific volume [m^3/mol] (convenience for flash solvers)."""

    def pressure_at_volume(self, T: float, v_molar: float, z: np.ndarray) -> float:
        """Pressure [Pa] at temperature ``T`` and molar volume ``v_molar``.

        Used by the flash solver for volume-based specifications
        (``rho``/``v``). Backends without a (T, v) interface should leave this
        to raise ``NotImplementedError``; the flash falls back to a generic
        solver in that case.
        """
        raise NotImplementedError(f"{self.name} backend does not implement pressure_at_volume.")

    # ------------------------------------------------------------------
    # Saturation / phase equilibrium (pure fluids primarily)
    # ------------------------------------------------------------------
    def saturation_pressure(self, T: float, z: np.ndarray | None = None) -> float:
        """Saturation pressure [Pa] at temperature ``T``."""
        raise NotImplementedError(f"{self.name} backend does not implement saturation_pressure.")

    def saturation_temperature(self, P: float, z: np.ndarray | None = None) -> float:
        """Saturation temperature [K] at pressure ``P``."""
        raise NotImplementedError(f"{self.name} backend does not implement saturation_temperature.")

    def saturation_state(self, T: float, z: np.ndarray | None = None) -> SaturationState:
        """Saturated liquid/vapor properties at ``T`` (pure fluid)."""
        raise NotImplementedError(f"{self.name} backend does not implement saturation_state.")

    def is_two_phase(self, T: float, P: float, z: np.ndarray) -> bool:
        """Return True if (T, P, z) lies inside the two-phase region."""
        raise NotImplementedError(f"{self.name} backend does not implement is_two_phase.")

    # ------------------------------------------------------------------
    # Critical point
    # ------------------------------------------------------------------
    def critical_temperature(self, z: np.ndarray | None = None) -> float:
        raise NotImplementedError(f"{self.name} backend does not implement critical_temperature.")

    def critical_pressure(self, z: np.ndarray | None = None) -> float:
        raise NotImplementedError(f"{self.name} backend does not implement critical_pressure.")

    # ------------------------------------------------------------------
    # Optional curve helpers (for plotting). May return None / raise.
    # ------------------------------------------------------------------
    def saturation_curve(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(f"{self.name} backend does not implement saturation_curve.")

    def isobar(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(f"{self.name} backend does not implement isobar.")

    def isotherm(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(f"{self.name} backend does not implement isotherm.")

    def isentrope(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(f"{self.name} backend does not implement isentrope.")