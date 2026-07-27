"""Fluid objects: :class:`Gas` (pure fluid or named pseudo-mixture) and the
shared :class:`_FluidBase`.

A fluid object owns a backend bound to a component set and a *reference*
composition. Calling :meth:`state` with any two independent variables produces a
:class:`thermolab.state.State`.
"""

from __future__ import annotations

import numpy as np

from ._fluid_db import is_alias, normalize_name, resolve_alias
from .backends import get_backend
from .backends.base import Phase
from .exceptions import FluidAliasError, UnsupportedFluidError
from .flash import flash
from .state import State

_DEFAULT_BACKEND = "thermopack"


class _FluidBase:
    """Common machinery for :class:`Gas` and :class:`Mixture`."""

    def __init__(self, backend: str, components: list[str], fractions, *,
                 reference_state: str = "DEFAULT", eos: str | None = None):
        self._backend_name = backend
        self._components = list(components)
        self._fractions = np.asarray(fractions, dtype=float)
        if len(self._fractions) != len(self._components):
            raise ValueError("components and fractions must have the same length.")
        if (self._fractions < 0).any():
            raise ValueError("fractions must be non-negative.")
        total = self._fractions.sum()
        if total <= 0:
            raise ValueError("fractions must sum to a positive value.")
        self._fractions = self._fractions / total  # normalize
        self._backend = get_backend(
            backend, self._components, reference_state=reference_state, eos=eos
        )

    # ------------------------------------------------------------------
    @property
    def backend(self):
        return self._backend

    @property
    def components(self) -> list[str]:
        return list(self._components)

    @property
    def fractions(self) -> np.ndarray:
        return self._fractions.copy()

    @property
    def molar_mass(self) -> float:
        """Mixture molar mass of the reference composition [kg/mol]."""
        return float(np.dot(self._fractions, self._backend.molar_masses()))

    def set_fractions(self, fractions) -> None:
        """Update the reference composition (mole fractions)."""
        f = np.asarray(fractions, dtype=float)
        if len(f) != len(self._components):
            raise ValueError("fractions length must match components.")
        if (f < 0).any():
            raise ValueError("fractions must be non-negative.")
        s = f.sum()
        if s <= 0:
            raise ValueError("fractions must sum to a positive value.")
        self._fractions = f / s

    # ------------------------------------------------------------------
    def state(self, *, phase: str | Phase | None = None, **specs) -> State:
        """Create a :class:`State` from two independent variables.

        Examples
        --------
        >>> gas.state(T=500.0, P=2e5)
        >>> gas.state(P=1e5, h=2.5e6)
        """
        fr = flash(self._backend, self._fractions, specs, phase=phase)
        return State(
            self._backend, self._fractions, fr.T, fr.P, fr.phase,
            two_phase=fr.two_phase, quality=fr.quality, sat=fr.sat,
        )

    # ------------------------------------------------------------------
    def saturation_pressure(self, T: float) -> float:
        """Saturation pressure [Pa] at temperature ``T`` (pure fluid)."""
        return self._backend.saturation_pressure(T, self._fractions)

    def saturation_temperature(self, P: float) -> float:
        """Saturation temperature [K] at pressure ``P`` (pure fluid)."""
        return self._backend.saturation_temperature(P, self._fractions)

    def critical_temperature(self) -> float:
        return self._backend.critical_temperature(self._fractions)

    def critical_pressure(self) -> float:
        return self._backend.critical_pressure(self._fractions)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} components={self._components} "
            f"backend={self._backend_name!r}>"
        )


class Gas(_FluidBase):
    """A pure fluid or a named pseudo-fluid (e.g. ``"Air"``).

    Parameters
    ----------
    name:
        Component name (case-insensitive, e.g. ``"N2"``, ``"H2O"``,
        ``"R134a"``) or a named alias such as ``"Air"``.
    backend:
        Thermodynamic backend name (default ``"thermopack"``).
    reference_state:
        ThermoPack reference state (default ``"DEFAULT"``).
    eos:
        Optional explicit EOS override; otherwise chosen automatically
        (MEOS for pure multiparameter fluids, GERG2008 for Air-like mixtures,
        SRK cubic otherwise).
    """

    def __init__(self, name: str, *, backend: str = _DEFAULT_BACKEND,
                 reference_state: str = "DEFAULT", eos: str | None = None):
        self.name = str(name)
        norm = normalize_name(name)
        if is_alias(norm):
            components, fractions = resolve_alias(norm)
            self._is_alias = True
        else:
            components, fractions = [name], [1.0]
            self._is_alias = False
        super().__init__(backend, components, fractions,
                         reference_state=reference_state, eos=eos)

    @property
    def is_pseudo_mixture(self) -> bool:
        """True if this gas is a named composition alias (e.g. Air)."""
        return self._is_alias


def list_fluids() -> list[str]:
    """Return the curated list of fluid names supported by the default backend."""
    from ._fluid_db import KNOWN_SUPPORTED, FLUID_ALIASES
    return sorted(set(KNOWN_SUPPORTED) | set(FLUID_ALIASES))