"""Thermodynamic state — the central user-facing object.

A :class:`State` is created by :meth:`thermolab.Gas.state` /
:meth:`thermolab.Mixture.state` from any pair of independent variables. It
exposes every property in the ThermoLab catalogue as a **lazy, cached,
mass-based** attribute, which makes it suitable for repeated evaluation inside
CFD / heat-transfer loops: each property is computed once on first access and
reused thereafter.

Two-phase states (detected via the backend) are reported with a ``quality`` and
blended intensive properties where defined; single-phase-only quantities
(``cp``, ``cv``, ``sound_speed``, transport) are returned as ``nan`` with a
warning, since they are not meaningful for a flashing mixture.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import Any

import numpy as np

from .backends.base import BaseBackend, Phase
from .exceptions import TwoPhaseError
from .properties import PropertyBundle, format_property_table
from . import transport
from .units import mixture_molar_mass

_NAN = float("nan")


class State:
    """A fully determined thermodynamic state (mass-based SI)."""

    __slots__ = (
        "_backend", "_z", "_M", "_T", "_P", "_phase", "_two_phase",
        "_quality", "_sat", "_names", "_Tcs", "_M_i", "__dict__",
    )

    def __init__(
        self,
        backend: BaseBackend,
        z: np.ndarray,
        T: float,
        P: float,
        phase: Phase,
        *,
        two_phase: bool = False,
        quality: float | None = None,
        sat: Any = None,
    ) -> None:
        self._backend = backend
        self._z = np.asarray(z, dtype=float)
        self._M = mixture_molar_mass(self._z, backend.molar_masses())
        self._T = float(T)
        self._P = float(P)
        self._phase = phase
        self._two_phase = bool(two_phase)
        self._quality = quality
        self._sat = sat
        # transport inputs (precomputed once, cheap)
        try:
            self._names = backend.component_names_normalized()  # type: ignore[attr-defined]
            self._Tcs = backend.component_critical_temperatures()  # type: ignore[attr-defined]
        except AttributeError:
            self._names = [str(c) for c in backend.components]
            self._Tcs = np.array([None] * backend.nc, dtype=object)
        self._M_i = backend.molar_masses()

    # ------------------------------------------------------------------
    # Core coordinates
    # ------------------------------------------------------------------
    @property
    def T(self) -> float:
        return self._T

    @property
    def P(self) -> float:
        return self._P

    @property
    def phase(self) -> str:
        return self._phase.value if isinstance(self._phase, Phase) else str(self._phase)

    @property
    def two_phase(self) -> bool:
        return self._two_phase

    @property
    def quality(self) -> float | None:
        return self._quality

    @property
    def backend(self) -> BaseBackend:
        return self._backend

    @property
    def composition(self) -> np.ndarray:
        return self._z.copy()

    @property
    def molar_mass(self) -> float:
        """Mixture molar mass [kg/mol]."""
        return self._M

    # ------------------------------------------------------------------
    # Single-phase molar property set (cached)
    # ------------------------------------------------------------------
    @cached_property
    def _mp(self):
        """Molar property set for the single-phase root at (T, P).

        Selects the root matching this state's phase so that liquid states
        (compressed / saturated liquid) read the liquid root rather than the
        vapour root. Two-phase / unknown states fall back to the vapour root,
        which is only used when the quality-based blend in ``s``/``h``/``rho``
        does not apply.
        """
        if self._phase == Phase.LIQUID:
            return self._backend.molar_properties(self._T, self._P, self._z, Phase.LIQUID)
        return self._backend.molar_properties(self._T, self._P, self._z, Phase.VAPOR)

    @cached_property
    def _mp_liq(self):
        return self._backend.molar_properties(self._T, self._P, self._z, Phase.LIQUID)

    # ------------------------------------------------------------------
    # Mass-based properties
    # ------------------------------------------------------------------
    @cached_property
    def rho(self) -> float:
        if self._two_phase and self._sat is not None and self._quality is not None:
            rf = self._sat.rho_f * self._M  # mol/m3 -> kg/m3
            rg = self._sat.rho_g * self._M
            x = self._quality
            return 1.0 / ((1.0 - x) / rf + x / rg)
        v_mol = self._mp.v
        return self._M / v_mol

    @cached_property
    def v(self) -> float:
        return 1.0 / self.rho

    @cached_property
    def h(self) -> float:
        if self._two_phase and self._sat is not None and self._quality is not None:
            hf = self._sat.h_f / self._M
            hg = self._sat.h_g / self._M
            return hf + self._quality * (hg - hf)
        return self._mp.h / self._M

    @cached_property
    def s(self) -> float:
        if self._two_phase and self._sat is not None and self._quality is not None:
            sf = self._sat.s_f / self._M
            sg = self._sat.s_g / self._M
            return sf + self._quality * (sg - sf)
        return self._mp.s / self._M

    @cached_property
    def u(self) -> float:
        return self.h - self.P * self.v

    @cached_property
    def g(self) -> float:
        return self.h - self.T * self.s

    @cached_property
    def a_helmholtz(self) -> float:
        return self.u - self.T * self.s

    @cached_property
    def cp(self) -> float:
        self._require_single_phase("cp")
        return self._mp.cp / self._M

    @cached_property
    def cv(self) -> float:
        self._require_single_phase("cv")
        return self._mp.cv / self._M

    @cached_property
    def gamma(self) -> float:
        self._require_single_phase("gamma")
        cv = self._mp.cv
        if cv == 0.0:
            return _NAN
        return self._mp.cp / cv

    @cached_property
    def Z(self) -> float:
        if self._two_phase:
            return _NAN
        return self._mp.Z

    @cached_property
    def sound_speed(self) -> float:
        self._require_single_phase("sound_speed")
        return self._mp.w

    # alias used in the spec
    @property
    def a(self) -> float:
        """Alias for :attr:`sound_speed` (speed of sound, m/s)."""
        return self.sound_speed

    # ------------------------------------------------------------------
    # Derivative-based properties (intensive; identical molar/mass)
    # ------------------------------------------------------------------
    @cached_property
    def joule_thomson(self) -> float:
        if self._two_phase:
            return _NAN
        return self._mp.jt

    @cached_property
    def beta_thermal_expansion(self) -> float:
        if self._two_phase:
            return _NAN
        return self._mp.beta

    @cached_property
    def kappa_t(self) -> float:
        """Isothermal compressibility [1/Pa]."""
        if self._two_phase:
            return _NAN
        return self._mp.kappa_t

    # ------------------------------------------------------------------
    # Transport (gas-phase correlations)
    # ------------------------------------------------------------------
    @cached_property
    def mu(self) -> float:
        """Dynamic viscosity [Pa.s] (gas-phase correlation)."""
        if self._two_phase:
            return _NAN
        if self.phase == "liquid":
            transport.warn_transport_approx("liquid")
        return transport.mixture_viscosity(
            self._T, self._z, self._M_i, self._names,
            [float(t) if t is not None else None for t in self._Tcs],
        )

    @cached_property
    def k(self) -> float:
        """Thermal conductivity [W/(m.K)] (gas-phase Eucken estimate)."""
        if self._two_phase:
            return _NAN
        if self.phase == "liquid":
            transport.warn_transport_approx("liquid")
        mu = self.mu
        cp_molar = self._mp.cp
        return transport._eucken_conductivity(mu, cp_molar, self._M)

    @cached_property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity alpha = k / (rho * cp) [m^2/s]."""
        if self._two_phase:
            return _NAN
        rho = self.rho
        cp = self.cp
        if rho == 0.0 or cp == 0.0:
            return _NAN
        return self.k / (rho * cp)

    @cached_property
    def prandtl(self) -> float:
        """Prandtl number Pr = mu * cp / k [-]."""
        if self._two_phase:
            return _NAN
        k = self.k
        if k == 0.0:
            return _NAN
        return self.mu * self.cp / k

    # ------------------------------------------------------------------
    # Helpers / output
    # ------------------------------------------------------------------
    def _require_single_phase(self, prop: str) -> None:
        if self._two_phase:
            raise TwoPhaseError(
                f"{prop!r} is not defined for a two-phase state "
                f"(quality={self._quality}). Use saturated liquid/vapor values."
            )

    def bundle(self) -> PropertyBundle:
        """Materialize the full :class:`PropertyBundle` (all properties)."""
        return PropertyBundle(
            T=self.T, P=self.P, rho=self.rho, v=self.v,
            u=self.u, h=self.h, s=self.s, g=self.g, a_helmholtz=self.a_helmholtz,
            cp=self._safe("cp"), cv=self._safe("cv"),
            gamma=self._safe("gamma"), Z=self._safe("Z"),
            sound_speed=self._safe("sound_speed"),
            mu=self._safe("mu"), k=self._safe("k"),
            thermal_diffusivity=self._safe("thermal_diffusivity"),
            prandtl=self._safe("prandtl"),
            joule_thomson=self._safe("joule_thomson"),
            beta_thermal_expansion=self._safe("beta_thermal_expansion"),
            kappa_t=self._safe("kappa_t"),
            phase=self.phase, quality=self._quality,
        )

    def _safe(self, prop: str) -> float:
        try:
            val = getattr(self, prop)
            return float(val)
        except TwoPhaseError:
            return _NAN

    def to_dict(self) -> dict[str, Any]:
        return self.bundle().to_dict()

    def to_series(self):
        return self.bundle().to_series()

    def __repr__(self) -> str:
        title = (
            f"ThermoLab State: T={self.T:.4g} K, P={self.P:.4g} Pa, "
            f"phase={self.phase}"
        )
        if self._two_phase:
            q = "n/a" if self._quality is None else f"{self._quality:.4f}"
            title += f", two-phase (quality={q})"
        return f"{title}\n{format_property_table(self.bundle())}"