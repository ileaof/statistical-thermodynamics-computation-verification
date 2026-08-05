"""Liquid-water reference models for the vapour–liquid saturation problem.

Applying statistical thermodynamics *directly* to the liquid is the hard part of the humid-air
problem: liquid water is a strongly hydrogen-bonded, associating fluid whose molecular partition
function is not separable into translational/rotational/vibrational/electronic factors (the modes
are coupled by the intermolecular potential, and the configurational integral does not factorise).
A rigorous molecular treatment therefore requires a *many-body* statistical model — see
``docs/HUMID_AIR.md`` for the critical analysis and recommended approaches (SAFT, thermodynamic
perturbation theory, lattice/association models, integral equations).

Pending a fully first-principles liquid, this module provides a small, **pluggable** hierarchy of
liquid *reference* models. Each supplies the liquid's molar enthalpy, entropy and volume on its
own arbitrary reference scale; the :class:`~statthermopy.humidair.saturation.SaturationCalculator`
reconciles that scale with the statistical-mechanics vapour scale using two physical anchors (the
triple point and the enthalpy of vaporisation there), so only the liquid's *temperature
dependence* (essentially its heat capacity) and molar volume actually enter the result.

Two models are provided:

* :class:`ConstantCpLiquid` — transparent, dependency-free: constant molar heat capacity and molar
  volume. Predicts water's saturation pressure to ≈0.1 % near room temperature and ≈1.5 % at
  100 °C when combined with the statistical vapour (no empirical vapour-pressure correlation).
* :class:`IAPWSLiquid` — the international reference formulation IAPWS-95 (via the optional
  ``iapws`` package) for the liquid's accurate ``c_p,l(T)``, ``s_l(T)`` and ``v_l(T)``. Highest
  accuracy; used automatically when ``iapws`` is importable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

#: Triple point of water (defining/measured constants), used as the reference anchor.
T_TRIPLE: float = 273.16          # K
P_TRIPLE: float = 611.657         # Pa
#: Enthalpy of vaporisation of water at the triple point (J/mol) — the single calorimetric anchor
#: that reconciles the liquid reference scale with the statistical-mechanics vapour scale.
DH_VAP_TRIPLE: float = 45054.0    # J/mol
#: Critical point of water (upper validity bound for the liquid reference).
T_CRITICAL: float = 647.096       # K
P_CRITICAL: float = 22.064e6      # Pa

__all__ = [
    "LiquidWaterModel",
    "ConstantCpLiquid",
    "IAPWSLiquid",
    "default_liquid_model",
    "T_TRIPLE",
    "P_TRIPLE",
    "DH_VAP_TRIPLE",
    "T_CRITICAL",
    "P_CRITICAL",
]


class LiquidWaterModel(ABC):
    """Abstract liquid-water reference: molar enthalpy, entropy and volume of the liquid.

    Values may be on any self-consistent reference scale — the saturation solver anchors the scale
    to the vapour via the triple point and the enthalpy of vaporisation, so only the *temperature
    dependence* matters. All quantities are molar and SI (J/mol, J/mol/K, m³/mol).
    """

    #: Human-readable model name.
    name: str = "liquid"
    #: Validity range (K) as a hint for callers.
    T_min: float = T_TRIPLE
    T_max: float = T_CRITICAL

    @abstractmethod
    def enthalpy(self, T: float, P: float) -> float:
        """Liquid molar enthalpy h_l(T, P) (J/mol, arbitrary reference)."""

    @abstractmethod
    def entropy(self, T: float, P: float) -> float:
        """Liquid molar entropy s_l(T, P) (J/mol/K, arbitrary reference)."""

    @abstractmethod
    def molar_volume(self, T: float, P: float) -> float:
        """Liquid molar volume v_l(T, P) (m³/mol) — for the Poynting pressure correction."""


class ConstantCpLiquid(LiquidWaterModel):
    """Incompressible liquid with a constant molar heat capacity.

    The simplest honest reference: ``h_l = c_p (T − T_ref)``, ``s_l = c_p ln(T/T_ref)``, constant
    molar volume. Hypotheses: temperature-independent ``c_p,l`` and ``v_l``, incompressible liquid.
    These hold to ≈1–2 % for water over 0–100 °C; the residual error grows toward the boiling point
    where ``c_p,l`` rises. No empirical vapour-pressure correlation is used.

    Parameters
    ----------
    cp : float, default 75.35
        Liquid molar heat capacity (J/mol/K).
    molar_volume_value : float, default 1.807e-5
        Liquid molar volume (m³/mol) ≈ 18.07 cm³/mol.
    T_ref : float, default 273.16
        Reference temperature for the enthalpy/entropy zero (K).
    """

    name = "constant-cp"

    def __init__(
        self,
        cp: float = 75.35,
        molar_volume_value: float = 1.807e-5,
        T_ref: float = T_TRIPLE,
    ) -> None:
        self.cp = float(cp)
        self._v = float(molar_volume_value)
        self.T_ref = float(T_ref)

    def enthalpy(self, T: float, P: float) -> float:
        return self.cp * (T - self.T_ref)

    def entropy(self, T: float, P: float) -> float:
        import math
        return self.cp * math.log(T / self.T_ref)

    def molar_volume(self, T: float, P: float) -> float:
        return self._v


class IAPWSLiquid(LiquidWaterModel):
    """Liquid water from the IAPWS-95 international reference formulation (optional ``iapws``).

    Uses the accurate reference ``c_p,l(T)``, ``s_l(T)`` and ``v_l(T)`` of the saturated liquid.
    Only the liquid *reference* comes from IAPWS; the vapour phase remains pure statistical
    mechanics and the two are joined by the chemical-potential equality with the triple-point
    anchor, so no empirical vapour-pressure correlation enters the calculation.
    """

    name = "iapws95"

    def __init__(self) -> None:
        self._iapws = _import_iapws()
        from ..database import get
        self._M = get("H2O").molar_mass  # kg/mol

    def _sat_liquid(self, T: float):
        # Saturated liquid (x = 0) at temperature T; properties are weak functions of P.
        # Clamp to the liquid-vapour range so stray edge probes never raise.
        T = min(max(float(T), T_TRIPLE), T_CRITICAL - 1e-3)
        return self._iapws.IAPWS95(T=T, x=0.0)

    def enthalpy(self, T: float, P: float) -> float:
        st = self._sat_liquid(T)
        return st.h * 1.0e3 * self._M      # kJ/kg -> J/kg -> J/mol

    def entropy(self, T: float, P: float) -> float:
        st = self._sat_liquid(T)
        return st.s * 1.0e3 * self._M      # kJ/kg/K -> J/mol/K

    def molar_volume(self, T: float, P: float) -> float:
        st = self._sat_liquid(T)
        return st.v * self._M              # m³/kg -> m³/mol


def _import_iapws():
    try:
        import iapws
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "IAPWSLiquid requires the optional 'iapws' package (pip install iapws)."
        ) from exc
    return iapws


def default_liquid_model() -> LiquidWaterModel:
    """Return IAPWS-95 if the ``iapws`` package is importable, else the constant-cp reference."""
    try:
        return IAPWSLiquid()
    except ImportError:  # pragma: no cover - depends on the environment
        return ConstantCpLiquid()
