"""Vapour–liquid saturation of water from the equality of chemical potentials.

The saturation (dew) line is fixed by phase equilibrium between liquid water and its vapour,

    μ_v(T, P) = μ_l(T, P)      ⇔      g_v(T, P) = g_l(T, P) .

The **vapour** Gibbs energy is obtained *exclusively* from the molecular partition function
(translational + rotational + vibrational + electronic), via
:class:`~statthermopy.thermodynamics.Thermodynamics` on the database's H₂O molecule — the same
first-principles engine used everywhere else. Its absolute entropy (Sackur–Tetrode + rigid rotor +
harmonic oscillators + electronic) reproduces the experimental standard entropy of steam to better
than 0.1 %, so ``g_v(T, P)`` carries the correct temperature dependence with no fitted constant.

The **liquid** Gibbs energy comes from a pluggable :class:`~statthermopy.humidair.liquid.
LiquidWaterModel`. Because the two phases use different energy-reference conventions, the liquid
scale is reconciled to the vapour scale with **two physical anchors at the triple point**:

* coexistence there, ``g_v(T_t, P_t) = g_l(T_t, P_t)`` ;
* the enthalpy of vaporisation ``Δh_vap(T_t)`` (a single calorimetric constant),

which together fix the liquid's enthalpy- and entropy-reference offsets (Δh₀, Δs₀). No empirical
vapour-pressure correlation (Antoine, Magnus, Tetens, …) is used: the temperature dependence of
``P_sat`` is *predicted* by the statistical vapour together with the liquid heat capacity.

For an ideal vapour ``g_v(T, P) = g_v(T, P_ref) + R T ln(P/P_ref)`` and a nearly incompressible
liquid ``g_l(T, P) = g_l(T, P_ref) + v_l (P − P_ref)`` (the Poynting term), so at each temperature

    P_sat = P_ref · exp{ [g_l(T, P) − g_v(T, P_ref)] / (R T) } ,

solved by a short fixed-point iteration (the Poynting correction shifts ``P_sat`` by < 0.1 %).
"""

from __future__ import annotations

import math

from ..constants import R
from ..core.state import State
from ..database import get
from ..thermodynamics import Thermodynamics
from .liquid import (
    DH_VAP_TRIPLE,
    P_TRIPLE,
    T_CRITICAL,
    T_TRIPLE,
    LiquidWaterModel,
    default_liquid_model,
)

__all__ = ["SaturationCalculator"]


class SaturationCalculator:
    """Solve water's vapour–liquid saturation from ``g_v = g_l`` (statistical vapour).

    Parameters
    ----------
    liquid : LiquidWaterModel, optional
        Liquid-water reference model. Defaults to IAPWS-95 if the ``iapws`` package is available,
        otherwise the transparent constant-``c_p`` reference.
    species : str, default "H2O"
        Condensing species (kept general so the machinery can be reused for other vapours).
    triple : tuple(float, float), optional
        ``(T_t, P_t)`` triple-point anchor (K, Pa). Defaults to water's.
    dh_vap_triple : float, optional
        Enthalpy of vaporisation at the triple point (J/mol). Defaults to water's 45.054 kJ/mol.
    """

    def __init__(
        self,
        liquid: LiquidWaterModel | None = None,
        *,
        species: str = "H2O",
        triple: tuple[float, float] | None = None,
        dh_vap_triple: float = DH_VAP_TRIPLE,
    ) -> None:
        self.molecule = get(species)
        self.liquid = liquid if liquid is not None else default_liquid_model()
        self.T_triple, self.P_triple = triple if triple is not None else (T_TRIPLE, P_TRIPLE)
        self.dh_vap_triple = float(dh_vap_triple)
        self.molar_mass = self.molecule.molar_mass  # kg/mol
        self._anchor()

    # -- vapour (statistical mechanics) ---------------------------------------

    def vapour_properties(self, T: float, P: float):
        """:class:`ThermoProperties` of the water vapour at ``(T, P)`` (first-principles)."""
        return Thermodynamics(self.molecule, State(T=float(T), P=float(P))).compute()

    def vapour_gibbs(self, T: float, P: float) -> float:
        """Molar Gibbs energy g_v(T, P) of the vapour (J/mol), from the partition function."""
        return self.vapour_properties(T, P).G_m

    # -- reference reconciliation ---------------------------------------------

    def _anchor(self) -> None:
        """Fix the liquid enthalpy/entropy offsets (Δh₀, Δs₀) from the triple-point anchors."""
        vp = self.vapour_properties(self.T_triple, self.P_triple)
        hl0 = self.liquid.enthalpy(self.T_triple, self.P_triple)
        sl0 = self.liquid.entropy(self.T_triple, self.P_triple)
        # aligned liquid must satisfy h_l(T_t) = h_v − Δh_vap and s_l(T_t) = s_v − Δh_vap/T_t
        self._dh0 = vp.H_m - self.dh_vap_triple - hl0
        self._ds0 = vp.S_m - self.dh_vap_triple / self.T_triple - sl0

    def liquid_enthalpy(self, T: float, P: float) -> float:
        """Liquid molar enthalpy on the vapour's absolute scale (J/mol)."""
        return self.liquid.enthalpy(T, P) + self._dh0

    def liquid_entropy(self, T: float, P: float) -> float:
        """Liquid molar entropy on the vapour's absolute scale (J/mol/K)."""
        return self.liquid.entropy(T, P) + self._ds0

    def liquid_gibbs(self, T: float, P: float) -> float:
        """Liquid molar Gibbs energy on the vapour's absolute scale (J/mol)."""
        return self.liquid_enthalpy(T, P) - T * self.liquid_entropy(T, P)

    # -- saturation -----------------------------------------------------------

    def saturation_pressure(self, T: float, *, poynting: bool = True, tol: float = 1e-9) -> float:
        """Saturation (vapour) pressure of water at temperature ``T`` (Pa).

        Solves ``g_v(T, P) = g_l(T, P)`` for ``P``; ``poynting`` toggles the liquid pressure
        correction (a < 0.1 % effect).
        """
        T = float(T)
        P_ref = self.P_triple
        gv_ref = self.vapour_gibbs(T, P_ref)             # ideal vapour: g_v(T,P)=g_v(T,P_ref)+RT ln(P/P_ref)
        gl_ref = self.liquid_gibbs(T, P_ref)
        v_l = self.liquid.molar_volume(T, P_ref)
        P = P_ref * math.exp((gl_ref - gv_ref) / (R * T))
        if not poynting:
            return P
        for _ in range(50):
            gl = gl_ref + v_l * (P - P_ref)
            P_new = P_ref * math.exp((gl - gv_ref) / (R * T))
            if abs(P_new - P) <= tol * P_new:
                return P_new
            P = P_new
        return P  # pragma: no cover - converges in a few iterations

    def saturation_temperature(self, P: float, *, tol: float = 1e-6) -> float:
        """Temperature at which the saturation pressure equals ``P`` (K) — inverts ``P_sat(T)``.

        This is also the **dew point** of moist air whose water partial pressure is ``P``. Bounded
        to the liquid–vapour range ``[T_triple, T_critical)``; below the triple-point pressure the
        equilibrium is with ice (frost point), which this liquid model does not cover, so the
        result is clamped to the triple point.
        """
        P = float(P)
        # Liquid-vapour equilibrium exists only between the triple and critical points.
        lo, hi = self.T_triple, T_CRITICAL - 1e-3
        if self.saturation_pressure(lo) >= P:
            return lo
        if self.saturation_pressure(hi) <= P:  # pragma: no cover - above ~647 K
            return hi
        # P_sat is monotonic increasing in T -> bisection
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.saturation_pressure(mid) < P:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    def dew_point(self, water_partial_pressure: float) -> float:
        """Dew-point temperature (K) for a given water-vapour partial pressure (Pa)."""
        return self.saturation_temperature(water_partial_pressure)

    def enthalpy_of_vaporisation(self, T: float) -> float:
        """Enthalpy of vaporisation Δh_vap(T) = h_v(T, P_sat) − h_l(T) (J/mol)."""
        P = self.saturation_pressure(T)
        return self.vapour_properties(T, P).H_m - self.liquid_enthalpy(T, P)
