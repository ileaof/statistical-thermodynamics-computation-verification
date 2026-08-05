"""The :class:`HumidAir` model — maximum water-vapour solubility and full psychrometrics.

``HumidAir`` combines a **statistical-mechanics** description of the gas phase (dry air + water
vapour, each species from its molecular partition function via the ideal-mixture relations) with
the vapour–liquid :class:`~statthermopy.humidair.saturation.SaturationCalculator` to answer the
central question: *how much water vapour can the air hold at a given temperature and pressure
before it condenses?* From the saturation limit it derives the complete psychrometric and
thermodynamic property set, and it exposes how each partition-function factor (translational,
rotational, vibrational, electronic) of the water molecule builds up the vapour Gibbs energy that
sets the saturation.

The dry air defaults to the standard N₂/O₂/Ar/CO₂ composition (see
:mod:`statthermopy.fluids`) but any :class:`~statthermopy.mixture.IdealGasMixture` may be supplied,
so trace gases, other dry-gas backgrounds, and (in future) planetary atmospheres slot in without
touching the saturation physics.
"""

from __future__ import annotations

import math

from ..constants import R
from ..core.state import State
from ..database import get
from ..fluids import air
from ..mixture import IdealGasMixture
from ..thermodynamics import Thermodynamics
from .saturation import SaturationCalculator
from .state import HumidAirState

__all__ = ["HumidAir"]


class HumidAir:
    """Statistical humid-air model over a dry-gas background and a water saturation calculator.

    Parameters
    ----------
    dry_air : IdealGasMixture, optional
        Dry-gas background. Defaults to standard dry air (N₂/O₂/Ar/CO₂).
    saturation : SaturationCalculator, optional
        Water saturation model. Defaults to the standard calculator (IAPWS-95 liquid if available,
        else the constant-``c_p`` reference).
    """

    def __init__(
        self,
        dry_air: IdealGasMixture | None = None,
        saturation: SaturationCalculator | None = None,
    ) -> None:
        self.dry_air = dry_air if dry_air is not None else air()
        self.saturation = saturation if saturation is not None else SaturationCalculator()
        self.M_dry = self.dry_air.M_avg                       # kg/mol
        self.M_water = get("H2O").molar_mass                  # kg/mol
        self.epsilon = self.M_water / self.M_dry              # ~0.622
        self._dry_x = {mol.name: x for mol, x in self.dry_air.x.items()}

    # -- convenience ----------------------------------------------------------

    def saturation_pressure(self, T: float) -> float:
        """Water saturation pressure at ``T`` (Pa)."""
        return self.saturation.saturation_pressure(T)

    def max_mole_fraction(self, T: float, P: float) -> float:
        """Maximum H₂O mole fraction (solubility) at ``(T, P)`` = P_sat/P, capped at 1."""
        return min(self.saturation.saturation_pressure(T) / float(P), 1.0)

    # -- humid-gas mixture ----------------------------------------------------

    def _humid_mixture(self, x_w: float) -> IdealGasMixture:
        """Ideal-gas mixture of the dry background scaled to ``1 − x_w`` plus H₂O at ``x_w``."""
        if x_w <= 0.0:
            return self.dry_air
        comp = {name: x * (1.0 - x_w) for name, x in self._dry_x.items()}
        comp["H2O"] = comp.get("H2O", 0.0) + x_w
        return IdealGasMixture.from_names(comp, basis="mole")

    def _vapor_mode_contributions(self, T: float, P_v: float) -> dict:
        """Per-partition-function-factor contribution to the water-vapour molar properties.

        The ideal-gas ``P V = R T`` (Gibbs/enthalpy) term is assigned to the translational factor,
        so the per-mode ``G_m``/``H_m`` sum to the totals.
        """
        st = State(T=T, P=P_v)
        contribs = Thermodynamics(self.saturation.molecule, st).partition.contributions(st)
        out: dict[str, dict] = {}
        for name, c in contribs.items():
            pv = R * T if name == "translational" else 0.0
            out[name] = {
                "ln_q": c.ln_q, "U_m": c.U_m, "H_m": c.U_m + pv,
                "S_m": c.S_m, "A_m": c.A_m, "G_m": c.A_m + pv, "Cv_m": c.Cv_m,
            }
        return out

    # -- adiabatic-saturation (thermodynamic) wet-bulb temperature ------------

    def _wet_bulb(self, T: float, P: float, x_w: float) -> float:
        """Adiabatic-saturation temperature (K): energy balance on a per-mol-dry-air basis.

        h_da(T) + N h_v(T) + (N_s* − N) h_l(T*) = h_da(T*) + N_s* h_v(T*),
        with N = P_v/(P−P_v) the vapour/dry-air mole ratio and N_s* its saturation value at T*.
        """
        P_v = x_w * P
        if P_v >= P:  # pragma: no cover - unphysical guard
            return T
        N = P_v / (P - P_v)

        def h_da(t: float) -> float:
            return self.dry_air.compute(State(T=t, P=P)).H_m

        def h_v(t: float) -> float:
            return self.saturation.vapour_properties(t, P).H_m

        def h_l(t: float) -> float:
            return self.saturation.liquid_enthalpy(t, P)

        def Ns(t: float) -> float:
            ps = self.saturation.saturation_pressure(t)
            return ps / (P - ps) if ps < P else float("inf")

        def f(t: float) -> float:
            ns = Ns(t)
            lhs = h_da(T) + N * h_v(T) + (ns - N) * h_l(t)
            rhs = h_da(t) + ns * h_v(t)
            return lhs - rhs

        lo = self.saturation.dew_point(P_v) if P_v > 0 else 150.0
        hi = T
        if hi - lo < 1e-6:
            return T  # already saturated
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            # f(dew) > 0, f(T) < 0  -> monotone decreasing
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-4:
                break
        return 0.5 * (lo + hi)

    # -- full state -----------------------------------------------------------

    def state(
        self,
        T: float,
        P: float,
        *,
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
        saturated: bool = False,
        wet_bulb: bool = True,
    ) -> HumidAirState:
        """Evaluate the complete moist-air state at ``(T, P)``.

        The actual water content is set by at most one of ``relative_humidity`` (0–1),
        ``humidity_ratio`` (kg/kg dry air) or ``mole_fraction``; with none given (or
        ``saturated=True``) the **saturation limit** is used — the maximum-water-holding state,
        which is the module's headline result. Set ``wet_bulb=False`` to skip the (iterative)
        adiabatic-saturation temperature.
        """
        T = float(T)
        P = float(P)
        given = [relative_humidity is not None, humidity_ratio is not None, mole_fraction is not None]
        if sum(given) > 1:
            raise ValueError(
                "Specify at most one of relative_humidity, humidity_ratio, mole_fraction."
            )

        P_sat = self.saturation.saturation_pressure(T)
        x_sat = min(P_sat / P, 1.0)
        eps = self.epsilon

        if saturated or not any(given):
            x_w = x_sat
            is_sat = True
        elif relative_humidity is not None:
            x_w = min(float(relative_humidity) * P_sat / P, 1.0)
            is_sat = False
        elif mole_fraction is not None:
            x_w = float(mole_fraction)
            is_sat = False
        else:  # humidity_ratio
            r = float(humidity_ratio) / eps            # = P_v/(P - P_v)
            x_w = r / (1.0 + r)
            is_sat = False
        x_w = max(0.0, min(x_w, 1.0 - 1e-12))

        P_v = x_w * P
        mix = self._humid_mixture(x_w)
        pm = mix.compute(State(T=T, P=P))

        def w_of(pp: float) -> float:
            return eps * pp / (P - pp) if pp < P else float("inf")

        w = w_of(P_v)
        w_sat = w_of(P_sat)
        M_avg = pm.M_avg
        M_avg_sat = (1.0 - x_sat) * self.M_dry + x_sat * self.M_water
        dew = self.saturation.dew_point(P_v) if P_v > 0.0 else float("nan")
        wb = self._wet_bulb(T, P, x_w) if wet_bulb else float("nan")

        return HumidAirState(
            T=T, P=P, saturated=is_sat, liquid_model=self.saturation.liquid.name,
            # saturation limit
            P_sat=P_sat,
            x_h2o_max=x_sat,
            mass_fraction_h2o_max=x_sat * self.M_water / M_avg_sat,
            humidity_ratio_max=w_sat,
            absolute_humidity_max=P_sat * self.M_water / (R * T),
            vapor_concentration_max=P_sat / (R * T),
            # actual state
            P_vapor=P_v,
            x_h2o=x_w,
            mass_fraction_h2o=x_w * self.M_water / M_avg,
            humidity_ratio=w,
            absolute_humidity=P_v * self.M_water / (R * T),
            vapor_concentration=P_v / (R * T),
            relative_humidity=(P_v / P_sat) if P_sat > 0 else float("nan"),
            degree_of_saturation=(w / w_sat) if math.isfinite(w) and w_sat > 0 else float("nan"),
            dew_point=dew,
            wet_bulb=wb,
            # bulk
            density=P * M_avg / (R * T),
            M_avg=M_avg,
            R_specific=R / M_avg,
            # molar thermodynamics (moist-air mixture)
            U_m=pm.U_m, H_m=pm.H_m, S_m=pm.S_m, A_m=pm.A_m, G_m=pm.G_m,
            Cv_m=pm.Cv_m, Cp_m=pm.Cp_m, gamma=pm.gamma, mu_m=pm.mu_m, S_mixing=pm.S_mixing,
            # massic thermodynamics
            U_s=pm.U_s, H_s=pm.H_s, S_s=pm.S_s, A_s=pm.A_s, G_s=pm.G_s,
            Cv_s=pm.Cv_s, Cp_s=pm.Cp_s,
            # breakdowns
            components=dict(pm.components),
            vapor_mode_contributions=self._vapor_mode_contributions(T, P_v if P_v > 0 else P_sat),
        )

    # alias
    max_solubility = state
