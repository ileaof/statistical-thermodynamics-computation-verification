"""Vapor-compression refrigeration cycle.

States:
    1  compressor inlet   (saturated or slightly superheated vapor)
    2  compressor outlet  (P = condenser pressure)
    3  condenser outlet   (saturated liquid)
    4  throttle outlet    (P = evaporator pressure, isenthalpic)
"""

from __future__ import annotations

from .base import CyclePoint, CycleResult, isentropic_state, sat_liquid, sat_vapor


def refrigeration(
    fluid=None,
    *,
    T_evap: float = 263.15,        # -10 C
    T_cond: float = 313.15,        #  40 C
    superheat: float = 5.0,
    eta_compressor: float = 1.0,
) -> CycleResult:
    """Compute a vapor-compression refrigeration cycle.

    Parameters
    ----------
    fluid:
        Working fluid (default :class:`~thermolab.Gas` ``"R134a"``).
    T_evap, T_cond:
        Evaporator and condenser temperatures [K].
    superheat:
        Compressor-inlet superheat above saturation [K].
    eta_compressor:
        Isentropic compressor efficiency (1.0 = ideal).
    """
    if fluid is None:
        from .. import Gas
        fluid = Gas("R134a")

    P_evap = fluid.saturation_pressure(T_evap)
    P_cond = fluid.saturation_pressure(T_cond)

    if superheat > 0:
        st1 = fluid.state(P=P_evap, T=T_evap + superheat)
    else:
        st1 = sat_vapor(fluid, P_evap)

    s2s = isentropic_state(fluid, P=P_cond, s=st1.s)
    h2 = st1.h + (s2s.h - st1.h) / eta_compressor
    st2 = fluid.state(P=P_cond, h=h2)

    st3 = sat_liquid(fluid, P_cond)
    st4 = fluid.state(P=P_evap, h=st3.h)   # isenthalpic throttle

    q_l = st1.h - st4.h        # refrigeration effect
    q_h = st2.h - st3.h        # condenser heat rejection
    w_c = h2 - st1.h
    cop = q_l / w_c if w_c else float("nan")

    return CycleResult(
        name="Refrigeration (vapor-compression)",
        points=[
            CyclePoint("1", st1, "compressor inlet"),
            CyclePoint("2", st2, "compressor outlet"),
            CyclePoint("3", st3, "condenser outlet (sat. liquid)"),
            CyclePoint("4", st4, "throttle outlet"),
        ],
        q={"evaporator": q_l, "condenser": q_h},
        w={"compressor": w_c},
        cop=cop, net_work=w_c,
    )