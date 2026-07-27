"""Rankine cycle (ideal / non-ideal turbine & pump).

States:
    1  saturated liquid at condenser pressure
    2  compressed liquid after pump      (P = boiler pressure)
    3  superheated vapor at boiler pressure
    4  wet vapor after turbine           (P = condenser pressure)
"""

from __future__ import annotations

from .base import CyclePoint, CycleResult, isentropic_state, sat_liquid, sat_vapor


def rankine(
    fluid=None,
    *,
    P_boiler: float = 8e6,
    P_condenser: float = 1e4,
    T_superheat: float | None = None,
    eta_turbine: float = 1.0,
    eta_pump: float = 1.0,
) -> CycleResult:
    """Compute a Rankine cycle.

    Parameters
    ----------
    fluid:
        Working fluid (default :class:`~thermolab.Gas` ``"H2O"``).
    P_boiler, P_condenser:
        Boiler and condenser pressures [Pa].
    T_superheat:
        Turbine-inlet temperature [K]. If ``None``, saturated vapor at the
        boiler pressure (no superheat).
    eta_turbine, eta_pump:
        Isentropic efficiencies (1.0 = ideal).
    """
    if fluid is None:
        from .. import Gas
        fluid = Gas("H2O")

    st1 = sat_liquid(fluid, P_condenser)                       # state 1
    s2s = isentropic_state(fluid, P=P_boiler, s=st1.s)         # isentropic pump
    h2 = st1.h + (s2s.h - st1.h) / eta_pump
    st2 = fluid.state(P=P_boiler, h=h2)                        # actual pump outlet

    if T_superheat is not None:
        st3 = fluid.state(P=P_boiler, T=T_superheat)
    else:
        st3 = sat_vapor(fluid, P_boiler)

    s4s = isentropic_state(fluid, P=P_condenser, s=st3.s)      # isentropic turbine
    h4 = st3.h - eta_turbine * (st3.h - s4s.h)
    st4 = fluid.state(P=P_condenser, h=h4)

    q_in = st3.h - st2.h
    q_out = st1.h - st4.h
    w_t = st3.h - h4
    w_p = h2 - st1.h
    w_net = w_t - w_p
    eta = w_net / q_in if q_in else float("nan")
    bwr = w_p / w_t if w_t else float("nan")

    res = CycleResult(
        name="Rankine",
        points=[
            CyclePoint("1", st1, "condenser outlet (sat. liquid)"),
            CyclePoint("2", st2, "pump outlet"),
            CyclePoint("3", st3, "boiler/turbine inlet"),
            CyclePoint("4", st4, "turbine outlet"),
        ],
        q={"boiler": q_in, "condenser": q_out},
        w={"turbine": w_t, "pump": w_p},
        eta=eta, net_work=w_net, back_work_ratio=bwr,
    )
    return res