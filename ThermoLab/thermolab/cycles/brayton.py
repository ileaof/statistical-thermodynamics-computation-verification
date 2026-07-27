"""Brayton (Joule) cycle — open gas-turbine cycle.

States:
    1  compressor inlet
    2  compressor outlet       (P = P1 * pressure_ratio)
    3  turbine inlet           (constant-pressure heat addition)
    4  turbine outlet          (P = P1)
"""

from __future__ import annotations

from .base import CyclePoint, CycleResult, isentropic_state


def brayton(
    fluid=None,
    *,
    P1: float = 1e5,
    pressure_ratio: float = 10.0,
    T1: float = 300.0,
    T3: float = 1400.0,
    eta_compressor: float = 1.0,
    eta_turbine: float = 1.0,
) -> CycleResult:
    """Compute a Brayton cycle.

    Parameters
    ----------
    fluid:
        Working fluid (default :class:`~thermolab.Gas` ``"Air"``).
    P1:
        Compressor-inlet pressure [Pa].
    pressure_ratio:
        ``P2 / P1``.
    T1, T3:
        Compressor-inlet and turbine-inlet temperatures [K].
    eta_compressor, eta_turbine:
        Isentropic efficiencies (1.0 = ideal).
    """
    if fluid is None:
        from .. import Gas
        fluid = Gas("Air")

    P2 = P1 * pressure_ratio

    st1 = fluid.state(T=T1, P=P1)
    s2s = isentropic_state(fluid, P=P2, s=st1.s)
    h2 = st1.h + (s2s.h - st1.h) / eta_compressor
    st2 = fluid.state(P=P2, h=h2)

    st3 = fluid.state(T=T3, P=P2)

    s4s = isentropic_state(fluid, P=P1, s=st3.s)
    h4 = st3.h - eta_turbine * (st3.h - s4s.h)
    st4 = fluid.state(P=P1, h=h4)

    q_in = st3.h - st2.h
    q_out = st4.h - st1.h
    w_t = st3.h - h4
    w_c = h2 - st1.h
    w_net = w_t - w_c
    eta = w_net / q_in if q_in else float("nan")
    bwr = w_c / w_t if w_t else float("nan")

    return CycleResult(
        name="Brayton",
        points=[
            CyclePoint("1", st1, "compressor inlet"),
            CyclePoint("2", st2, "compressor outlet"),
            CyclePoint("3", st3, "combustor/turbine inlet"),
            CyclePoint("4", st4, "turbine outlet"),
        ],
        q={"combustor": q_in, "exhaust": q_out},
        w={"turbine": w_t, "compressor": w_c},
        eta=eta, net_work=w_net, back_work_ratio=bwr,
    )


# The Joule cycle is the closed ideal-gas form of the Brayton cycle; in
# ThermoLab they share the same calculation (real-fluid states make the open/
# closed distinction immaterial here).
joule = brayton