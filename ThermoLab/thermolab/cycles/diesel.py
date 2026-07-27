"""Diesel cycle — compression-ignition, air-standard.

States:
    1  start of compression
    2  end of isentropic compression   (v = v1 / r)
    3  end of constant-pressure heat addition (v = v2 * cutoff_ratio)
    4  end of isentropic expansion     (v = v1)
"""

from __future__ import annotations

from .base import CyclePoint, CycleResult


def diesel(
    fluid=None,
    *,
    compression_ratio: float = 18.0,
    cutoff_ratio: float = 2.0,
    T1: float = 300.0,
    P1: float = 1e5,
) -> CycleResult:
    """Compute a Diesel cycle.

    Parameters
    ----------
    fluid:
        Working fluid (default :class:`~thermolab.Gas` ``"Air"``).
    compression_ratio:
        ``r = v1 / v2``.
    cutoff_ratio:
        ``rc = v3 / v2`` (constant-pressure heat addition volume ratio).
    T1, P1:
        State-1 temperature [K] and pressure [Pa].
    """
    if fluid is None:
        from .. import Gas
        fluid = Gas("Air")

    r = compression_ratio
    st1 = fluid.state(T=T1, P=P1)
    v1 = st1.v
    v2 = v1 / r
    v3 = v2 * cutoff_ratio

    st2 = fluid.state(v=v2, s=st1.s)       # isentropic compression
    st3 = fluid.state(P=st2.P, v=v3)       # constant-pressure heat addition
    st4 = fluid.state(v=v1, s=st3.s)       # isentropic expansion

    q_in = st3.h - st2.h                   # constant pressure
    q_out = st4.u - st1.u                  # constant volume (rejected)
    w_net = q_in - q_out
    eta = 1.0 - q_out / q_in if q_in else float("nan")

    return CycleResult(
        name="Diesel",
        points=[
            CyclePoint("1", st1, "start of compression"),
            CyclePoint("2", st2, "end of compression"),
            CyclePoint("3", st3, "end of heat addition"),
            CyclePoint("4", st4, "end of expansion"),
        ],
        q={"q_in": q_in, "q_out": q_out},
        w={"w_net": w_net},
        eta=eta, net_work=w_net,
    )