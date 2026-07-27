"""Cycle / property optimization helpers (lightweight).

Wraps :mod:`scipy.optimize` to maximize or minimize a scalar objective derived
from a cycle (e.g. thermal efficiency, net work, COP) over design parameters
such as pressure ratio or peak temperature.

Example
-------
>>> from thermolab import cycles, optimization as opt
>>> def builder(rp):
...     return cycles.brayton(pressure_ratio=rp)
>>> res = opt.optimize_cycle(builder, bounds=[(4, 40)], objective="eta")
>>> res.x, res.fun
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def optimize_cycle(
    cycle_builder: Callable,
    *,
    bounds: list[tuple[float, float]],
    objective: str = "eta",
    maximize: bool = True,
    x0: list[float] | None = None,
    method: str = "L-BFGS-B",
) -> object:
    """Optimize a cycle metric over design parameters.

    Parameters
    ----------
    cycle_builder:
        ``f(*params) -> CycleResult``. Called with the trial parameters unpacked.
    bounds:
        Per-parameter ``(min, max)`` bounds.
    objective:
        Attribute of :class:`~thermolab.cycles.CycleResult` to optimize
        (``"eta"``, ``"net_work"``, ``"cop"``, ...).
    maximize:
        If True, maximize the objective; else minimize.
    x0:
        Optional starting guess.
    method:
        Scipy optimizer (``"L-BFGS-B"`` for local, ``"differential_evolution"``
        handled separately).
    """
    from scipy.optimize import minimize

    sign = -1.0 if maximize else 1.0

    def neg(x):
        try:
            res = cycle_builder(*x)
            val = float(getattr(res, objective))
        except Exception:
            return 1e30
        if not np.isfinite(val):
            return 1e30
        return sign * val

    if method == "differential_evolution":
        from scipy.optimize import differential_evolution
        return differential_evolution(neg, bounds)

    return minimize(neg, x0=[b[0] for b in bounds] if x0 is None else x0,
                    bounds=bounds, method=method)


def sweep(cycle_builder: Callable, param_name: str, values, *,
          objective: str = "eta"):
    """One-parameter sweep returning ``(values, objective_values)`` arrays.

    ``cycle_builder`` must accept a keyword argument named ``param_name``.
    """
    import pandas as pd

    rows = []
    for v in values:
        try:
            res = cycle_builder(**{param_name: v})
            rows.append({param_name: float(v),
                         objective: float(getattr(res, objective)),
                         "net_work": float(res.net_work)})
        except Exception:
            rows.append({param_name: float(v), objective: np.nan, "net_work": np.nan})
    return pd.DataFrame(rows)