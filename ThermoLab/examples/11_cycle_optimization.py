"""Example 11 — Cycle optimization: maximize Brayton efficiency / net work.

Uses :func:`thermolab.optimization.optimize_cycle` to find the pressure ratio
that maximizes the thermal efficiency of a Brayton cycle at fixed peak
temperature (local L-BFGS-B search), then a global ``differential_evolution``
search over ``(pressure_ratio, T3)`` that maximizes net specific work. A sweep
over the peak temperature shows the efficiency / work trade-off.

Run with::

    python examples/11_cycle_optimization.py
"""
from __future__ import annotations

import numpy as np

from thermolab import cycles as C
from thermolab import optimization as opt


def main() -> None:
    # --- Local search: optimal pressure ratio for max efficiency ----------
    # NOTE: optimize_cycle calls ``cycle_builder(*x)``, i.e. it *unpacks* the
    # parameter vector, so the builder receives scalar args (not the array).
    res = opt.optimize_cycle(
        lambda rp: C.brayton(pressure_ratio=rp, T3=1400.0),
        bounds=[(4.0, 40.0)], objective="eta", x0=[10.0],
    )
    rp_opt = res.x[0]
    eta_opt = -res.fun
    print("L-BFGS-B  (maximize eta, T3=1400 K):")
    print(f"  optimal pressure_ratio = {rp_opt:.3f}")
    print(f"  max eta                = {eta_opt:.4%}")

    # --- Global search: max net work over (pressure_ratio, T3) ------------
    de = opt.optimize_cycle(
        lambda rp, T3: C.brayton(pressure_ratio=rp, T3=T3),
        bounds=[(4.0, 40.0), (1200.0, 1800.0)],
        objective="net_work", method="differential_evolution",
    )
    print("\nDifferential evolution (maximize net_work):")
    print(f"  pressure_ratio = {de.x[0]:.3f},  T3 = {de.x[1]:.2f} K")
    print(f"  max net_work   = {-de.fun:.4g} J/kg")

    # --- Sweep: efficiency / net-work trade-off vs. T3 --------------------
    T3s = [1200, 1300, 1400, 1500, 1600, 1700, 1800]
    df = opt.sweep(lambda T3: C.brayton(pressure_ratio=rp_opt, T3=T3),
                   "T3", T3s)
    print("\nSweep at the optimal pressure ratio (varying T3):")
    print(df[["T3", "eta", "net_work"]].round(4).to_string(index=False))

    best = df.loc[df["net_work"].idxmax()]
    print(f"\nHighest net_work in sweep: T3={best['T3']:.0f} K -> "
          f"{best['net_work']:.4g} J/kg (eta={best['eta']:.4%})")


if __name__ == "__main__":
    main()