"""Example 03 — CFD interface: scalar snapshots and vectorized grid evaluation.

ThermoLab exposes two CFD-oriented helpers in ``thermolab.cfd``:

* ``CFDScalars.from_state(state)`` — a lightweight snapshot of the scalars a
  finite-volume / finite-difference solver needs at a cell (rho, cp, cv, gamma,
  mu, k, sound_speed, Pr, alpha, T, P, phase). Zero per-call overhead once the
  state is resolved.
* ``evaluate_grid(fluid, T, P)`` — vectorized evaluation over arrays of T and P
  (e.g. a mesh slice), returning a pandas DataFrame.

Run with::

    python examples/03_cfd_interface.py
"""
from __future__ import annotations

import numpy as np

from thermolab import Gas
from thermolab.cfd import CFDScalars, bulk_properties, evaluate_grid


def main() -> None:
    air = Gas("Air")

    # --- Single-point scalar snapshot (the per-cell CFD surface) ----------
    st = air.state(T=800.0, P=5e5)
    scalars = CFDScalars.from_state(st)
    print("CFDScalars snapshot:")
    print(scalars.to_dict())

    # --- Vectorized evaluation over a T-P grid ---------------------------
    T = np.linspace(300.0, 1500.0, 6)
    P = np.array([1e5, 5e5, 1e6])
    TT, PP = np.meshgrid(T, P)
    df = evaluate_grid(air, TT.ravel(), PP.ravel())
    print("\nGrid evaluation (head):")
    cols = ["T", "P", "rho", "cp", "gamma", "mu", "k", "Pr"]
    # %.4g keeps 4 significant figures, so small values like mu ~ 1.8e-5 Pa.s
    # print as 1.847e-05 instead of being truncated to 0.0000 by a fixed scale.
    print(df[cols].head(8).to_string(float_format=lambda x: f"{x:.4g}"))

    # --- Bulk properties over a list of resolved states ------------------
    states = [air.state(T=float(t), P=2e5) for t in T]
    bulk = bulk_properties(states)
    print(f"\nBulk (mean over T sweep at P=2e5): "
          f"rho={bulk['rho'].mean():.3f} kg/m^3, "
          f"gamma={bulk['gamma'].mean():.4f}")


if __name__ == "__main__":
    main()