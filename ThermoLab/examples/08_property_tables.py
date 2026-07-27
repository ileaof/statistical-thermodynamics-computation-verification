"""Example 08 — Property tables and fast multidimensional interpolation.

ThermoLab's :mod:`thermolab.tables` module builds gridded property data over a
(T, P) mesh and exposes a SciPy ``RegularGridInterpolator``-backed lookup
function suitable as a CFD lookup table.

* :class:`PropertyTable` — long-format DataFrame of every property over a
  ``(T, P)`` grid, plus ``.interpolate()`` returning a callable ``f(T, P)``.
* :class:`SaturationTable` — saturated liquid/vapor properties along the
  saturation curve of a pure fluid.

Run with::

    python examples/08_property_tables.py
"""
from __future__ import annotations

import numpy as np

from thermolab import Gas
from thermolab.tables import PropertyTable, SaturationTable


def main() -> None:
    water = Gas("H2O")

    # --- Gridded property table over (T, P) -------------------------------
    # (min, max, n) triples are accepted in lieu of explicit point lists.
    tab = PropertyTable(
        water,
        T_range=(300.0, 700.0, 15),
        P_range=(1e4, 1e7, 12),
    )
    print(f"PropertyTable: {len(tab.df)} rows, "
          f"{tab.df['rho'].notna().sum()} converged states")
    cols = ["T", "P", "rho", "h", "s", "cp", "gamma", "mu", "k", "prandtl"]
    print(tab.df[cols].head(6).to_string(
        float_format=lambda x: f"{x:.4g}", index=False))

    # --- Fast interpolated lookup vs. a real flash -------------------------
    # Pick a query point well inside a single-phase region (superheated
    # vapour): grid interpolation is meaningless across the two-phase dome,
    # where liquid and vapor points sit side by side in the same mesh.
    f = tab.interpolate()
    T_q, P_q = 600.0, 5.0e5
    interp = f(T_q, P_q)
    direct = water.state(T=T_q, P=P_q)
    print(f"\nAt T={T_q:g} K, P={P_q:g} Pa (superheated vapour):")
    for prop in ("rho", "h", "cp", "gamma"):
        print(f"  {prop:>6s}: interp={interp[prop]:.4g}  "
              f"flash={getattr(direct, prop):.4g}")

    # --- Saturation table along the vapor dome -----------------------------
    sat = SaturationTable(water, T_range=(300.0, 640.0, 18))
    print("\nSaturationTable (head):")
    print(sat.df[["T", "P", "rho_f", "rho_g", "h_f", "h_g"]].head(6).to_string(
        float_format=lambda x: f"{x:.4g}", index=False))

    # Latent heat of vaporization h_fg = h_g - h_f, falls with T -> 0 at Tc.
    hfg = sat.df["h_g"] - sat.df["h_f"]
    print(f"\nh_fg at T={sat.df['T'].iloc[0]:.1f} K: {hfg.iloc[0]:.4g} J/kg")
    print(f"h_fg at T={sat.df['T'].iloc[-1]:.1f} K: {hfg.iloc[-1]:.4g} J/kg "
          f"(-> 0 near the critical point)")


if __name__ == "__main__":
    main()