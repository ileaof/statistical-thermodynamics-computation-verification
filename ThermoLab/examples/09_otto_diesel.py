"""Example 09 — Otto vs Diesel air-standard cycles.

Both are spark/compression-ignition reciprocating-engine cycles built from
real-fluid ThermoLab states (Air). The Otto cycle adds heat at constant volume;
the Diesel cycle adds it at constant pressure. This example solves a baseline
case for each, sweeps the compression ratio, and overlays both on a T-s diagram.

Run with::

    python examples/09_otto_diesel.py
    python examples/09_otto_diesel.py --save otto_diesel.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermolab import cycles as C
from thermolab import optimization as opt


def main() -> None:
    # --- Baseline cases ---------------------------------------------------
    otto = C.otto(compression_ratio=8.0, T3=2500.0)
    diesel = C.diesel(compression_ratio=18.0, cutoff_ratio=2.0)
    print(otto)
    print(diesel)

    # --- Compression-ratio sweep: efficiency vs. specific work -----------
    rs = [6, 8, 10, 12, 14, 16, 18, 20]
    df_o = opt.sweep(lambda compression_ratio: C.otto(compression_ratio=compression_ratio,
                                                     T3=2500.0),
                     "compression_ratio", rs)
    df_d = opt.sweep(lambda compression_ratio: C.diesel(compression_ratio=compression_ratio,
                                                        cutoff_ratio=2.0),
                     "compression_ratio", rs)
    print("\nCompression-ratio sweep (eta / net_work [J/kg]):")
    print("  r   Otto_eta Diesel_eta  Otto_w   Diesel_w")
    for r, eo, ed, wo, wd in zip(rs, df_o["eta"], df_d["eta"],
                                 df_o["net_work"], df_d["net_work"]):
        print(f"  {r:<3d}  {eo:.4f}    {ed:.4f}     "
              f"{wo:8.3g}  {wd:8.3g}")

    # --- T-s diagram overlay ---------------------------------------------
    ax = otto.plot(diagram="ts")
    diesel.plot(diagram="ts", ax=ax)
    ax.set_title("Otto (red) vs Diesel cycle on a T-s diagram")
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        ax.figure.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()