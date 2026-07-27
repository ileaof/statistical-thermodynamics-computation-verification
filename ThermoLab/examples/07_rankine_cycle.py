"""Example 07 — Rankine cycle with superheat, water as the working fluid.

Solves a regenerative-free Rankine cycle with superheated steam, prints the
efficiency / back-work-ratio / net work, and plots it on a T-s diagram
(saturation dome included).

Run with::

    python examples/07_rankine_cycle.py
    python examples/07_rankine_cycle.py --save rankine.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermolab import Gas
from thermolab import cycles as C
from thermolab import plotting as P


def main() -> None:
    water = Gas("H2O")
    res = C.rankine(P_boiler=8e6, P_condenser=1e4, T_superheat=773.0)
    print(res)
    print(f"  points : {[p.label for p in res.points]}")
    for p in res.points:
        print(f"    {p.label}: T={p.state.T:8.2f} K  P={p.state.P:.3e} Pa  "
              f"h={p.state.h:.4g} J/kg  s={p.state.s:.4g}")

    ax = res.plot(diagram="ts")
    P.plot_saturation(water, T_range=(300, 640), diagram="ts", ax=ax)
    ax.set_title("Rankine cycle (superheated) on a T-s diagram")

    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        ax.figure.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()