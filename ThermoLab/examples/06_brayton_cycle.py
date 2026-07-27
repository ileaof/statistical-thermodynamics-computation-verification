"""Example 06 — Brayton (Joule) cycle with air as the working fluid.

Solves an air-standard Brayton cycle, prints the efficiency / back-work-ratio /
net work, and plots it on a T-s diagram. A short optimization sweep over the
pressure ratio is included.

Run with::

    python examples/06_brayton_cycle.py
    python examples/06_brayton_cycle.py --save brayton.png
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
    res = C.brayton(pressure_ratio=12, T3=1400)
    print(res)
    print(f"  points : {[p.label for p in res.points]}")

    # Sweep pressure ratio to see the efficiency / specific-work trade-off.
    df = opt.sweep(
        lambda pressure_ratio: C.brayton(pressure_ratio=pressure_ratio, T3=1400),
        "pressure_ratio", [4, 8, 12, 16, 20, 24],
    )
    print("\nPressure-ratio sweep:")
    print(df[["pressure_ratio", "eta", "net_work"]].round(4))

    ax = res.plot(diagram="ts")
    ax.set_title("Brayton cycle on a T-s diagram")
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        ax.figure.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()