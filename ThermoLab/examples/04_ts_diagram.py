"""Example 04 — T-s diagram for air with isobars and isochores.

Run with::

    python examples/04_ts_diagram.py   # shows the figure on screen
    python examples/04_ts_diagram.py --save ts_air.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermolab import Gas
from thermolab import plotting as P


def main() -> None:
    air = Gas("Air")

    ax = P.plot_ts(air, isobars=[1e5, 5e5, 1e6, 2e6], T_range=(250, 1500))
    P.plot_isochores(air, [0.5, 1.0, 2.0], T_range=(250, 1500), ax=ax)
    ax.set_title("Air: T-s diagram (isobars solid, isochores dashed)")
    ax.legend(loc="best", fontsize=7)

    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        ax.figure.savefig(out, dpi=140, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()