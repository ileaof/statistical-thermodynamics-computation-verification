"""Example 05 — Mollier (h-s) diagram for water with isobars and saturation.

Run with::

    python examples/05_mollier.py
    python examples/05_mollier.py --save mollier_water.png
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
    water = Gas("H2O")

    ax = P.plot_mollier(water, isobars=[1e4, 1e5, 5e5, 1e6, 5e6], T_range=(300, 700))
    P.plot_saturation(water, T_range=(300, 640), diagram="mollier", ax=ax)
    ax.set_xlabel("s  [J/(kg.K)]")
    ax.set_ylabel("h  [J/kg]")
    ax.set_title("Water: Mollier h-s diagram")

    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        ax.figure.savefig(out, dpi=140, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()