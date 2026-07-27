"""Example 10 — Vapor-compression refrigeration cycle (R134a).

Solves an ideal and a non-ideal vapor-compression refrigeration cycle, prints
the COP and per-state properties, and overlays the cycle on a P-h diagram with
the saturation dome and the evaporator/condenser isotherms.

Run with::

    python examples/10_refrigeration.py
    python examples/10_refrigeration.py --save refrigeration.png
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
    r134a = Gas("R134a")

    ideal = C.refrigeration(T_evap=263.15, T_cond=313.15, superheat=5.0)
    real = C.refrigeration(T_evap=263.15, T_cond=313.15, superheat=5.0,
                           eta_compressor=0.75)
    print(ideal)
    print(real)
    print(f"\nCOP drop from ideal -> eta_c=0.75: "
          f"{ideal.cop:.3f} -> {real.cop:.3f}")

    # --- P-h diagram with saturation dome + cycle overlay -----------------
    ax = P.plot_ph(r134a, isotherms=[263.15, 313.15], T_range=(240, 360))
    real.plot(diagram="ph", ax=ax)
    ax.set_title("Vapor-compression refrigeration (R134a) on a P-h diagram")

    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        ax.figure.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()