"""Example 12 — Transport & acoustic properties across a temperature sweep.

ThermoPack is thermodynamic-only; ThermoLab adds gas-phase transport properties
(viscosity, thermal conductivity) via engineering correlations and derives the
Prandtl number, thermal diffusivity, and speed of sound from the EOS. This
example sweeps temperature at fixed pressure for Air and CO2, prints a compact
table, and plots viscosity and Prandtl number vs. T.

Run with::

    python examples/12_transport.py
    python examples/12_transport.py --save transport.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermolab import Gas


def sweep(fluid, Ts, P=1e5):
    rows = []
    for T in Ts:
        st = fluid.state(T=float(T), P=P)
        rows.append((T, st.mu, st.k, st.prandtl, st.gamma, st.sound_speed))
    return np.array(rows, dtype=float)


def main() -> None:
    air = Gas("Air")
    co2 = Gas("CO2")
    Ts = np.linspace(300.0, 1500.0, 9)

    a = sweep(air, Ts)
    c = sweep(co2, Ts)
    print("Transport sweep at P = 1 bar (Air | CO2):")
    print(f"{'T [K]':>7} {'mu_air':>9} {'mu_co2':>9} "
          f"{'k_air':>9} {'k_co2':>9} {'Pr_air':>7} {'Pr_co2':>7}")
    for i, T in enumerate(Ts):
        print(f"{T:7.0f} {a[i,1]*1e6:9.3f} {c[i,1]*1e6:9.3f} "
              f"{a[i,2]:9.4g} {c[i,2]:9.4g} {a[i,3]:7.3f} {c[i,3]:7.3f}")
    print("\n(mu in uPa.s, k in W/(m.K))")

    # --- Plot viscosity and Prandtl number vs. T --------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(a[:, 0], a[:, 1] * 1e6, "-o", label="Air")
    ax1.plot(c[:, 0], c[:, 1] * 1e6, "-s", label="CO2")
    ax1.set_xlabel("T [K]")
    ax1.set_ylabel(r"dynamic viscosity $\mu$  [$\mu$Pa.s]")
    ax1.set_title("Viscosity vs. temperature (P = 1 bar)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(a[:, 0], a[:, 3], "-o", label="Air")
    ax2.plot(c[:, 0], c[:, 3], "-s", label="CO2")
    ax2.set_xlabel("T [K]")
    ax2.set_ylabel("Prandtl number  Pr  [-]")
    ax2.set_title("Prandtl number vs. temperature (P = 1 bar)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()