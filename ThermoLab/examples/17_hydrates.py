"""Example 17 — Gas hydrates: equilibrium curve and inhibitor sizing (Hammerschmidt).

ThermoPack's Python wrapper in this build does not expose the van der Waals-
Platteeuw hydrate-equilibrium model (no hydrate binding in the installed
source), so this example uses the engineering correlations a process engineer
reaches for instead:

* a Clausius-Clapeyron fit of the methane-hydrate P-T equilibrium on the
  water/hydrate/vapour branch (273.15-286 K), anchored to tabulated data
  (Sloan & Koh, *Clathrate Hydrates of Natural Gases*), and
* the Hammerschmidt equation for the depression of the hydrate-equilibrium
  temperature by methanol (MeOH) or ethylene glycol (MEG) in the water phase.

The example finds the hydrate-equilibrium temperature at a pipeline pressure,
sizes the inhibitor concentration needed for a target subcooling margin, and
compares MeOH vs MEG.

Run with::

    python examples/17_hydrates.py
    python examples/17_hydrates.py --save hydrates.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Methane-hydrate equilibrium (Lw-H-V), illustrative Clausius-Clapeyron fit:
#   ln P[Pa] = A - B/T,   valid ~273.15-286 K.
# Anchored to methane-hydrate data: 2.63 MPa @ 273.15 K, 7.0 MPa @ 285 K.
A_H, B_H = 38.459, 6468.0


def P_hydrate(T: float) -> float:
    """Hydrate-equilibrium pressure [Pa] at temperature T [K]."""
    return float(np.exp(A_H - B_H / T))


def T_hydrate(P: float) -> float:
    """Hydrate-equilibrium temperature [K] at pressure P [Pa]."""
    return B_H / (A_H - np.log(P))


# Hammerschmidt:  dT = K * W / (M * (100 - W))   [K]
#   W = inhibitor weight percent in the aqueous phase
#   M = inhibitor molar mass [g/mol]
#   K = 1297 K*kg/mol
K_H = 1297.0


def suppression(W: float, M: float) -> float:
    """Hydrate-equilibrium temperature depression [K] for inhibitor wt% W."""
    return K_H * W / (M * (100.0 - W))


def wt_percent_needed(dT: float, M: float) -> float:
    """Invert Hammerschmidt: inhibitor wt% required to suppress by dT [K]."""
    return 100.0 * dT * M / (K_H + dT * M)


def main() -> None:
    print("Methane-hydrate equilibrium (illustrative CC fit, 273.15-286 K):")
    for T in [273.15, 278.15, 283.15, 285.0]:
        print(f"  T={T:6.2f} K  ->  P_eq = {P_hydrate(T)/1e6:6.3f} MPa")

    # --- Pipeline scenario -----------------------------------------------
    P_op = 5.0e6                     # operating pressure, 50 bar
    T_op = 278.15                    # ambient temperature, 5 C
    T_eq = T_hydrate(P_op)
    print(f"\nPipeline: P_op = {P_op/1e6:g} MPa,  T_op = {T_op - 273.15:.0f} C")
    print(f"  Hydrate-equilibrium T at {P_op/1e6:g} MPa = {T_eq:.2f} K "
          f"({T_eq - 273.15:.2f} C)")
    print(f"  T_op < T_eq  ->  hydrate would form; inhibition required.")

    margin = 3.0                      # K subcooling safety margin
    dT_req = max(T_eq - T_op, 0.0) + margin
    print(f"  Required suppression = (T_eq - T_op) + {margin:g} K margin = "
          f"{dT_req:.2f} K")

    # --- Inhibitor sizing: MeOH vs MEG -----------------------------------
    print(f"\nInhibitor required in the water phase (for {dT_req:.2f} K suppression):")
    print(f"  {'inhibitor':>9} {'M [g/mol]':>10} {'wt%':>7} {'check dT [K]':>13}")
    for name, M in [("MeOH", 32.04), ("MEG", 62.07)]:
        W = wt_percent_needed(dT_req, M)
        check = suppression(W, M)
        print(f"  {name:>9} {M:10.2f} {W:7.2f} {check:13.2f}")
    print("\n(MeOH is more effective per unit mass; MEG needs a higher wt%.)")

    # --- Plot: hydrate curve + inhibited curves -------------------------
    Ts = np.linspace(273.15, 286.0, 60)
    Ps = [P_hydrate(T) for T in Ts]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(Ts - 273.15, np.array(Ps) / 1e6, "-C0", lw=2,
                label="hydrate equilibrium (no inhibitor)")
    # Inhibited curves: T_eq shifted down by the Hammerschmidt suppression
    for name, M, color, W in [("MeOH", 32.04, "C2", 20.0),
                             ("MEG", 62.07, "C3", 30.0)]:
        dT = suppression(W, M)
        ax.semilogy(Ts - 273.15 - dT, np.array(Ps) / 1e6, "--", color=color,
                    lw=1.3, label=f"{name} {W:g} wt%  (dT={dT:.1f} K)")
    ax.axhline(P_op / 1e6, color="0.6", ls=":", lw=1.0, label=f"P_op = {P_op/1e6:g} MPa")
    ax.axvline(T_op - 273.15, color="0.6", ls=":", lw=1.0,
               label=f"T_op = {T_op - 273.15:.0f} C")
    ax.set_xlabel("T  [C]")
    ax.set_ylabel("P  [MPa]")
    ax.set_title("Methane-hydrate equilibrium and inhibitor suppression")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()