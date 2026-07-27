"""Example 16 — Solid phase (dry ice): melting curve and solid properties.

ThermoLab's high-level API wraps the fluid phases (vapour / liquid / two-phase);
the solid phase is reached through ThermoPack's ``thermo`` engine directly. This
example builds a Peng-Robinson + solid-correlation engine for CO2 and computes
the melting line P_melt(T), the solid (dry-ice) density, enthalpy and heat
capacity, and the enthalpy of fusion at the triple point.

Reference-state caveat: solid enthalpies from the thermo engine use a different
zero from ThermoLab's fluid enthalpies, so *absolute* enthalpies cannot be
compared across the two. The fusion enthalpy below is therefore computed from
the *same* engine (liquid minus solid), where the reference cancels. Densities
are reference-independent and may be compared freely.

Run with::

    python examples/16_solids.py
    python examples/16_solids.py --save solids.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermopack import thermo

from thermolab import Gas

# CO2 molar mass [kg/mol]; ThermoPack's compmoleweight returns g/mol (1-based).
M_CO2 = 0.04401


def main() -> None:
    # --- Build a PR fluid engine with the solid phase attached ------------
    eng = thermo.thermo()
    eng.init_thermo("PR", "Classic", "Classic", "CO2", 1)
    eng.init_solid("CO2")
    x = np.array([1.0])

    # --- Melting line P_melt(T) from the correlation ----------------------
    Tm, Pm = eng.melting_pressure_correlation(1, 240.0, 80)
    print("CO2 melting line (solid <-> liquid):")
    print(f"  triple point end : T = {Tm[0]:.2f} K,  P = {Pm[0]/1e5:.3f} bar")
    print(f"  upper end        : T = {Tm[-1]:.2f} K, P = {Pm[-1]/1e6:.2f} MPa")
    print("  (literature: triple point 216.55 K, 5.18 bar; dry ice sublimes at "
          "194.7 K, 1 atm)")

    # --- Solid density, enthalpy, and Cp vs T at 1 bar --------------------
    Ts = np.linspace(180.0, 214.0, 8)
    rho_s, h_s = [], []
    for T in Ts:
        v_s = eng.solid_volume(float(T), 1e5, x)[0]
        rho_s.append(M_CO2 / v_s)
        h_s.append(eng.solid_enthalpy(float(T), 1e5, x)[0])
    rho_s = np.array(rho_s)
    h_s = np.array(h_s)
    # Cp_solid by central finite difference of the solid enthalpy
    Cp_s = np.array([
        (eng.solid_enthalpy(float(T + 1), 1e5, x)[0]
         - eng.solid_enthalpy(float(T - 1), 1e5, x)[0]) / 2.0
        for T in Ts[1:-1]
    ])
    print(f"\nSolid CO2 @ 1 bar:")
    print(f"  rho_solid = {rho_s[0]:.1f} ... {rho_s[-1]:.1f} kg/m3 "
          f"(dry ice ~ 1560 kg/m3)")
    print(f"  Cp_solid  ~ {Cp_s.mean():.1f} J/(mol.K)  (crystal CO2 ~ 38-54)")

    # --- Enthalpy of fusion at the triple point (SAME engine) -------------
    Tt, Pt = 216.55, 5.18e5
    h_solid_tp = eng.solid_enthalpy(Tt, Pt, x)[0]
    h_liq_tp = eng.enthalpy(Tt, Pt, x, 1)[0]   # phase flag LIQPH = 1
    dh_fus = h_liq_tp - h_solid_tp
    # Liquid density at the triple point (reference-independent) via ThermoLab
    rho_liq_tp = Gas("CO2").state(T=Tt, P=Pt, phase="liquid").rho
    print(f"\nAt the triple point (T={Tt} K, P={Pt/1e5:g} bar):")
    print(f"  dh_fus (liquid - solid) = {dh_fus/1e3:6.2f} kJ/mol  "
          f"(literature ~ 8.3-9.0)")
    print(f"  rho_solid = {M_CO2/eng.solid_volume(Tt, Pt, x)[0]:.1f} kg/m3,  "
          f"rho_liquid = {rho_liq_tp:.1f} kg/m3  (solid denser, as expected)")

    # --- Plot: melting curve + solid density -----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogy(Tm, Pm / 1e5, "-C0", label="melting line P_melt(T)")
    ax1.scatter([Tt], [Pt / 1e5], color="k", zorder=5, label="triple point")
    ax1.axhline(1.01325, color="0.6", ls=":", lw=1.0, label="1 atm")
    ax1.set_xlabel("T [K]")
    ax1.set_ylabel("P [bar]")
    ax1.set_title("CO2 solid-liquid melting curve")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(Ts, rho_s, "-C1", label="solid CO2 @ 1 bar")
    ax2.axhline(1560.0, color="0.6", ls="--", lw=1.0, label="dry ice (lit.)")
    ax2.set_xlabel("T [K]")
    ax2.set_ylabel("density [kg/m3]")
    ax2.set_title("Dry-ice density vs temperature")
    ax2.legend(fontsize=8)
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