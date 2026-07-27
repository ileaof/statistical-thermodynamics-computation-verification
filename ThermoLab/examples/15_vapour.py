"""Example 15 — Vapour-phase properties and the vapour-pressure curve.

The vapour-pressure curve P_sat(T) of a pure fluid, the Clausius-Clapeyron
linearization (ln P vs 1/T -> slope -dh_vap/R), the latent heat of
vaporization along the dome (falling to zero at the critical point), and a
superheated-vapour isobar showing how h becomes T-only (ideal-gas-like) far
from the dome.

As in Example 14, ``saturation_pressure`` is sampled only up to ~0.99 Tc
(``bubble_pressure`` fails a few K below Tc) and saturated liquid/vapour are
taken at a pressure nudged just off P_sat so the flash returns the intended
root on the saturation boundary.

Run with::

    python examples/15_vapour.py
    python examples/15_vapour.py --save vapour.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermolab import Gas

R = 8.314462618  # J/(mol.K)
T_DOME_MAX = 640.0


def main() -> None:
    water = Gas("H2O")
    M = water.molar_mass
    Tc = water.critical_temperature()

    # --- Vapour-pressure curve -------------------------------------------
    Ts = np.linspace(300.0, T_DOME_MAX, 25)
    Psat = np.array([water.saturation_pressure(float(T)) for T in Ts])
    print(f"Water vapour-pressure curve ({len(Ts)} pts, 300 K -> {T_DOME_MAX:g} K):")
    print(f"  P_sat(300 K)  = {Psat[0]:.3g} Pa  ({Psat[0]/1e3:.2f} kPa)")
    print(f"  P_sat(373 K) = {water.saturation_pressure(373.15):.4g} Pa "
          f"(~1 atm = {1.01325e5:.4g})")
    print(f"  P_sat({T_DOME_MAX:g} K) = {Psat[-1]:.4g} Pa  (-> Pc at Tc = {Tc:.1f} K)")

    # --- Clausius-Clapeyron: ln(P) = A - B/T,  slope ~ -dh_vap/R ------------
    # Fit well below Tc (where the relation is linear) for an accurate dh_vap.
    fit_mask = Ts <= 480.0
    A, B = np.polyfit(1.0 / Ts[fit_mask], np.log(Psat[fit_mask]), 1)
    dh_vap_cc = -A * R  # J/mol
    # Direct latent heat at a mid-curve temperature (nudged off P_sat)
    Tmid = 420.0
    Pm = water.saturation_pressure(Tmid)
    h_f = water.state(T=Tmid, P=Pm * 1.001, phase="liquid").h
    h_g = water.state(T=Tmid, P=Pm * 0.999, phase="vapor").h
    dh_vap_direct = (h_g - h_f) * M  # J/mol
    print(f"\nClausius-Clapeyron slope (300-480 K) -> dh_vap = {dh_vap_cc/1e3:7.2f} kJ/mol")
    print(f"Direct h_g - h_f @ {Tmid:g} K            -> dh_vap = {dh_vap_direct/1e3:7.2f} kJ/mol")

    # --- Latent heat along the dome -> 0 at the critical point ------------
    Td = np.linspace(300.0, T_DOME_MAX, 20)
    hfg = []
    for T in Td:
        Ps = water.saturation_pressure(float(T))
        hf = water.state(T=float(T), P=Ps * 1.001, phase="liquid").h
        hg = water.state(T=float(T), P=Ps * 0.999, phase="vapor").h
        hfg.append((hg - hf) * M / 1e3)  # kJ/mol
    hfg = np.array(hfg)
    print(f"\nLatent heat h_fg: {hfg[0]:.1f} kJ/mol at {Td[0]:.0f} K -> "
          f"{hfg[-1]:.1f} kJ/mol at {Td[-1]:.0f} K (-> 0 at Tc)")

    # --- Superheated vapour isobar: h depends on T only far from the dome --
    P_iso = 1e5
    Tsat_iso = water.saturation_temperature(P_iso)
    Ts_sh = np.array([Tsat_iso + 5, Tsat_iso + 50, 500.0, 700.0, 1000.0])
    print(f"\nSuperheated vapour isobar at P = {P_iso/1e5:g} bar "
          f"(Tsat = {Tsat_iso:.1f} K):")
    print(f"{'T [K]':>8} {'rho [kg/m3]':>11} {'h [J/kg]':>11} {'Z':>7}")
    for T in Ts_sh:
        st = water.state(T=float(T), P=P_iso, phase="vapor")
        print(f"{T:8.1f} {st.rho:11.4g} {st.h:11.4g} {st.Z:7.3f}")

    # --- Plot: Clapeyron line + latent-heat curve -------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(1.0 / Ts, np.log(Psat), "oC0", label="P_sat(T)")
    ax1.plot(1.0 / Ts, A / Ts + B, "--C3", lw=1.2,
             label=f"Clapeyron fit (300-480 K)\ndh_vap={dh_vap_cc/1e3:.1f} kJ/mol")
    ax1.set_xlabel("1/T  [1/K]")
    ax1.set_ylabel("ln P_sat  [ln Pa]")
    ax1.set_title("Vapour-pressure curve (Clausius-Clapeyron)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(Td, hfg, "-C1")
    ax2.axvline(Tc, color="0.6", ls="--", lw=1.0, label=f"Tc = {Tc:.1f} K")
    ax2.set_xlabel("T [K]")
    ax2.set_ylabel("h_fg  [kJ/mol]")
    ax2.set_title("Latent heat of vaporization -> 0 at Tc")
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