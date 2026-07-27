"""Example 14 — Liquid-phase properties: saturated, compressed, and the dome.

ThermoLab resolves liquid states explicitly via ``phase="liquid"``. This
example examines compressed-liquid water at fixed temperature (isothermal
compressibility and thermal expansion, cross-checked against the EOS
derivatives), the saturated-liquid branch of the vapour-pressure curve, and
how the liquid and vapour densities converge at the critical point along the
saturation dome.

Two practical gotchas surface here:

* At *exactly* P = P_sat the flash can return either root, so saturated liquid
  / vapour are taken at a pressure nudged just off P_sat (1.001 / 0.999 x).
* ``bubble_pressure`` (which backs ``saturation_pressure``) fails within a few
  K of the critical point, so the dome is sampled only up to ~0.99 Tc.

Transport properties (mu, k) are gas-phase correlations and are *not* accurate
for liquids; their warning is suppressed here.

Run with::

    python examples/14_liquids.py
    python examples/14_liquids.py --save liquids.png
"""
from __future__ import annotations

import sys
import warnings

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermolab import Gas

warnings.filterwarnings("ignore", message="Transport properties are gas-phase*")

T_DOME_MAX = 640.0  # keep a safe margin below Tc (bubble_pressure fails ~644 K)


def main() -> None:
    water = Gas("H2O")
    Tc = water.critical_temperature()
    print(f"Water  Tc = {Tc:.2f} K,  Pc = {water.critical_pressure():.4g} Pa")

    # --- Compressed liquid at T = 300 K, swept over pressure --------------
    T0 = 300.0
    print(f"\nCompressed liquid water at T = {T0:.1f} K")
    print(f"{'P [bar]':>9} {'rho [kg/m3]':>11} {'h [J/kg]':>11} {'v [m3/kg]':>11}")
    for P in [1e5, 1e6, 1e7, 5e7]:
        st = water.state(T=T0, P=P, phase="liquid")
        print(f"{P/1e5:9.3f} {st.rho:11.3f} {st.h:11.4g} {st.v:11.3e}")

    # --- Isothermal compressibility and thermal expansion (numeric vs EOS) -
    P0 = 1e5
    dP = 1e4
    st0 = water.state(T=T0, P=P0, phase="liquid")
    v0, rho0 = st0.v, st0.rho
    vm = water.state(T=T0, P=P0 - dP, phase="liquid").v
    vp = water.state(T=T0, P=P0 + dP, phase="liquid").v
    kappa_num = -(1.0 / v0) * (vp - vm) / (2 * dP)

    # ThermoPack reports beta = (1/rho)(d rho/d T)|P  (negative for a normal
    # liquid), so the numeric check uses density to match its sign convention.
    dT = 2.0
    rho_lo = water.state(T=T0 - dT, P=P0, phase="liquid").rho
    rho_hi = water.state(T=T0 + dT, P=P0, phase="liquid").rho
    beta_num = (1.0 / rho0) * (rho_hi - rho_lo) / (2 * dT)

    print(f"\nAt T={T0} K, P={P0/1e5:g} bar (compressed liquid):")
    print(f"  kappa_T  (isothermal compressibility)  numeric={kappa_num:.3e}  "
          f"EOS={st0.kappa_t:.3e}  1/Pa")
    print(f"  beta     (thermal expansion, dln(rho)/dT) numeric={beta_num:.3e}  "
          f"EOS={st0.beta_thermal_expansion:.3e}  1/K")

    # --- Saturated liquid / vapour densities along the dome ---------------
    # Nudge P off P_sat so each root is the *stable* phase (override alone is
    # unreliable right on the saturation boundary).
    Ts = np.linspace(300.0, T_DOME_MAX, 28)
    rho_f, rho_g = [], []
    for T in Ts:
        Psat = water.saturation_pressure(float(T))
        rho_f.append(water.state(T=float(T), P=Psat * 1.001, phase="liquid").rho)
        rho_g.append(water.state(T=float(T), P=Psat * 0.999, phase="vapor").rho)
    rho_f = np.array(rho_f)
    rho_g = np.array(rho_g)
    print(f"\nSaturated densities:  rho_f(300 K)={rho_f[0]:.2f}, "
          f"rho_g(300 K)={rho_g[0]:.4g} kg/m3")
    print(f"                      rho_f({Ts[-1]:.0f} K)={rho_f[-1]:.1f}, "
          f"rho_g({Ts[-1]:.0f} K)={rho_g[-1]:.1f} kg/m3  (-> merge at Tc)")

    # --- Plot: liquid & vapour density branches meeting at the critical pt -
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ts, rho_f, "-C0", label="saturated liquid  rho_f")
    ax.plot(Ts, rho_g, "-C1", label="saturated vapour  rho_g")
    ax.axvline(Tc, color="0.6", ls="--", lw=1.0, label=f"Tc = {Tc:.1f} K")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("density  [kg/m3]")
    ax.set_title("Water: saturated liquid/vapour density along the dome")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()