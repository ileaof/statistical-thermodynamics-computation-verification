#!/usr/bin/env python3
"""
Example 7.2 (direct numerical) -- The second virial coefficient of a Lennard-Jones
gas by numerical integration.

The second virial coefficient measures the leading correction to the ideal-gas law,
p V / N k_B T = 1 + B(T)/v + ...  It is an integral over the pair potential,

    B(T) = -2 pi N_A integral_0^inf [ exp(-u(r)/k_B T) - 1 ] r^2 dr,

with the Lennard-Jones potential u(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ].  In
reduced units T* = k_B T/eps, r* = r/sigma, and with b0 = (2/3) pi N_A sigma^3,

    B*(T*) = B/b0 = -3 integral_0^inf [ exp(-(4/T*)(r*^-12 - r*^-6)) - 1 ] r*^2 dr*.

This script integrates B*(T*) numerically, locates the Boyle temperature where
B* = 0 (known value T*_B = 3.418), checks the hard-sphere and high-T limits, and
converts to a real gas (argon) for comparison with experiment.  numpy/scipy only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

NA = 6.02214076e23


def B_star(Tstar):
    """Reduced second virial coefficient of the Lennard-Jones fluid."""
    def integrand(r):
        u = 4.0 * (r ** -12 - r ** -6) / Tstar        # u/kT in reduced units
        return (np.exp(-u) - 1.0) * r ** 2
    # integrate from a small core (integrand -> -r^2, finite) to a cutoff
    val, _ = quad(integrand, 1e-4, 12.0, limit=400)
    return -3.0 * val


def main():
    print("=" * 70)
    print("Example 7.2  Second virial coefficient of a Lennard-Jones gas")
    print("=" * 70)

    print(f"\n{'T*':>6} {'B*(T*)':>12}")
    for Ts in [0.7, 1.0, 1.5, 2.0, 3.0, 3.418, 5.0, 10.0]:
        print(f"{Ts:>6.3f} {B_star(Ts):>12.5f}")

    # --- Boyle temperature: B*(T*) = 0 ---------------------------------
    T_boyle = brentq(B_star, 2.5, 4.5, xtol=1e-8)
    print(f"\nBoyle temperature  T*_B = {T_boyle:.4f} "
          f"(accepted 3.418)")

    # --- hard-sphere / high-T limit ------------------------------------
    print(f"high-T limit:  B*(100) = {B_star(100.0):.4f} "
          f"(repulsion-dominated, approaches HS positive constant)")

    # --- real argon: LJ parameters eps/k=119.8 K, sigma=0.3405 nm ------
    eps_k = 119.8          # K
    sigma = 3.405e-10      # m
    b0 = (2.0 / 3.0) * np.pi * NA * sigma ** 3          # m^3/mol
    print(f"\nArgon (eps/k={eps_k} K, sigma={sigma*1e9:.3f} nm), "
          f"b0 = {b0*1e6:.2f} cm^3/mol")
    print(f"  {'T (K)':>7} {'T*':>7} {'B (cm^3/mol)':>14} {'experiment':>12}")
    exp_data = {150: -86.2, 200: -47.6, 300: -15.5, 400: -1.0, 500: 7.0}
    for T, Bexp in exp_data.items():
        Ts = T / eps_k
        B = B_star(Ts) * b0 * 1e6                        # cm^3/mol
        print(f"  {T:>7} {Ts:>7.3f} {B:>14.2f} {Bexp:>12.1f}")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    Ts = np.linspace(0.6, 12, 300)
    Bs = np.array([B_star(t) for t in Ts])
    ax1.plot(Ts, Bs, color="#2c3e50", lw=2)
    ax1.axhline(0, color="k", lw=0.6)
    ax1.axvline(T_boyle, color="#c0392b", ls=":", lw=1.5,
                label=f"Boyle $T^*_B={T_boyle:.3f}$")
    ax1.set_xlabel(r"reduced temperature  $T^*=k_BT/\varepsilon$")
    ax1.set_ylabel(r"$B^*(T^*)=B/b_0$")
    ax1.set_title("(a)  reduced virial coefficient")
    ax1.legend(frameon=False)

    eps_k, sigma = 119.8, 3.405e-10
    b0 = (2/3) * np.pi * NA * sigma ** 3
    Tg = np.linspace(120, 600, 200)
    ax2.plot(Tg, [B_star(T/eps_k) * b0 * 1e6 for T in Tg], color="#2c3e50",
             lw=2, label="LJ theory")
    ax2.plot(list(exp_data.keys()), list(exp_data.values()), "o",
             color="#c0392b", ms=7, label="argon experiment")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("temperature  $T$  (K)")
    ax2.set_ylabel(r"$B$  (cm$^3$/mol)")
    ax2.set_title("(b)  argon: theory vs experiment")
    ax2.legend(frameon=False)

    fig.suptitle("Second virial coefficient from the Lennard-Jones potential",
                 y=1.05, fontsize=13)
    fig.savefig("fig7_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig7_2.png")


if __name__ == "__main__":
    main()
