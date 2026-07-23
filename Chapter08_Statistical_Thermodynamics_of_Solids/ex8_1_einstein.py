#!/usr/bin/env python3
"""
Example 8.1 (analytical) -- The Einstein model of a solid.

Einstein (1907) modelled a solid as 3N independent quantum harmonic oscillators,
all of the same frequency omega_E.  The molar heat capacity is then 3 times the
single-oscillator (Chapter 3) result,

    C_V = 3 R (theta_E/T)^2 e^{theta_E/T} / (e^{theta_E/T} - 1)^2,
    theta_E = hbar omega_E / k_B.

This one formula explained, for the first time, why heat capacities fall below the
classical Dulong-Petit value 3R as the temperature drops -- a landmark success of
the quantum theory.  This script evaluates C_V for diamond, verifies the
high-temperature Dulong-Petit limit and the low-temperature exponential freeze-out,
and compares with representative experimental points.  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt

R = 8.314462618


def einstein_Cv(T, theta_E):
    """Molar heat capacity of the Einstein solid (numerically stable)."""
    x = theta_E / np.asarray(T, float)
    emx = np.exp(-x)                                    # stable for large x
    return 3.0 * R * x ** 2 * emx / (1.0 - emx) ** 2


def main():
    print("=" * 70)
    print("Example 8.1  Einstein model of a solid")
    print("=" * 70)

    theta_E = 1320.0            # diamond Einstein temperature, K
    print(f"\nDiamond:  theta_E = {theta_E} K,  Dulong-Petit 3R = {3*R:.3f} "
          f"J/mol/K")

    print(f"\n{'T (K)':>7} {'T/theta_E':>10} {'Cv (J/mol/K)':>13} {'Cv/3R':>8}")
    for T in [100, 300, 500, 1000, 2000, 5000]:
        C = einstein_Cv(T, theta_E)
        print(f"{T:>7} {T/theta_E:>10.3f} {C:>13.4f} {C/(3*R):>8.4f}")

    # --- verify limits -------------------------------------------------
    print("\nLimit checks:")
    C_hi = einstein_Cv(50000.0, theta_E)
    print(f"  high-T (T=50000 K): Cv/3R = {C_hi/(3*R):.5f} (Dulong-Petit -> 1)")
    T_lo = 100.0
    x = theta_E / T_lo
    C_lo_exact = einstein_Cv(T_lo, theta_E)
    C_lo_asym = 3 * R * x ** 2 * np.exp(-x)            # low-T asymptotic form
    print(f"  low-T (T=100 K): Cv = {C_lo_exact:.4e}, asymptotic 3R x^2 e^-x = "
          f"{C_lo_asym:.4e} (ratio {C_lo_exact/C_lo_asym:.4f})")

    # --- representative experimental diamond heat capacities (J/mol/K) --
    exp_T = np.array([200, 300, 400, 600, 900, 1200])
    exp_C = np.array([1.90, 6.11, 10.8, 16.9, 21.6, 23.8])   # approximate
    print("\nDiamond experiment vs Einstein (J/mol/K):")
    print(f"  {'T (K)':>7} {'exp':>7} {'Einstein':>9}")
    for T, Ce in zip(exp_T, exp_C):
        print(f"  {T:>7} {Ce:>7.2f} {einstein_Cv(T, theta_E):>9.2f}")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    T = np.linspace(20, 3000, 400)
    ax1.plot(T, einstein_Cv(T, theta_E) / R, color="#2c3e50", lw=2,
             label="Einstein")
    ax1.axhline(3.0, color="#c0392b", ls="--", lw=1.5, label="Dulong-Petit $3R$")
    ax1.plot(exp_T, exp_C / R, "o", color="#27ae60", ms=7, label="diamond exp.")
    ax1.set_xlabel("temperature  $T$  (K)")
    ax1.set_ylabel(r"$C_V/R$")
    ax1.set_title("(a)  diamond heat capacity")
    ax1.legend(frameon=False)

    x = np.linspace(0.05, 2.0, 400)          # T/theta_E
    ax2.plot(x, einstein_Cv(x * theta_E, theta_E) / (3 * R), color="#2c3e50",
             lw=2, label="Einstein")
    ax2.axhline(1.0, color="#c0392b", ls="--", lw=1.5, label="Dulong-Petit")
    ax2.set_xlabel(r"$T/\theta_E$")
    ax2.set_ylabel(r"$C_V/3R$")
    ax2.set_title("(b)  universal Einstein curve")
    ax2.legend(frameon=False)

    fig.suptitle("The Einstein model of heat capacity", y=1.05, fontsize=13)
    fig.savefig("fig8_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig8_1.png")


if __name__ == "__main__":
    main()
