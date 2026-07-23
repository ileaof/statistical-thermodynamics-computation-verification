#!/usr/bin/env python3
"""
Example 8.2 (direct numerical) -- The Debye model of a solid.

Debye (1912) treated the vibrations of a solid as a continuum of elastic waves up
to a maximum (Debye) frequency, giving the molar heat capacity

    C_V = 9 R (T/theta_D)^3 integral_0^{theta_D/T} x^4 e^x / (e^x - 1)^2 dx,

with the Debye temperature theta_D = hbar omega_D / k_B.  This script evaluates the
Debye integral NUMERICALLY, verifies the two exact limits

    low T:   C_V -> (12 pi^4 / 5) R (T/theta_D)^3       (the Debye T^3 law)
    high T:  C_V -> 3 R                                  (Dulong-Petit),

compares Debye with Einstein, and fits real metals.  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

R = 8.314462618


def debye_integrand(x):
    # x^4 e^x/(e^x-1)^2, written stably for large x
    ex = np.exp(-x)
    return x ** 4 * ex / (1.0 - ex) ** 2


def debye_Cv(T, theta_D):
    """Molar Debye heat capacity by numerical integration."""
    T = np.atleast_1d(T).astype(float)
    out = np.empty_like(T)
    for i, t in enumerate(T):
        xD = theta_D / t
        I, _ = quad(debye_integrand, 1e-8, xD, limit=200)
        out[i] = 9.0 * R * (t / theta_D) ** 3 * I
    return out


def einstein_Cv(T, theta_E):
    x = theta_E / np.asarray(T, float)
    emx = np.exp(-x)
    return 3.0 * R * x ** 2 * emx / (1.0 - emx) ** 2


def main():
    print("=" * 70)
    print("Example 8.2  Debye model of a solid")
    print("=" * 70)

    theta_D = 343.0            # copper Debye temperature, K
    print(f"\nCopper:  theta_D = {theta_D} K")
    print(f"\n{'T (K)':>7} {'T/theta_D':>10} {'Cv (J/mol/K)':>13} {'Cv/3R':>8}")
    for T in [10, 30, 100, 200, 343, 800]:
        C = debye_Cv(T, theta_D)[0]
        print(f"{T:>7} {T/theta_D:>10.3f} {C:>13.4f} {C/(3*R):>8.4f}")

    # --- verify the low-T T^3 coefficient ------------------------------
    coeff_exact = 12 * np.pi ** 4 / 5.0
    print(f"\nLow-T T^3 law: Cv/R = {coeff_exact:.4f} (T/theta_D)^3")
    for T in [3, 5, 10]:
        C = debye_Cv(T, theta_D)[0]
        coeff_num = (C / R) / (T / theta_D) ** 3
        print(f"  T={T:>3} K:  measured coefficient = {coeff_num:.4f} "
              f"(exact {coeff_exact:.4f})")

    # --- verify high-T limit -------------------------------------------
    print(f"\nHigh-T: Cv/3R at T=5 theta_D = "
          f"{debye_Cv(5*theta_D, theta_D)[0]/(3*R):.5f} (Dulong-Petit -> 1)")

    # --- fit real metals -----------------------------------------------
    metals = {"Cu": 343, "Ag": 225, "Pb": 105, "Al": 428}
    print("\nMolar Cv at 100 K for several metals (Debye, J/mol/K):")
    for m, td in metals.items():
        print(f"  {m}: theta_D={td} K,  Cv(100K) = {debye_Cv(100.0, td)[0]:.3f}")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    x = np.linspace(0.02, 1.6, 200)          # T/theta
    ax1.plot(x, debye_Cv(x * theta_D, theta_D) / (3 * R), color="#2c3e50",
             lw=2.2, label="Debye")
    ax1.plot(x, einstein_Cv(x * theta_D, theta_D) / (3 * R), color="#2980b9",
             lw=2, ls="--", label="Einstein (same $\\theta$)")
    ax1.axhline(1.0, color="#c0392b", ls=":", lw=1.5, label="Dulong-Petit")
    ax1.set_xlabel(r"$T/\theta$"); ax1.set_ylabel(r"$C_V/3R$")
    ax1.set_title("(a)  Debye vs Einstein")
    ax1.legend(frameon=False)

    # low-T: Cv/T^3 vs T showing the constant Debye coefficient
    Tlo = np.linspace(2, 40, 200)
    ax2.plot(Tlo, debye_Cv(Tlo, theta_D) / Tlo ** 3, color="#2c3e50", lw=2,
             label="Debye $C_V/T^3$")
    ax2.axhline(coeff_exact * R / theta_D ** 3, color="#c0392b", ls="--", lw=1.5,
                label=r"$(12\pi^4/5)R/\theta_D^3$")
    ax2.set_xlabel("temperature  $T$  (K)")
    ax2.set_ylabel(r"$C_V/T^3$  (J mol$^{-1}$ K$^{-4}$)")
    ax2.set_title(r"(b)  the Debye $T^3$ law")
    ax2.legend(frameon=False)

    fig.suptitle("The Debye model and the $T^3$ law", y=1.05, fontsize=13)
    fig.savefig("fig8_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig8_2.png")


if __name__ == "__main__":
    main()
