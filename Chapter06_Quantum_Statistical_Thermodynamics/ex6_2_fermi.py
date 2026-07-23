#!/usr/bin/env python3
"""
Example 6.2 (direct numerical) -- The ideal Fermi gas: chemical potential, energy
and heat capacity across the classical-to-degenerate crossover.

For spin-1/2 fermions the density of states is g(eps) ~ sqrt(eps).  Working in
reduced units where the Fermi energy E_F = 1, Boltzmann constant k_B = 1 and the
Fermi temperature T_F = 1, and normalising the density of states so that the
particle number is one, g(eps) = (3/2) sqrt(eps), the number and energy are

    n = 1 = integral_0^inf (3/2) sqrt(eps) f(eps) deps,
    u   = integral_0^inf (3/2) eps^{3/2} f(eps) deps,   f = 1/(e^{(eps-mu)/T}+1).

The chemical potential mu(T) is found by ROOT FINDING on the number constraint,
and the heat capacity from a finite difference of u(T).  Results are checked
against three exact limits:

    T -> 0:      mu -> E_F = 1,  u -> 3/5
    low T:       mu ~ 1 - (pi^2/12) T^2  (Sommerfeld),  C ~ (pi^2/2) T
    high T:      C -> 3/2  (classical ideal gas)

numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def fermi(eps, mu, T):
    # stable logistic form, avoids overflow for large (eps-mu)/T
    return 0.5 * (1.0 - np.tanh(0.5 * (eps - mu) / T))


def _grid(mu, T):
    """Energy grid from 0 to well above the Fermi edge (integrand negligible)."""
    hi = max(3.0, mu + 60.0 * T)
    return np.linspace(0.0, hi, 4000)


def number(mu, T):
    """n(mu,T) = integral (3/2) sqrt(eps) f deps by trapezoidal quadrature."""
    e = _grid(mu, T)
    return trapezoid(1.5 * np.sqrt(e) * fermi(e, mu, T), e)


def energy(mu, T):
    e = _grid(mu, T)
    return trapezoid(1.5 * e ** 1.5 * fermi(e, mu, T), e)


def mu_of_T(T):
    """Solve number(mu,T) = 1 for the chemical potential."""
    # bracket: mu is ~1 at low T, large negative at high T
    lo, hi = -50.0 / max(T, 1e-3), 5.0
    return brentq(lambda m: number(m, T) - 1.0, lo, hi, xtol=1e-12)


def main():
    print("=" * 70)
    print("Example 6.2  Ideal Fermi gas: chemical potential, energy, C_V")
    print("=" * 70)

    # --- T -> 0 checks --------------------------------------------------
    print("\n(1) Low-temperature limits (reduced units E_F=T_F=1):")
    for T in [0.02, 0.05, 0.1]:
        mu = mu_of_T(T)
        mu_somm = 1.0 - (np.pi ** 2 / 12.0) * T ** 2
        print(f"  T/T_F={T:.2f}:  mu={mu:.6f}  Sommerfeld={mu_somm:.6f}  "
              f"(diff {abs(mu-mu_somm):.1e})")
    print(f"  u(T->0) target 3/5 = 0.6;  computed at T=0.02: "
          f"{energy(mu_of_T(0.02), 0.02):.5f}")

    # --- heat capacity by finite difference ----------------------------
    def heat_capacity(T, dT=1e-3):
        return (energy(mu_of_T(T + dT), T + dT)
                - energy(mu_of_T(T - dT), T - dT)) / (2 * dT)

    print("\n(2) Heat capacity vs limits:")
    print(f"  {'T/T_F':>7} {'C (num)':>10} {'Sommerfeld (pi^2/2)T':>22} "
          f"{'classical 3/2':>14}")
    for T in [0.05, 0.1, 0.2, 1.0, 3.0]:
        C = heat_capacity(T)
        print(f"  {T:>7.2f} {C:>10.5f} {np.pi**2/2*T:>22.5f} {1.5:>14.3f}")

    # verify low-T slope
    Tlo = 0.05
    slope = heat_capacity(Tlo) / Tlo
    print(f"\n  low-T slope C/T at T=0.05: {slope:.5f} "
          f"(Sommerfeld pi^2/2 = {np.pi**2/2:.5f})")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    Tg = np.linspace(0.02, 3.0, 60)
    mus = np.array([mu_of_T(t) for t in Tg])
    axes[0].plot(Tg, mus, color="#2c3e50", lw=2, label="numerical $\\mu(T)$")
    axes[0].plot(Tg, 1 - np.pi**2/12*Tg**2, "--", color="#c0392b", lw=1.8,
                 label="Sommerfeld")
    axes[0].axhline(1.0, color="#27ae60", ls=":", lw=1.2, label="$E_F$")
    axes[0].set_ylim(-3, 1.4)
    axes[0].set_xlabel(r"$T/T_F$"); axes[0].set_ylabel(r"$\mu/E_F$")
    axes[0].set_title("(a)  chemical potential"); axes[0].legend(frameon=False)

    Cg = np.array([heat_capacity(t) for t in Tg])
    axes[1].plot(Tg, Cg, color="#2c3e50", lw=2, label="numerical")
    axes[1].plot(Tg, np.pi**2/2*Tg, "--", color="#c0392b", lw=1.8,
                 label=r"$(\pi^2/2)\,T$")
    axes[1].axhline(1.5, color="#27ae60", ls=":", lw=1.2, label="classical $3/2$")
    axes[1].set_xlabel(r"$T/T_F$"); axes[1].set_ylabel(r"$C_V/Nk_B$")
    axes[1].set_title("(b)  heat capacity: linear then classical")
    axes[1].legend(frameon=False)

    eps = np.linspace(0, 2.5, 400)
    for T0, c in [(0.03, "#2c3e50"), (0.2, "#2980b9"), (0.6, "#c0392b")]:
        axes[2].plot(eps, fermi(eps, mu_of_T(T0), T0), color=c, lw=2,
                     label=f"$T/T_F={T0}$")
    axes[2].axvline(1.0, color="k", ls=":", lw=1)
    axes[2].set_xlabel(r"$\varepsilon/E_F$"); axes[2].set_ylabel(r"$f(\varepsilon)$")
    axes[2].set_title("(c)  Fermi-Dirac occupation")
    axes[2].legend(frameon=False)

    fig.suptitle("Ideal Fermi gas: degenerate to classical crossover", y=1.05,
                 fontsize=13)
    fig.savefig("fig6_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig6_2.png")


if __name__ == "__main__":
    main()
