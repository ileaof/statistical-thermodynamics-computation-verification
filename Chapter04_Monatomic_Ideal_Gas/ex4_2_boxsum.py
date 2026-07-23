#!/usr/bin/env python3
"""
Example 4.2 (direct numerical) -- The translational partition function by
summation over particle-in-a-box states, and the classical continuum limit.

A particle in a cubic box of side L has energy levels
    eps(nx,ny,nz) = (h^2 / 8 m L^2) (nx^2 + ny^2 + nz^2),  n_i = 1,2,3,...
so the translational partition function factorises, z = z1^3, with the
one-dimensional sum

    z1(alpha) = sum_{n=1}^inf exp(-alpha n^2),   alpha = h^2 / (8 m L^2 k_B T).

The parameter alpha is the level spacing measured in units of k_B T.  The
classical (continuum) approximation replaces the sum by an integral,

    z1_cont = integral_0^inf exp(-alpha n^2) dn = (1/2) sqrt(pi/alpha) = L/Lambda,

with Lambda the thermal wavelength.  Poisson summation shows the exact sum equals
the continuum value minus 1/2 (a boundary term) plus exponentially small
corrections, so the RELATIVE error of the continuum approximation is

    (z1_cont - z1) / z1_cont  ->  sqrt(alpha/pi) = Lambda / (2 L).

This script sums z1 numerically, verifies the sqrt(alpha) error law, and shows
that for a macroscopic box the error is utterly negligible -- the quantitative
justification for treating translation classically.  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt

h  = 6.62607015e-34
kB = 1.380649e-23
u  = 1.66053906660e-27


def z1_sum(alpha):
    """Exact one-dimensional sum sum_{n>=1} exp(-alpha n^2), truncated safely."""
    n_max = max(50, int(10.0 / np.sqrt(alpha)))       # e^{-alpha n^2} negligible
    n = np.arange(1, n_max + 1)
    return np.sum(np.exp(-alpha * n.astype(float) ** 2))


def z1_continuum(alpha):
    """Continuum approximation (1/2) sqrt(pi/alpha) = L/Lambda."""
    return 0.5 * np.sqrt(np.pi / alpha)


def main():
    print("=" * 70)
    print("Example 4.2  Translational partition function: sum vs continuum")
    print("=" * 70)

    # --- Verify the sqrt(alpha/pi) relative-error law -------------------
    print("\n(1) One-dimensional sum vs continuum:")
    print(f"    {'alpha':>10} {'z1 (sum)':>14} {'z1 (cont)':>14} "
          f"{'rel err':>11} {'sqrt(a/pi)':>11}")
    alphas = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    rel = []
    for a in alphas:
        zs, zc = z1_sum(a), z1_continuum(a)
        r = (zc - zs) / zc
        rel.append(r)
        print(f"    {a:>10.1e} {zs:>14.4f} {zc:>14.4f} {r:>11.3e} "
              f"{np.sqrt(a/np.pi):>11.3e}")
    rel = np.array(rel)
    slope = np.polyfit(np.log(alphas), np.log(np.abs(rel)), 1)[0]
    print(f"    fitted error slope d(log err)/d(log alpha) = {slope:.3f} "
          f"(expected 0.5)")

    # --- A macroscopic argon sample: how tiny is the error? ------------
    print("\n(2) Realistic argon gas at T = 300 K, L = 1 cm:")
    m = 39.948 * u
    T, L = 300.0, 1.0e-2
    alpha = h ** 2 / (8.0 * m * L ** 2 * kB * T)
    Lam = h / np.sqrt(2 * np.pi * m * kB * T)
    print(f"    alpha = {alpha:.3e},  Lambda = {Lam:.3e} m,  Lambda/L = "
          f"{Lam/L:.3e}")
    print(f"    continuum relative error ~ Lambda/(2L) = {Lam/(2*L):.3e}")
    print(f"    number of thermally accessible states per axis ~ L/Lambda = "
          f"{L/Lam:.3e}")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    # (a) sum and continuum overlaid vs alpha
    ag = np.logspace(-6, 0.3, 60)
    ax1.loglog(ag, [z1_sum(a) for a in ag], "o", color="#2c3e50", ms=4,
               label="exact sum")
    ax1.loglog(ag, [z1_continuum(a) for a in ag], "-", color="#c0392b", lw=2,
               label=r"continuum $L/\Lambda$")
    ax1.set_xlabel(r"$\alpha = h^2/(8mL^2k_BT)$")
    ax1.set_ylabel(r"$z_1$")
    ax1.set_title("(a)  sum meets continuum as $\\alpha\\to0$")
    ax1.legend(frameon=False)

    # (b) relative error law
    ag = np.logspace(-6, -0.5, 60)
    err = [(z1_continuum(a) - z1_sum(a)) / z1_continuum(a) for a in ag]
    ax2.loglog(ag, err, "o", color="#2c3e50", ms=4, label="relative error")
    ax2.loglog(ag, np.sqrt(ag / np.pi), "-", color="#c0392b", lw=2,
               label=r"$\sqrt{\alpha/\pi}=\Lambda/2L$")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("continuum relative error")
    ax2.set_title(r"(b)  error $\propto \sqrt{\alpha}$")
    ax2.legend(frameon=False)

    fig.suptitle("Translational partition function: discrete sum vs classical "
                 "continuum", y=1.05, fontsize=13)
    fig.savefig("fig4_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig4_2.png")


if __name__ == "__main__":
    main()
