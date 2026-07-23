#!/usr/bin/env python3
"""
Example 3.1 (analytical) -- The two-level system: thermodynamics from the
canonical partition function, and the Schottky anomaly.

A single element has just two non-degenerate levels, of energy 0 and eps.  Its
canonical partition function is

    Z = 1 + exp(-beta eps),     beta = 1 / (k_B T),

from which EVERY thermodynamic quantity follows by differentiation:

    U  = eps exp(-beta eps) / Z            (mean energy)
    F  = -k_B T ln Z                        (Helmholtz free energy)
    S  = (U - F) / T                        (entropy)
    C  = dU/dT = k_B (beta eps)^2 e^{beta eps} / (e^{beta eps}+1)^2   (heat cap.)

The heat capacity rises from zero, peaks near k_B T ~ 0.417 eps, and falls again
-- the Schottky anomaly, a fingerprint of a finite level gap seen in paramagnetic
salts, defects, and two-level tunnelling systems.  This script derives each
quantity, locates the Schottky peak with a root finder, and VERIFIES the heat
capacity two independent ways: by finite-differencing U(T) and by the
fluctuation formula C = (<E^2> - <E>^2)/(k_B T^2).

Reduced units: eps = 1, k_B = 1, so temperature is measured in units of eps/k_B.
Only numpy, scipy, matplotlib are used.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

EPS = 1.0          # level gap (energy unit)
kB  = 1.0          # Boltzmann constant (reduced units)


def partition(T):
    """Two-level canonical partition function Z(T)."""
    return 1.0 + np.exp(-EPS / (kB * T))


def energy(T):
    """Mean energy U(T) = eps / (e^{eps/kT} + 1)."""
    return EPS / (np.exp(EPS / (kB * T)) + 1.0)


def entropy(T):
    """Entropy S(T) from S = (U - F)/T with F = -kT ln Z."""
    F = -kB * T * np.log(partition(T))
    return (energy(T) - F) / T


def heat_capacity_closed(T):
    """Closed-form heat capacity (the Schottky form)."""
    x = EPS / (kB * T)
    ex = np.exp(x)
    return kB * x ** 2 * ex / (ex + 1.0) ** 2


def heat_capacity_fluctuation(T):
    """Heat capacity from energy fluctuations: C = (<E^2>-<E>^2)/(kB T^2)."""
    x = EPS / (kB * T)
    p1 = np.exp(-x) / partition(T)                 # occupation of the upper level
    mean_E = EPS * p1
    mean_E2 = EPS ** 2 * p1                         # E^2 = eps^2 only in upper level
    var_E = mean_E2 - mean_E ** 2                   # = eps^2 p1 (1 - p1)
    return var_E / (kB * T ** 2)


def main():
    print("=" * 70)
    print("Example 3.1  Two-level system: thermodynamics and the Schottky peak")
    print("=" * 70)

    # --- Locate the Schottky maximum:  d C / dT = 0 ---------------------
    dCdx = lambda x: (np.exp(x)*(x**2*(1-np.exp(x)) + 2*x*(1+np.exp(x)))
                      / (1+np.exp(x))**3)
    x_star = brentq(dCdx, 1.0, 5.0)
    T_star = EPS / (kB * x_star)
    C_star = heat_capacity_closed(T_star)
    print(f"\nSchottky peak: x* = eps/kT* = {x_star:.4f}, "
          f"kT*/eps = {1/x_star:.4f}")
    print(f"               C_max/k_B = {C_star:.4f}")

    # --- Verify heat capacity three ways at sample temperatures --------
    print(f"\n{'kT/eps':>8} {'C closed':>11} {'C fluct':>11} {'C finite-diff':>14}")
    for T in [0.2, 0.417, 0.6, 1.0, 2.0]:
        dT = 1e-5
        C_fd = (energy(T + dT) - energy(T - dT)) / (2 * dT)
        print(f"{T:>8.3f} {heat_capacity_closed(T):>11.6f} "
              f"{heat_capacity_fluctuation(T):>11.6f} {C_fd:>14.6f}")
    # max discrepancy across a fine grid
    Tg = np.linspace(0.05, 5, 500)
    err = np.max(np.abs(heat_capacity_closed(Tg) - heat_capacity_fluctuation(Tg)))
    print(f"\nmax|C_closed - C_fluct| over grid = {err:.2e}  (should be ~0)")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
    T = np.linspace(0.02, 3.0, 500)

    ax = axes[0, 0]
    ax.plot(T, 1.0 / partition(T), color="#2c3e50", lw=2, label="$P_0$ (ground)")
    ax.plot(T, np.exp(-EPS/(kB*T))/partition(T), color="#c0392b", lw=2,
            label="$P_1$ (excited)")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel(r"$k_BT/\varepsilon$"); ax.set_ylabel("occupation probability")
    ax.set_title("(a)  level populations"); ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(T, energy(T), color="#2c3e50", lw=2)
    ax.axhline(0.5, color="#c0392b", ls="--", lw=1.5, label=r"$\varepsilon/2$ limit")
    ax.set_xlabel(r"$k_BT/\varepsilon$"); ax.set_ylabel(r"$U/\varepsilon$")
    ax.set_title("(b)  mean energy saturates"); ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(T, entropy(T), color="#2c3e50", lw=2)
    ax.axhline(np.log(2), color="#c0392b", ls="--", lw=1.5, label=r"$\ln 2$ limit")
    ax.set_xlabel(r"$k_BT/\varepsilon$"); ax.set_ylabel(r"$S/k_B$")
    ax.set_title("(c)  entropy $\\to k_B\\ln 2$"); ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(T, heat_capacity_closed(T), color="#2c3e50", lw=2, label="closed form")
    ax.plot(T, heat_capacity_fluctuation(T), "o", color="#c0392b", ms=3,
            markevery=15, label="fluctuation")
    ax.axvline(T_star, color="#27ae60", ls=":", lw=1.5,
               label=f"peak at $k_BT/\\varepsilon={1/x_star:.3f}$")
    ax.set_xlabel(r"$k_BT/\varepsilon$"); ax.set_ylabel(r"$C/k_B$")
    ax.set_title("(d)  Schottky anomaly"); ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Two-level system: thermodynamics from the partition function",
                 y=1.03, fontsize=13)
    fig.savefig("fig3_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig3_1.png")


if __name__ == "__main__":
    main()
