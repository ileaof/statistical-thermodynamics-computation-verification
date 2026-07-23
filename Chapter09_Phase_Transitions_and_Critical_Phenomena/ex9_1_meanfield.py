#!/usr/bin/env python3
"""
Example 9.1 (analytical) -- Mean-field theory of the Ising ferromagnet.

In the mean-field (Weiss) approximation each spin feels the average field of its z
neighbours, giving the self-consistent equation for the magnetization per spin

    m = tanh( (z J m + H) / k_B T ).

With H = 0 this has a nonzero (ferromagnetic) solution only below the critical
temperature k_B T_c = z J.  In reduced temperature t = T/T_c the equation is
m = tanh(m/t), and near t = 1 one finds m ~ sqrt(3(1-t)), the mean-field
magnetization exponent beta = 1/2.  The Landau free energy

    f(m) = (1/2) a (T - T_c) m^2 + (1/4) b m^4 - H m

has a single minimum above T_c and a symmetric double well below -- the hallmark of
a continuous (second-order) transition.  This script solves for m(t), builds the
Landau free energy, and VERIFIES the critical exponents beta = 1/2 (magnetization)
and gamma = 1 (susceptibility).  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


def magnetization(t):
    """Spontaneous magnetization m(t): largest root of m = tanh(m/t).

    For small t the root lies within machine epsilon of 1 (full saturation), so
    the bracket loses its sign change; that case is returned as m = 1."""
    if t >= 1.0:
        return 0.0
    f = lambda m: np.tanh(m / t) - m
    try:
        return brentq(f, 1e-10, 1.0 - 1e-12)
    except ValueError:
        return 1.0                                # fully saturated


def susceptibility(t):
    """Zero-field susceptibility chi = dm/dH above T_c via m = tanh((m+h)/t).

    The field step is scaled to the shrinking linear-response window near T_c,
    dh << (t-1)^{3/2}, so the measured slope is the true linear susceptibility."""
    dh = 1e-3 * (t - 1.0)
    def m_of_h(h):
        f = lambda m: np.tanh((m + h) / t) - m
        return brentq(f, -0.5, 0.5)
    return (m_of_h(dh) - m_of_h(-dh)) / (2 * dh)


def main():
    print("=" * 70)
    print("Example 9.1  Mean-field theory of the Ising ferromagnet")
    print("=" * 70)

    print(f"\n{'t=T/Tc':>8} {'m(t)':>10}")
    for t in [0.2, 0.5, 0.8, 0.95, 0.99, 1.0, 1.1]:
        print(f"{t:>8.2f} {magnetization(t):>10.5f}")

    # --- verify beta = 1/2 near Tc -------------------------------------
    eps = np.logspace(-4, -1.5, 30)          # 1 - t
    m = np.array([magnetization(1 - e) for e in eps])
    beta_fit = np.polyfit(np.log(eps), np.log(m), 1)[0]
    print(f"\nMagnetization exponent: m ~ (1-t)^beta,  beta_fit = {beta_fit:.4f} "
          f"(mean-field 0.5)")
    print(f"  amplitude check: m/(1-t)^0.5 near Tc = {m[0]/eps[0]**0.5:.4f} "
          f"(sqrt3 = {np.sqrt(3):.4f})")

    # --- verify gamma = 1 (susceptibility) -----------------------------
    tt = 1 + np.logspace(-4, -2, 25)         # T just above Tc (asymptotic)
    chi = np.array([susceptibility(t) for t in tt])
    gamma_fit = -np.polyfit(np.log(tt - 1), np.log(chi), 1)[0]
    print(f"\nSusceptibility exponent: chi ~ (t-1)^-gamma, gamma_fit = "
          f"{gamma_fit:.4f} (mean-field 1)")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)

    # (a) order parameter m(t)
    ax = axes[0, 0]
    tg = np.linspace(0.01, 1.6, 300)
    ax.plot(tg, [magnetization(t) for t in tg], color="#2c3e50", lw=2.4)
    ax.axvline(1.0, color="#c0392b", ls=":", lw=1.5, label="$T_c$")
    ax.set_xlabel(r"$T/T_c$"); ax.set_ylabel(r"magnetization  $m$")
    ax.set_title("(a)  order parameter (second-order)"); ax.legend(frameon=False)

    # (b) Landau free energy
    ax = axes[0, 1]
    m = np.linspace(-1, 1, 300)
    for a_eff, c, lab in [(0.6, "#2980b9", "$T>T_c$"), (0.0, "#2c3e50", "$T=T_c$"),
                          (-0.6, "#c0392b", "$T<T_c$")]:
        f = 0.5 * a_eff * m ** 2 + 0.25 * m ** 4
        ax.plot(m, f, color=c, lw=2, label=lab)
    ax.set_xlabel(r"$m$"); ax.set_ylabel(r"Landau free energy  $f(m)$")
    ax.set_title("(b)  free energy: single well to double well")
    ax.legend(frameon=False)

    # (c) log-log magnetization exponent
    ax = axes[1, 0]
    eps = np.logspace(-4, -1, 40)
    m = np.array([magnetization(1 - e) for e in eps])
    ax.loglog(eps, m, "o", color="#2c3e50", ms=4, label="mean field")
    ax.loglog(eps, np.sqrt(3) * eps ** 0.5, "-", color="#c0392b", lw=2,
              label=r"$\sqrt{3}\,(1-t)^{1/2}$")
    ax.set_xlabel(r"$1-T/T_c$"); ax.set_ylabel(r"$m$")
    ax.set_title(r"(c)  exponent $\beta=1/2$"); ax.legend(frameon=False)

    # (d) susceptibility divergence
    ax = axes[1, 1]
    tt = 1 + np.logspace(-4, -1.5, 40)
    chi = np.array([susceptibility(t) for t in tt])
    ax.loglog(tt - 1, chi, "o", color="#2c3e50", ms=4, label="mean field")
    ax.loglog(tt - 1, 1.0 / (tt - 1), "-", color="#c0392b", lw=2,
              label=r"$(t-1)^{-1}$")
    ax.set_xlabel(r"$T/T_c-1$"); ax.set_ylabel(r"susceptibility  $\chi$")
    ax.set_title(r"(d)  exponent $\gamma=1$"); ax.legend(frameon=False)

    fig.suptitle("Mean-field theory of the Ising ferromagnet", y=1.03,
                 fontsize=13)
    fig.savefig("fig9_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig9_1.png")


if __name__ == "__main__":
    main()
