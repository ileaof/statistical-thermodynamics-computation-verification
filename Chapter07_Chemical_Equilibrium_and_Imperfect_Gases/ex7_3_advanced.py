#!/usr/bin/env python3
"""
Example 7.3 (advanced, research-grade) -- Virial coefficients of hard spheres by
Monte Carlo evaluation of the Mayer cluster integrals, with verification.

The virial expansion  Z = pV/Nk_B T = 1 + B2 n + B3 n^2 + ...  has coefficients
given by cluster integrals of the Mayer function f_ij = exp(-u_ij/k_B T) - 1.  For
hard spheres of diameter sigma, f = -1 when two centres overlap (r < sigma) and 0
otherwise, so the integrals become geometric volumes:

    B2 = -1/2 integral f12 d^3r   = (2/3) pi sigma^3            (exact),
    B3 = -1/3 integral f12 f13 f23 d^3r2 d^3r3 = (5/18) pi^2 sigma^6  (exact).

B3 is a genuine three-body integral.  Writing sigma = 1, the integrand of B3 is
-1 exactly when all three pairwise distances are < 1, so

    B3 = (16/27) pi^2 * p,   p = Prob(|r2-r3| < 1 | r2,r3 uniform in unit ball),

and the exact value fixes p = 0.46875.  This script estimates p by Monte Carlo,
recovers B2 and B3, shows the 1/sqrt(N) error law with error bars, and compares the
truncated virial equation of state with the accurate Carnahan-Starling formula.

numpy/scipy/matplotlib only; fixed seed.
"""

import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260723)

B2_EXACT = (2.0 / 3.0) * np.pi                 # sigma = 1
B3_EXACT = (5.0 / 18.0) * np.pi ** 2
P_EXACT  = 0.46875                             # (5/18)/(16/27)


def sample_unit_ball(n):
    """n uniform points in the unit ball via rejection from the cube."""
    pts = np.empty((n, 3))
    filled = 0
    while filled < n:
        m = int((n - filled) * 1.95) + 16
        c = RNG.uniform(-1, 1, size=(m, 3))
        keep = c[np.sum(c * c, axis=1) < 1.0]
        take = min(len(keep), n - filled)
        pts[filled:filled + take] = keep[:take]
        filled += take
    return pts


def mc_overlap_probability(n):
    """Estimate p = Prob(|r2-r3|<1) for r2,r3 uniform in the unit ball."""
    r2 = sample_unit_ball(n)
    r3 = sample_unit_ball(n)
    d = np.linalg.norm(r2 - r3, axis=1)
    hits = d < 1.0
    p = hits.mean()
    err = np.sqrt(p * (1 - p) / n)             # binomial standard error
    return p, err


def carnahan_starling(eta):
    """Carnahan-Starling compressibility factor Z(eta)."""
    return (1 + eta + eta ** 2 - eta ** 3) / (1 - eta) ** 3


def main():
    print("=" * 72)
    print("Example 7.3  Hard-sphere virial coefficients by Monte Carlo")
    print("=" * 72)
    print(f"\nExact (sigma=1):  B2 = (2/3)pi = {B2_EXACT:.6f},  "
          f"B3 = (5/18)pi^2 = {B3_EXACT:.6f}")

    # ---- (1) Monte Carlo convergence of p, B3 -------------------------
    print("\n(1) Monte Carlo estimate of the 3-body overlap probability p:")
    print(f"    {'N':>10} {'p (MC)':>12} {'+- err':>10} {'B3 (MC)':>12} "
          f"{'B3 err %':>10}")
    sizes = [10_000, 40_000, 160_000, 640_000, 2_560_000]
    ps, perr = [], []
    for n in sizes:
        p, e = mc_overlap_probability(n)
        ps.append(p); perr.append(e)
        B3 = (16.0 / 27.0) * np.pi ** 2 * p
        print(f"    {n:>10} {p:>12.6f} {e:>10.6f} {B3:>12.6f} "
              f"{100*abs(B3-B3_EXACT)/B3_EXACT:>10.4f}")
    ps, perr = np.array(ps), np.array(perr)
    order = np.polyfit(np.log(sizes), np.log(perr), 1)[0]
    print(f"    p_exact = {P_EXACT};  error-bar scaling order = {order:.3f} "
          f"(expected -0.5)")

    # ---- (2) final best estimate --------------------------------------
    p_best, e_best = ps[-1], perr[-1]
    B3_best = (16.0 / 27.0) * np.pi ** 2 * p_best
    B3_best_err = (16.0 / 27.0) * np.pi ** 2 * e_best
    print(f"\n(2) Best estimates (N={sizes[-1]:,}):")
    print(f"    B2 (exact geometry) = {B2_EXACT:.6f}")
    print(f"    B3 (MC)             = {B3_best:.6f} +- {B3_best_err:.6f} "
          f"(exact {B3_EXACT:.6f})")

    # ---- (3) equation of state: virial vs Carnahan-Starling -----------
    # reduced density in packing fraction eta = (pi/6) n sigma^3  -> n = 6 eta/pi
    print("\n(3) Compressibility factor Z vs Carnahan-Starling:")
    print(f"    {'eta':>6} {'Z 2-term':>10} {'Z 3-term':>10} {'CS':>10}")
    for eta in [0.05, 0.1, 0.2, 0.3]:
        n = 6 * eta / np.pi
        Z2 = 1 + B2_EXACT * n
        Z3 = 1 + B2_EXACT * n + B3_EXACT * n ** 2
        print(f"    {eta:>6.2f} {Z2:>10.4f} {Z3:>10.4f} "
              f"{carnahan_starling(eta):>10.4f}")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    # (a) MC convergence of B3 with error bars
    B3s = (16.0 / 27.0) * np.pi ** 2 * ps
    B3e = (16.0 / 27.0) * np.pi ** 2 * perr
    axes[0].errorbar(sizes, B3s, yerr=B3e, fmt="o", color="#2c3e50", ms=6,
                     capsize=3, label="MC")
    axes[0].axhline(B3_EXACT, color="#c0392b", lw=2, label=r"exact $(5/18)\pi^2$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("samples  $N$"); axes[0].set_ylabel(r"$B_3/\sigma^6$")
    axes[0].set_title("(a)  three-body coefficient")
    axes[0].legend(frameon=False)

    # (b) error-bar 1/sqrt(N) law
    axes[1].loglog(sizes, perr, "o", color="#2c3e50", ms=6, label="MC error")
    axes[1].loglog(sizes, perr[0] * np.sqrt(sizes[0] / np.array(sizes)), "-",
                   color="#c0392b", lw=2, label=r"$\propto N^{-1/2}$")
    axes[1].set_xlabel("samples  $N$"); axes[1].set_ylabel("statistical error in $p$")
    axes[1].set_title(r"(b)  Monte Carlo error law")
    axes[1].legend(frameon=False)

    # (c) equation of state
    eta = np.linspace(0, 0.45, 200)
    n = 6 * eta / np.pi
    axes[2].plot(eta, 1 + B2_EXACT * n, "--", color="#2980b9", lw=1.8,
                 label="virial (2-term)")
    axes[2].plot(eta, 1 + B2_EXACT * n + B3_EXACT * n ** 2, "-.",
                 color="#27ae60", lw=1.8, label="virial (3-term)")
    axes[2].plot(eta, carnahan_starling(eta), color="#c0392b", lw=2.2,
                 label="Carnahan-Starling")
    axes[2].set_xlabel(r"packing fraction  $\eta$")
    axes[2].set_ylabel(r"$Z=pV/Nk_BT$")
    axes[2].set_title("(c)  hard-sphere equation of state")
    axes[2].legend(frameon=False)
    axes[2].set_ylim(1, 6)

    fig.suptitle("Hard-sphere virial coefficients and equation of state", y=1.05,
                 fontsize=13)
    fig.savefig("fig7_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig7_3.png")


if __name__ == "__main__":
    main()
