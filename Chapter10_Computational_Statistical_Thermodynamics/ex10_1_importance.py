#!/usr/bin/env python3
"""
Example 10.1 (analytical + Monte Carlo) -- Monte Carlo evaluation of a thermal
average, and the power of importance sampling.

We evaluate the canonical average <x^2> for a classical anharmonic oscillator with
potential V(x) = x^2/2 + lambda x^4 at temperature k_B T = 1/beta, i.e. over the
Boltzmann distribution p(x) = exp(-beta V(x)) / Z.  The exact value is obtained by
quadrature and used to VERIFY two Monte Carlo estimators:

  * Uniform sampling on [-L, L]: draws are spread evenly, so most land where the
    Boltzmann weight is tiny -- high variance.
  * Importance sampling: draws come from a Gaussian close to p(x), and each is
    reweighted by w = exp(-beta V)/g; samples concentrate where they matter --
    dramatically lower variance for the same number of draws.

Both estimators converge as 1/sqrt(N); importance sampling has a far smaller
prefactor.  numpy/scipy/matplotlib only; fixed seed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

RNG = np.random.default_rng(20260723)
BETA = 1.0
LAM = 0.1


def V(x):
    return 0.5 * x ** 2 + LAM * x ** 4


def exact_x2():
    """Exact <x^2> by quadrature."""
    num, _ = quad(lambda x: x ** 2 * np.exp(-BETA * V(x)), -np.inf, np.inf)
    den, _ = quad(lambda x: np.exp(-BETA * V(x)), -np.inf, np.inf)
    return num / den


def uniform_estimate(N, L=12.0):
    """<x^2> by uniform sampling on [-L, L] (ratio of two MC integrals)."""
    x = RNG.uniform(-L, L, N)
    w = np.exp(-BETA * V(x))
    return np.sum(x ** 2 * w) / np.sum(w)


def importance_estimate(N, sigma=1.0):
    """<x^2> by importance sampling from a Gaussian proposal N(0, sigma^2)."""
    x = RNG.normal(0.0, sigma, N)
    g = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    w = np.exp(-BETA * V(x)) / g
    return np.sum(x ** 2 * w) / np.sum(w)


def main():
    print("=" * 70)
    print("Example 10.1  Monte Carlo thermal average and importance sampling")
    print("=" * 70)

    x2_exact = exact_x2()
    print(f"\nExact <x^2> (quadrature) = {x2_exact:.6f}")

    # --- convergence and variance comparison ---------------------------
    print("\nRMS error over 40 independent runs:")
    print(f"  {'N':>8} {'uniform err':>14} {'importance err':>16} "
          f"{'var. reduction':>15}")
    Ns = [100, 400, 1600, 6400, 25600]
    err_u, err_i = [], []
    for N in Ns:
        eu = np.std([uniform_estimate(N) for _ in range(40)])
        ei = np.std([importance_estimate(N) for _ in range(40)])
        err_u.append(eu); err_i.append(ei)
        print(f"  {N:>8} {eu:>14.5f} {ei:>16.5f} {eu/ei:>15.1f}x")
    err_u, err_i = np.array(err_u), np.array(err_i)
    ou = np.polyfit(np.log(Ns), np.log(err_u), 1)[0]
    oi = np.polyfit(np.log(Ns), np.log(err_i), 1)[0]
    print(f"\n  convergence order: uniform {ou:.3f}, importance {oi:.3f} "
          f"(both -0.5 expected)")

    # single best estimate
    big = importance_estimate(1_000_000)
    print(f"\n  importance estimate (N=10^6) = {big:.6f} "
          f"(exact {x2_exact:.6f}, error {abs(big-x2_exact):.2e})")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    x = np.linspace(-8, 8, 400)
    Z, _ = quad(lambda t: np.exp(-BETA * V(t)), -np.inf, np.inf)
    ax1.plot(x, np.exp(-BETA * V(x)) / Z, color="#2c3e50", lw=2.4,
             label=r"Boltzmann $p(x)$")
    ax1.plot(x, np.full_like(x, 1 / 24), color="#2980b9", lw=1.8, ls="--",
             label="uniform proposal")
    ax1.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color="#c0392b",
             lw=1.8, ls=":", label="Gaussian proposal")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("probability density")
    ax1.set_title("(a)  target and proposal distributions")
    ax1.legend(frameon=False)

    ax2.loglog(Ns, err_u, "o", color="#2980b9", ms=7, label="uniform")
    ax2.loglog(Ns, err_i, "s", color="#c0392b", ms=7, label="importance")
    ax2.loglog(Ns, err_u[0] * np.sqrt(Ns[0] / np.array(Ns)), "-",
               color="#2c3e50", lw=1.5, label=r"$\propto N^{-1/2}$")
    ax2.loglog(Ns, err_i[0] * np.sqrt(Ns[0] / np.array(Ns)), "-",
               color="#2c3e50", lw=1.5)
    ax2.set_xlabel("samples  $N$"); ax2.set_ylabel("RMS error in $\\langle x^2\\rangle$")
    ax2.set_title("(b)  importance sampling reduces variance")
    ax2.legend(frameon=False)

    fig.suptitle("Monte Carlo integration and importance sampling", y=1.05,
                 fontsize=13)
    fig.savefig("fig10_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig10_1.png")


if __name__ == "__main__":
    main()
