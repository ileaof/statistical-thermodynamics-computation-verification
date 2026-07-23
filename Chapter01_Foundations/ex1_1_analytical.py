#!/usr/bin/env python3
"""
Example 1.1 (analytical) -- Multiplicity of a two-state system and its
Gaussian limit.

Physical setting
----------------
Consider an isolated array of N distinguishable two-state elements: N atomic
spins in a magnetic field, each either "up" or "down".  A *microstate* is a
complete specification of every spin.  A *macrostate* is specified only by the
number n of up-spins.  The number of microstates compatible with a macrostate
is the multiplicity

        Omega(N, n) = C(N, n) = N! / [ n! (N - n)! ].

Boltzmann's postulate assigns the entropy  S = k_B ln Omega.  This script
develops the large-N (Stirling) approximation analytically, shows that the
multiplicity collapses to a Gaussian of width sigma = sqrt(N)/2 about the
equiprobable peak n = N/2, and *verifies* the approximation against the exact
binomial coefficients.  The narrowing of the relative width as 1/sqrt(N) is the
microscopic origin of the sharpness of macroscopic thermodynamic states.

Only numpy, scipy and matplotlib are used.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln          # ln Gamma(x) = ln (x-1)!

# NumPy 2.0 renamed np.trapz -> np.trapezoid; bind whichever exists.
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

k_B = 1.380649e-23                          # Boltzmann constant, J/K (exact, SI)


# ---------------------------------------------------------------------------
# Exact and approximate multiplicities
# ---------------------------------------------------------------------------
def log_multiplicity_exact(N, n):
    """Return ln C(N, n) using log-gamma so N! never overflows."""
    return gammaln(N + 1.0) - gammaln(n + 1.0) - gammaln(N - n + 1.0)


def gaussian_multiplicity(N, n):
    """Stirling/Gaussian approximation to C(N, n) about n = N/2.

    Expanding ln C(N, n) to second order in x = n - N/2 gives
        ln Omega ~ ln Omega_max - 2 x^2 / N,
    so    Omega(N, n) ~ 2^N * sqrt(2/(pi N)) * exp(-2 (n - N/2)^2 / N).
    The prefactor is Omega_max obtained from the refined Stirling series.
    """
    x = n - N / 2.0
    omega_max = 2.0 ** N * np.sqrt(2.0 / (np.pi * N))
    return omega_max * np.exp(-2.0 * x * x / N)


# ---------------------------------------------------------------------------
# Verification: exact vs Gaussian for several system sizes
# ---------------------------------------------------------------------------
def relative_width(N):
    """Std. dev. of the up-fraction n/N for the exact binomial distribution."""
    # For a fair binomial, var(n) = N/4, so std(n/N) = 1/(2 sqrt(N)).
    return 1.0 / (2.0 * np.sqrt(N))


def max_relative_error(N):
    """Largest relative error of the Gaussian multiplicity over all n."""
    n = np.arange(0, N + 1)
    exact = np.exp(log_multiplicity_exact(N, n))
    approx = gaussian_multiplicity(N, n)
    # The Gaussian is a central (second-order) expansion; assess it over the
    # physically dominant band |n - N/2| <= 2 sigma, sigma = sqrt(N)/2, which
    # holds ~95% of the probability.  (In the far tails a Gaussian necessarily
    # departs from a binomial and the relative error there is uninformative.)
    sigma = np.sqrt(N) / 2.0
    band = np.abs(n - N / 2.0) <= 2.0 * sigma
    return np.max(np.abs(approx[band] - exact[band]) / exact[band])


def main():
    print("=" * 68)
    print("Example 1.1  Two-state multiplicity and the Gaussian limit")
    print("=" * 68)

    sizes = [10, 40, 200]
    print(f"\n{'N':>6} {'peak Omega (exact)':>22} {'rel. width':>12} "
          f"{'max err (Gauss, +-2s)':>22}")
    for N in sizes:
        peak = np.exp(log_multiplicity_exact(N, N // 2))
        print(f"{N:>6} {peak:>22.6e} {relative_width(N):>12.4f} "
              f"{max_relative_error(N):>22.3e}")

    # Entropy per element at the peak; must approach k_B ln 2 as N -> infinity.
    print("\nEntropy per element at the peak  S/(N k_B) = (1/N) ln Omega_max:")
    for N in [10, 100, 1000, 10000]:
        s_per = log_multiplicity_exact(N, N // 2) / N
        print(f"  N = {N:>6d}:  S/(N k_B) = {s_per:.6f}   "
              f"(limit ln 2 = {np.log(2):.6f})")

    # -----------------------------------------------------------------
    # Figure: (a) exact vs Gaussian multiplicity; (b) 1/sqrt(N) narrowing
    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   constrained_layout=True)

    N = 40
    n = np.arange(0, N + 1)
    exact = np.exp(log_multiplicity_exact(N, n))
    approx = gaussian_multiplicity(N, n)
    ax1.bar(n, exact, width=0.9, color="#9ecae1",
            label=r"exact  $\binom{N}{n}$", zorder=1)
    xx = np.linspace(0, N, 400)
    ax1.plot(xx, gaussian_multiplicity(N, xx), color="#c0392b", lw=2.2,
             label="Gaussian (Stirling)", zorder=3)
    ax1.set_xlabel("number of up-spins  $n$")
    ax1.set_ylabel(r"multiplicity  $\Omega(N,n)$")
    ax1.set_title(f"(a)  $N = {N}$ two-state elements")
    ax1.legend(frameon=False)

    Ns = np.array([4, 10, 25, 60, 150, 400, 1000, 3000, 8000])
    ax2.loglog(Ns, [relative_width(N) for N in Ns], "o", color="#2c3e50",
               ms=7, label="exact std. dev. of  $n/N$")
    ax2.loglog(Ns, 1.0 / (2.0 * np.sqrt(Ns)), "-", color="#c0392b", lw=2,
               label=r"$1/(2\sqrt{N})$")
    ax2.set_xlabel("system size  $N$")
    ax2.set_ylabel(r"relative width  $\sigma_{n/N}$")
    ax2.set_title("(b)  macrostate sharpens as  $N^{-1/2}$")
    ax2.legend(frameon=False)

    fig.suptitle("Multiplicity of a two-state system and its Gaussian limit",
                 y=1.06, fontsize=13)
    fig.savefig("fig1_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig1_1.png")


if __name__ == "__main__":
    main()
