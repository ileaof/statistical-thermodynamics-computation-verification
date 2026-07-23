#!/usr/bin/env python3
"""
Example 3.2 (direct numerical) -- The three statistics: Maxwell-Boltzmann,
Bose-Einstein and Fermi-Dirac, and their common classical limit.

The mean number of particles occupying a single-particle state of energy eps,
when the system exchanges particles with a reservoir at temperature T and
chemical potential mu, is

    Maxwell-Boltzmann:  <n>_MB = exp[-(eps-mu)/kT]
    Bose-Einstein:      <n>_BE = 1 / (exp[(eps-mu)/kT] - 1)
    Fermi-Dirac:        <n>_FD = 1 / (exp[(eps-mu)/kT] + 1)

Writing x = (eps - mu)/kT, all three depend on x alone.  Fermions never exceed
one per state (Pauli); bosons diverge as x -> 0 (condensation tendency); and for
x >> 1 -- the dilute / non-degenerate regime -- the +-1 becomes negligible and
BOTH quantum statistics collapse onto the classical Maxwell-Boltzmann result.
This script evaluates the three distributions, verifies the classical limit by
measuring the relative deviation from MB as a function of x, and confirms it
decays as exp(-x).

Only numpy, scipy, matplotlib are used.
"""

import numpy as np
import matplotlib.pyplot as plt


def n_MB(x):
    return np.exp(-x)


def n_BE(x):
    # defined for x > 0 (eps > mu); diverges as x -> 0+
    return 1.0 / (np.exp(x) - 1.0)


def n_FD(x):
    return 1.0 / (np.exp(x) + 1.0)


def main():
    print("=" * 70)
    print("Example 3.2  Maxwell-Boltzmann, Bose-Einstein, Fermi-Dirac statistics")
    print("=" * 70)

    # --- Tabulate the three occupations and the deviation from MB -------
    print(f"\n{'x=(e-mu)/kT':>12} {'n_BE':>10} {'n_MB':>10} {'n_FD':>10} "
          f"{'(BE-MB)/MB':>12} {'(MB-FD)/MB':>12}")
    for x in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        be, mb, fd = n_BE(x), n_MB(x), n_FD(x)
        print(f"{x:>12.2f} {be:>10.4f} {mb:>10.4f} {fd:>10.4f} "
              f"{(be-mb)/mb:>12.4e} {(mb-fd)/mb:>12.4e}")

    # --- Verify the classical limit: deviations decay as exp(-x) --------
    # (BE - MB)/MB = 1/(1 - e^{-x}) * e^{-x} ... -> e^{-x} for large x; likewise
    # (MB - FD)/MB -> e^{-x}.  Fit the log-slope to confirm the rate.
    x = np.linspace(3.0, 10.0, 60)
    dev_BE = (n_BE(x) - n_MB(x)) / n_MB(x)
    dev_FD = (n_MB(x) - n_FD(x)) / n_MB(x)
    slope_BE = np.polyfit(x, np.log(dev_BE), 1)[0]
    slope_FD = np.polyfit(x, np.log(dev_FD), 1)[0]
    print(f"\nClassical-limit convergence (relative deviation ~ e^(-x)):")
    print(f"  BE deviation log-slope = {slope_BE:.4f}  (expected -1)")
    print(f"  FD deviation log-slope = {slope_FD:.4f}  (expected -1)")

    # --- Sanity identity: n_BE and n_FD bracket n_MB --------------------
    xt = np.linspace(0.2, 8, 400)
    assert np.all(n_BE(xt) >= n_MB(xt)) and np.all(n_MB(xt) >= n_FD(xt))
    print("  verified for all x>0:  n_FD <= n_MB <= n_BE")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    x = np.linspace(0.05, 5, 500)
    ax1.plot(x, n_BE(x), color="#2980b9", lw=2, label="Bose-Einstein")
    ax1.plot(x, n_MB(x), color="#2c3e50", lw=2, ls="--", label="Maxwell-Boltzmann")
    ax1.plot(x, n_FD(x), color="#c0392b", lw=2, label="Fermi-Dirac")
    ax1.axhline(1.0, color="k", ls=":", lw=1)
    ax1.set_ylim(0, 3)
    ax1.set_xlabel(r"$(\varepsilon-\mu)/k_BT$")
    ax1.set_ylabel(r"mean occupation  $\langle n\rangle$")
    ax1.set_title("(a)  the three distributions")
    ax1.legend(frameon=False)

    x = np.linspace(0.5, 10, 400)
    ax2.semilogy(x, np.abs(n_BE(x)-n_MB(x))/n_MB(x), color="#2980b9", lw=2,
                 label=r"$|n_{BE}-n_{MB}|/n_{MB}$")
    ax2.semilogy(x, np.abs(n_MB(x)-n_FD(x))/n_MB(x), color="#c0392b", lw=2,
                 label=r"$|n_{MB}-n_{FD}|/n_{MB}$")
    ax2.semilogy(x, np.exp(-x), "k:", lw=1.5, label=r"$e^{-x}$")
    ax2.set_xlabel(r"$(\varepsilon-\mu)/k_BT$")
    ax2.set_ylabel("relative deviation from MB")
    ax2.set_title("(b)  quantum $\\to$ classical as $e^{-x}$")
    ax2.legend(frameon=False)

    fig.suptitle("Quantum statistics and the classical (dilute) limit",
                 y=1.05, fontsize=13)
    fig.savefig("fig3_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig3_2.png")


if __name__ == "__main__":
    main()
