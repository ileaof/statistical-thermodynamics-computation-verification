#!/usr/bin/env python3
"""
Example 6.1 (analytical) -- The photon gas: Planck's law, Stefan-Boltzmann, Wien.

Photons are massless spin-1 bosons with chemical potential zero (their number is
not conserved).  Counting the two polarizations and the photon density of states
gives the Planck spectral energy density (energy per unit volume per unit angular
frequency)

    u(omega, T) = (hbar omega^3) / (pi^2 c^3) * 1/(exp(hbar omega/kT) - 1).

Integrating over all frequencies gives the total energy density u = a T^4 with
a = pi^2 k^4 / (15 hbar^3 c^3), and the emitted flux is sigma T^4 with the
Stefan-Boltzmann constant sigma = a c / 4 = pi^2 k^4 / (60 hbar^3 c^2).  The
spectrum peaks at hbar omega = 2.821 kT (Wien displacement law).

This script integrates the Planck spectrum NUMERICALLY to recover sigma and
compares it with the CODATA value, finds the Wien peak by root finding, and plots
the spectrum.  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

h    = 6.62607015e-34
hbar = h / (2 * np.pi)
c    = 2.99792458e8
kB   = 1.380649e-23
SIGMA_CODATA = 5.670374419e-8          # W m^-2 K^-4
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def u_total_grid(T, xmax=60.0, n=4000):
    """Total energy density by trapezoidal integration on a T-scaled grid,
    so the sharply peaked Planck spectrum is always well resolved."""
    x = np.linspace(1e-4, xmax, n)
    omega = x * kB * T / hbar
    return trapezoid(planck_u_omega(omega, T), omega)


def planck_u_omega(omega, T):
    """Planck spectral energy density per angular frequency (J s m^-3)."""
    x = hbar * omega / (kB * T)
    return hbar * omega ** 3 / (np.pi ** 2 * c ** 3) / np.expm1(x)


def planck_u_nu(nu, T):
    """Planck spectral energy density per (ordinary) frequency (J s m^-3)."""
    x = h * nu / (kB * T)
    return 8.0 * np.pi * h * nu ** 3 / c ** 3 / np.expm1(x)


def main():
    print("=" * 70)
    print("Example 6.1  Photon gas: Planck, Stefan-Boltzmann, Wien")
    print("=" * 70)

    # --- Stefan-Boltzmann constant by numerical integration ------------
    T = 300.0
    # u = integral_0^inf u(omega) domega ; substitute x = hbar omega/kT
    integrand = lambda x: x ** 3 / np.expm1(x)
    I, _ = quad(integrand, 0, np.inf)                 # = pi^4/15
    print(f"\nDimensionless integral  int x^3/(e^x-1) dx = {I:.6f} "
          f"(exact pi^4/15 = {np.pi**4/15:.6f})")

    a_rad = (kB ** 4 / (np.pi ** 2 * c ** 3 * hbar ** 3)) * I
    a_exact = np.pi ** 2 * kB ** 4 / (15 * hbar ** 3 * c ** 3)
    sigma_num = a_rad * c / 4.0
    print(f"radiation constant a  = {a_rad:.6e} (exact {a_exact:.6e}) J m^-3 K^-4")
    print(f"Stefan-Boltzmann sigma = {sigma_num:.6e} W m^-2 K^-4")
    print(f"CODATA sigma           = {SIGMA_CODATA:.6e}")
    print(f"relative error         = {abs(sigma_num-SIGMA_CODATA)/SIGMA_CODATA:.2e}")

    # --- Verify total energy density directly at one temperature -------
    u_num = u_total_grid(T)
    print(f"\nDirect integral of Planck spectrum at T={T:.0f} K:")
    print(f"  u_num = {u_num:.6e} J/m^3,  a T^4 = {a_exact*T**4:.6e} J/m^3, "
          f"rel err {abs(u_num-a_exact*T**4)/(a_exact*T**4):.2e}")

    # --- Wien displacement law by root finding -------------------------
    # frequency peak: d/domega[omega^3/(e^x-1)] = 0 -> 3(1-e^{-x}) = x
    x_freq = brentq(lambda x: 3 * (1 - np.exp(-x)) - x, 1e-3, 10)
    # wavelength peak: 5(1-e^{-x}) = x
    x_wl = brentq(lambda x: 5 * (1 - np.exp(-x)) - x, 1e-3, 10)
    b_wien = h * c / (x_wl * kB)
    print(f"\nWien displacement:")
    print(f"  frequency peak  hbar*omega/kT = {x_freq:.5f} (known 2.82144)")
    print(f"  wavelength peak hc/(lambda kT) = {x_wl:.5f} (known 4.96511)")
    print(f"  Wien constant b = lambda_max T = {b_wien:.6e} m K "
          f"(known 2.897771e-3)")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    nu = np.linspace(1e12, 4e14, 500)
    for T0, col in [(3000, "#c0392b"), (4000, "#e67e22"), (5000, "#27ae60"),
                    (5778, "#2980b9")]:
        ax1.plot(nu / 1e12, planck_u_nu(nu, T0) * 1e12, color=col, lw=2,
                 label=f"{T0} K")
        nu_peak = x_freq * kB * T0 / h
        ax1.axvline(nu_peak / 1e12, color=col, ls=":", lw=1)
    ax1.set_xlabel(r"frequency  $\nu$  (THz)")
    ax1.set_ylabel(r"$u_\nu$  (J s m$^{-3}$ / THz)")
    ax1.set_title("(a)  Planck spectrum and Wien peak shift")
    ax1.legend(frameon=False, title="dotted: peak")

    Tg = np.logspace(2, 4, 60)
    u_tot = [u_total_grid(t) for t in Tg]
    ax2.loglog(Tg, u_tot, "o", color="#2c3e50", ms=4, label="numerical $u(T)$")
    ax2.loglog(Tg, a_exact * Tg ** 4, "-", color="#c0392b", lw=2,
               label=r"$a T^4$")
    ax2.set_xlabel("temperature  $T$  (K)")
    ax2.set_ylabel(r"energy density  $u$  (J/m$^3$)")
    ax2.set_title(r"(b)  Stefan-Boltzmann  $u \propto T^4$")
    ax2.legend(frameon=False)

    fig.suptitle("Blackbody radiation: the photon gas", y=1.05, fontsize=13)
    fig.savefig("fig6_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig6_1.png")


if __name__ == "__main__":
    main()
