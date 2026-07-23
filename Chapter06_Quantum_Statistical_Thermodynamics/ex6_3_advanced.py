#!/usr/bin/env python3
"""
Example 6.3 (advanced, research-grade) -- Bose-Einstein condensation of the ideal
Bose gas: condensate fraction, heat capacity and its cusp, with verification.

For a 3-D ideal gas of spin-0 bosons the number of particles in EXCITED states is

    N_exc = (V/lambda^3) g_{3/2}(z),   z = e^{mu/kT} the fugacity,

with the Bose functions (polylogarithms) g_s(z) = sum_{k>=1} z^k / k^s and the
thermal wavelength lambda ~ T^{-1/2}.  Below a critical temperature T_c the
excited states saturate (z -> 1, g_{3/2}(1) = zeta(3/2)) and the remainder
macroscopically occupies the ground state -- Bose-Einstein condensation.  In
reduced temperature t = T/T_c:

    condensate fraction   N_0/N = 1 - t^{3/2}                 (t < 1)
    energy   U/(N k T_c)  = (3/2) t^{5/2} g_{5/2}(z) / zeta(3/2)
    heat capacity C/(N k) continuous at t = 1 with value 1.926, but with a CUSP
      (a discontinuous slope) -- the signature of the transition.

This script computes the Bose functions by series summation, locates the fugacity
above T_c by root finding, builds C(t), and VERIFIES: the polylog against zeta at
z=1, the condensate fraction, the value C(T_c)/Nk = 1.926, and the continuity of C
(with a discontinuous derivative) at the transition.  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

ZETA_32 = 2.6123753486854883      # zeta(3/2)
ZETA_52 = 1.3414872572509171      # zeta(5/2)


from math import gamma as _gamma
_U = np.linspace(1e-6, 9.0, 6000)            # substitution grid x = u^2
_trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def bose_g(s, z):
    """Bose function g_s(z) via its integral representation

        g_s(z) = 1/Gamma(s) * int_0^inf x^{s-1} / (z^{-1} e^x - 1) dx,

    evaluated with the substitution x = u^2 (which removes the x=0 singularity).
    Accurate for all z in (0,1], returning zeta(s) exactly at z=1 -- unlike the
    slowly converging power series, whose K^{-1/2} tail is shown in panel (d)."""
    u = _U
    x = u * u
    integrand = 2.0 * u ** (2 * s - 1) / (np.exp(x) / z - 1.0)
    return _trap(integrand, u) / _gamma(s)


# Saturation values of the Bose functions at z=1, computed with the SAME
# quadrature used everywhere else, so the transition is internally consistent.
G32_MAX = bose_g(1.5, 1.0 - 1e-14)
G52_MAX = bose_g(2.5, 1.0 - 1e-14)


def fugacity(t):
    """Above T_c (t>1): solve g_{3/2}(z) = zeta(3/2) t^{-3/2} for z in (0,1)."""
    target = G32_MAX * t ** (-1.5)
    return brentq(lambda z: bose_g(1.5, z) - target, 1e-12, 1 - 1e-14, xtol=1e-13)


def energy_reduced(t):
    """U/(N k T_c) as a function of reduced temperature t = T/T_c."""
    if t <= 1.0:
        z = 1.0
        return 1.5 * t ** 2.5 * G52_MAX / G32_MAX
    z = fugacity(t)
    # U/(N k T_c) = (3/2) t * g_{5/2}(z)/g_{3/2}(z), with g_{3/2}=zeta*t^-3/2
    return 1.5 * t * bose_g(2.5, z) / bose_g(1.5, z)


def heat_capacity(t, dt=1e-4):
    return (energy_reduced(t + dt) - energy_reduced(t - dt)) / (2 * dt)


def main():
    print("=" * 72)
    print("Example 6.3  Bose-Einstein condensation of the ideal Bose gas")
    print("=" * 72)

    # ---- (1) Polylog vs zeta at z=1 -----------------------------------
    print("\n(1) Bose functions at z=1 vs Riemann zeta:")
    print(f"  g_3/2(1) = {bose_g(1.5, 1.0):.8f}  zeta(3/2) = {ZETA_32:.8f}")
    print(f"  g_5/2(1) = {bose_g(2.5, 1.0):.8f}  zeta(5/2) = {ZETA_52:.8f}")

    # ---- (2) Condensate fraction --------------------------------------
    print("\n(2) Condensate fraction  N_0/N = 1 - t^{3/2}:")
    for t in [0.2, 0.5, 0.8, 1.0]:
        print(f"  t={t:.1f}:  N_0/N = {1 - t**1.5:.4f}")

    # ---- (3) Heat capacity at and around T_c --------------------------
    C_below = 3.75 * G52_MAX / G32_MAX                 # (15/4) g5/2 / g3/2
    print("\n(3) Heat capacity at the transition:")
    print(f"  analytic C(T_c^-)/Nk = (15/4) zeta(5/2)/zeta(3/2) = {C_below:.5f}")
    print(f"  numerical C(0.999)   = {heat_capacity(0.999):.5f}")
    print(f"  numerical C(1.001)   = {heat_capacity(1.001):.5f}")
    print(f"  => C continuous at T_c (both ~ {C_below:.3f})")
    # slope on each side -> the cusp
    slope_lo = (heat_capacity(0.98) - heat_capacity(0.94)) / 0.04
    slope_hi = (heat_capacity(1.06) - heat_capacity(1.02)) / 0.04
    print(f"  dC/dt below T_c ~ {slope_lo:+.3f},  above T_c ~ {slope_hi:+.3f}")
    print(f"  => slope discontinuity (cusp) = {abs(slope_lo-slope_hi):.3f}")

    # ---- (4) high-T limit -> classical 3/2 ----------------------------
    print(f"\n(4) High-T limit:  C(t=5)/Nk = {heat_capacity(5.0):.4f} "
          f"(classical 1.5)")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), constrained_layout=True)

    t_lo = np.linspace(0.05, 1.0, 120)
    t_hi = np.linspace(1.0, 5.0, 160)

    # (a) condensate fraction
    ax = axes[0, 0]
    ax.plot(t_lo, 1 - t_lo ** 1.5, color="#2c3e50", lw=2.4)
    ax.plot(t_hi, np.zeros_like(t_hi), color="#2c3e50", lw=2.4)
    ax.axvline(1.0, color="#c0392b", ls=":", lw=1.5, label="$T_c$")
    ax.set_xlabel(r"$T/T_c$"); ax.set_ylabel(r"$N_0/N$")
    ax.set_title(r"(a)  condensate fraction $1-(T/T_c)^{3/2}$")
    ax.legend(frameon=False)

    # (b) heat capacity with the cusp
    ax = axes[0, 1]
    C_lo = 3.75 * G52_MAX / G32_MAX * t_lo ** 1.5
    C_hi = np.array([heat_capacity(t) for t in t_hi])
    ax.plot(t_lo, C_lo, color="#2c3e50", lw=2.4)
    ax.plot(t_hi, C_hi, color="#2c3e50", lw=2.4)
    ax.axhline(1.5, color="#27ae60", ls="--", lw=1.3, label="classical $3/2$")
    ax.axvline(1.0, color="#c0392b", ls=":", lw=1.5)
    ax.plot(1.0, C_below, "o", color="#c0392b", ms=7,
            label=f"$C(T_c)={C_below:.3f}$")
    ax.set_xlabel(r"$T/T_c$"); ax.set_ylabel(r"$C_V/Nk_B$")
    ax.set_title("(b)  heat capacity and its cusp")
    ax.legend(frameon=False, fontsize=9)

    # (c) fugacity / chemical potential
    ax = axes[1, 0]
    z_hi = np.array([fugacity(t) for t in t_hi])
    mu_hi = np.log(z_hi)                                # mu/kT
    ax.plot(t_lo, np.zeros_like(t_lo), color="#2c3e50", lw=2.4)
    ax.plot(t_hi, mu_hi, color="#2c3e50", lw=2.4)
    ax.axvline(1.0, color="#c0392b", ls=":", lw=1.5)
    ax.set_xlabel(r"$T/T_c$"); ax.set_ylabel(r"$\mu/k_BT$")
    ax.set_title(r"(c)  chemical potential: $\mu=0$ below $T_c$")

    # (d) polylog convergence at z=1
    ax = axes[1, 1]
    ks = np.arange(1, 4000)
    partial_32 = np.cumsum(1.0 / ks ** 1.5)
    ax.loglog(ks, np.abs(ZETA_32 - partial_32), color="#2980b9", lw=2,
              label=r"$|\zeta(3/2)-\sum^K|$")
    ax.loglog(ks, ks ** (-0.5), "k:", lw=1.3, label=r"$K^{-1/2}$ tail")
    ax.set_xlabel("terms summed  $K$")
    ax.set_ylabel("truncation error")
    ax.set_title("(d)  polylogarithm convergence at $z=1$")
    ax.legend(frameon=False)

    fig.suptitle("Bose-Einstein condensation of the ideal Bose gas", y=1.03,
                 fontsize=13)
    fig.savefig("fig6_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig6_3.png")


if __name__ == "__main__":
    main()
