#!/usr/bin/env python3
"""
Example 5.1 (analytical) -- The heat-capacity staircase of a diatomic gas.

A diatomic molecule stores energy in translation, rotation, and vibration.  Each
mode contributes to the molar heat capacity only once the temperature exceeds its
characteristic temperature, producing the famous staircase

    low T:   C_V = (3/2) R          (translation only)
    T > th_rot:  C_V -> (5/2) R      (rotation unlocked, 2 extra half-R)
    T > th_vib:  C_V -> (7/2) R      (vibration unlocked, +R)

Rotation contributes through the exact sum over levels eps_J = k_B th_rot J(J+1)
with degeneracy (2J+1); vibration contributes the Einstein form; translation is
the constant (3/2)R.  This script assembles C_V(T) for a real molecule and
VERIFIES the plateau values against equipartition.  numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt

R = 8.314462618        # J/(mol K)


def rot_thermo(T, th_rot, sigma=1, Jmax=400):
    """Exact rotational energy and heat capacity per mole by summation."""
    J = np.arange(0, Jmax + 1)
    g = (2 * J + 1)                                   # rotational degeneracy
    T = np.atleast_1d(T).astype(float)
    Cv = np.empty_like(T); U = np.empty_like(T)
    for i, t in enumerate(T):
        e = th_rot * J * (J + 1)                      # level energy in kelvin
        w = g * np.exp(-e / t)
        Z = w.sum()
        Eavg = (e * w).sum() / Z                      # <E> in kelvin
        E2 = (e ** 2 * w).sum() / Z
        U[i] = R * Eavg
        Cv[i] = R * (E2 - Eavg ** 2) / t ** 2         # fluctuation formula
    return U, Cv


def vib_Cv(T, th_vib):
    """Einstein vibrational heat capacity per mole."""
    x = th_vib / np.asarray(T, float)
    emx = np.exp(-x)                                   # stable for large x
    return R * x ** 2 * emx / (1.0 - emx) ** 2


def main():
    print("=" * 70)
    print("Example 5.1  Heat-capacity staircase of a diatomic gas")
    print("=" * 70)

    # HCl: th_rot = 15.2 K, th_vib = 4300 K, sigma = 1
    th_rot, th_vib, sigma = 15.2, 4300.0, 1
    print(f"\nHCl:  th_rot = {th_rot} K,  th_vib = {th_vib} K")

    print(f"\n{'T (K)':>8} {'Cv_tr/R':>8} {'Cv_rot/R':>9} {'Cv_vib/R':>9} "
          f"{'Cv_tot/R':>9}")
    for T in [5, 50, 300, 1000, 3000, 8000]:
        _, cr = rot_thermo(T, th_rot, sigma)
        cv = vib_Cv(T, th_vib)
        tot = 1.5 * R + cr[0] + cv
        print(f"{T:>8.0f} {1.5:>8.3f} {cr[0]/R:>9.3f} {cv/R:>9.3f} "
              f"{tot/R:>9.3f}")

    # Verify plateau values
    _, cr_lo = rot_thermo(1.0, th_rot)      # rotation frozen
    _, cr_mid = rot_thermo(300.0, th_rot)   # rotation active, vibration frozen
    cv_hi = vib_Cv(30000.0, th_vib)         # vibration active
    print(f"\nPlateau verification (Cv_tot/R):")
    print(f"  T -> 0        : {(1.5*R + cr_lo[0] + vib_Cv(1.0,th_vib))/R:.4f} "
          f"(expect 1.5 = 3/2)")
    print(f"  th_rot<<T<<th_vib: {(1.5*R + cr_mid[0] + vib_Cv(300.0,th_vib))/R:.4f} "
          f"(expect 2.5 = 5/2)")
    print(f"  T >> th_vib   : {(1.5*R + rot_thermo(30000.0,th_rot)[1][0] + cv_hi)/R:.4f} "
          f"(expect 3.5 = 7/2)")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   constrained_layout=True)

    T = np.logspace(0.3, 4.2, 400)
    _, Cr = rot_thermo(T, th_rot, sigma)
    Cvib = vib_Cv(T, th_vib)
    Ctot = 1.5 * R + Cr + Cvib
    ax1.semilogx(T, np.full_like(T, 1.5), color="#95a5a6", lw=1.5, ls=":",
                 label="translation")
    ax1.semilogx(T, 1.5 + Cr / R, color="#2980b9", lw=1.5, ls="--",
                 label="+ rotation")
    ax1.semilogx(T, Ctot / R, color="#c0392b", lw=2.4, label="total")
    for y in (1.5, 2.5, 3.5):
        ax1.axhline(y, color="k", lw=0.6, alpha=0.3)
    ax1.axvline(th_rot, color="#2980b9", lw=1, ls=":")
    ax1.axvline(th_vib, color="#27ae60", lw=1, ls=":")
    ax1.set_xlabel("temperature  $T$  (K)")
    ax1.set_ylabel(r"$C_V / R$")
    ax1.set_title(r"(a)  HCl: the $3/2 \to 5/2 \to 7/2$ staircase")
    ax1.legend(frameon=False, loc="upper left")

    # rotational contribution alone, showing the small overshoot above R/... = 1
    Tr = np.logspace(-0.3, 2.5, 400)
    _, Cr2 = rot_thermo(Tr, th_rot)
    ax2.semilogx(Tr / th_rot, Cr2 / R, color="#2980b9", lw=2)
    ax2.axhline(1.0, color="#c0392b", ls="--", lw=1.5, label="equipartition $R$")
    ax2.set_xlabel(r"$T/\theta_{\rm rot}$")
    ax2.set_ylabel(r"$C_V^{\rm rot}/R$")
    ax2.set_title("(b)  rotational $C_V$ and its overshoot")
    ax2.legend(frameon=False)

    fig.suptitle("Heat capacity of a diatomic gas: translation, rotation, "
                 "vibration", y=1.05, fontsize=13)
    fig.savefig("fig5_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig5_1.png")


if __name__ == "__main__":
    main()
