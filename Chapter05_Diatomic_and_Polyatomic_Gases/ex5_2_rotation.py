#!/usr/bin/env python3
"""
Example 5.2 (direct numerical) -- The rotational partition function: exact
summation versus the high-temperature (Euler-Maclaurin) approximation.

A linear rotor has levels eps_J = k_B th_rot J(J+1) with degeneracy (2J+1), so

    z_rot = (1/sigma) sum_{J=0}^inf (2J+1) exp(-th_rot J(J+1)/T),

sigma being the symmetry number (2 for homonuclear, 1 for heteronuclear).  For
T >> th_rot the Euler-Maclaurin formula gives the asymptotic series

    z_rot ~ (T/(sigma th_rot)) [ 1 + (1/3)(th_rot/T) + (1/15)(th_rot/T)^2 + ... ],

whose leading term is the classical result T/(sigma th_rot).  This script sums
z_rot exactly, compares it with the leading and Euler-Maclaurin approximations,
and shows that the classical formula fails for light molecules (large th_rot) at
low temperature while the correction terms restore accuracy.

numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt

R = 8.314462618


def z_rot_sum(T, th_rot, sigma=1, Jmax=2000):
    """Exact rotational partition function by summation."""
    J = np.arange(0, Jmax + 1)
    g = (2 * J + 1).astype(float)
    T = np.atleast_1d(T).astype(float)
    out = np.empty_like(T)
    for i, t in enumerate(T):
        out[i] = np.sum(g * np.exp(-th_rot * J * (J + 1) / t)) / sigma
    return out


def z_rot_leading(T, th_rot, sigma=1):
    """Classical leading term T/(sigma th_rot)."""
    return np.asarray(T, float) / (sigma * th_rot)


def z_rot_euler_maclaurin(T, th_rot, sigma=1):
    """High-T asymptotic series to third order in th_rot/T."""
    r = th_rot / np.asarray(T, float)
    return (1.0 / (sigma * r)) * (1 + r / 3 + r ** 2 / 15 + 4 * r ** 3 / 315)


def main():
    print("=" * 72)
    print("Example 5.2  Rotational partition function: sum vs Euler-Maclaurin")
    print("=" * 72)

    molecules = {
        "H2 (th_rot=87.6 K, s=2)":  (87.6, 2),
        "HCl(th_rot=15.2 K, s=1)":  (15.2, 1),
        "CO (th_rot=2.78 K, s=1)":  (2.78, 1),
    }

    for name, (th, sig) in molecules.items():
        print(f"\n{name}")
        print(f"  {'T/th_rot':>9} {'z (exact)':>11} {'z (leading)':>12} "
              f"{'z (E-M)':>11} {'lead %err':>10} {'E-M %err':>10}")
        for ratio in [0.5, 1.0, 2.0, 5.0, 10.0]:
            T = ratio * th
            ze = z_rot_sum(T, th, sig)[0]
            zl = z_rot_leading(T, th, sig)
            zem = z_rot_euler_maclaurin(T, th, sig)
            print(f"  {ratio:>9.1f} {ze:>11.4f} {zl:>12.4f} {zem:>11.4f} "
                  f"{100*(zl-ze)/ze:>10.3f} {100*(zem-ze)/ze:>10.4f}")

    # Verify: leading-term relative error ~ th_rot/(3T)
    th, sig = 87.6, 2
    Tg = np.linspace(3 * th, 30 * th, 40)
    err_lead = (z_rot_leading(Tg, th, sig) - z_rot_sum(Tg, th, sig)) \
        / z_rot_sum(Tg, th, sig)
    print(f"\nLeading-term error vs th_rot/(3T) for H2:")
    print(f"  at T=10 th_rot: measured {err_lead[np.argmin(abs(Tg-10*th))]:+.5f}, "
          f"predicted {-th/(3*10*th):+.5f}")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   constrained_layout=True)

    # (a) exact vs approximations for H2
    th, sig = 87.6, 2
    T = np.linspace(10, 900, 300)
    ax1.plot(T, z_rot_sum(T, th, sig), color="#2c3e50", lw=2.4, label="exact sum")
    ax1.plot(T, z_rot_leading(T, th, sig), color="#c0392b", lw=1.8, ls="--",
             label=r"leading  $T/\sigma\theta_{\rm rot}$")
    ax1.plot(T, z_rot_euler_maclaurin(T, th, sig), color="#27ae60", lw=1.8,
             ls=":", label="Euler-Maclaurin")
    ax1.set_xlabel("temperature  $T$  (K)")
    ax1.set_ylabel(r"$z_{\rm rot}$")
    ax1.set_title(r"(a)  H$_2$: classical formula fails at low $T$")
    ax1.legend(frameon=False)

    # (b) relative error of the two approximations vs T/th_rot
    ratio = np.linspace(0.5, 12, 200)
    for th, sig, c, lab in [(87.6, 2, "#2980b9", "H$_2$"),
                            (2.78, 1, "#c0392b", "CO")]:
        T = ratio * th
        ze = z_rot_sum(T, th, sig)
        ax2.semilogy(ratio, np.abs(z_rot_leading(T, th, sig) - ze) / ze,
                     color=c, lw=2, label=f"leading, {lab}")
        ax2.semilogy(ratio, np.abs(z_rot_euler_maclaurin(T, th, sig) - ze) / ze,
                     color=c, lw=2, ls=":", label=f"E-M, {lab}")
    ax2.set_xlabel(r"$T/\theta_{\rm rot}$")
    ax2.set_ylabel("relative error")
    ax2.set_title("(b)  Euler-Maclaurin restores accuracy")
    ax2.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Rotational partition function: exact sum vs high-$T$ expansion",
                 y=1.05, fontsize=13)
    fig.savefig("fig5_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig5_2.png")


if __name__ == "__main__":
    main()
