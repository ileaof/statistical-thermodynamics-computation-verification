#!/usr/bin/env python3
"""
Example 3.3 (advanced, research-grade) -- Thermodynamics of a quantum harmonic
oscillator from the partition function by DIRECT SUMMATION, with a full
verification campaign.

The oscillator has levels eps_n = (n + 1/2) hbar omega, n = 0,1,2,...  Defining
the characteristic temperature theta = hbar omega / k_B and the reduced variable
x = theta/T = hbar omega / k_B T, the partition function has the closed form

    Z(T) = sum_n exp(-x(n+1/2)) = exp(-x/2)/(1 - exp(-x)) = 1/[2 sinh(x/2)],

with mean energy and heat capacity

    U(T)/(k_B theta) = 1/2 + 1/(e^x - 1),
    C(T)/k_B         = x^2 e^x / (e^x - 1)^2   (the Einstein heat capacity).

We recompute these by summing the series numerically and VERIFY:

  (1) Truncation.  The partition sum converges geometrically; the number of terms
      needed grows with T.  We measure the truncation error vs the number of
      retained levels and confirm the geometric rate.
  (2) Heat capacity three independent ways -- closed form, a centred finite
      difference of U(T), and the fluctuation formula C=(<E^2>-<E>^2)/(k_B T^2)
      evaluated on the summed populations -- must all agree.
  (3) Limits.  High T: U -> k_B T and C -> k_B (equipartition / Dulong-Petit per
      mode).  Low T: U -> hbar omega / 2 (zero-point) and C -> 0 exponentially.

Only numpy, scipy, matplotlib are used.
"""

import numpy as np
import matplotlib.pyplot as plt

# Reduced units: hbar*omega = 1, k_B = 1, so theta = 1 and T is in units of theta.
HW = 1.0
kB = 1.0


def levels(n_max):
    """Energies eps_n = (n + 1/2) hbar omega for n = 0..n_max."""
    n = np.arange(n_max + 1)
    return (n + 0.5) * HW


def partition_sum(T, n_max):
    """Z, mean energy U and mean-square energy <E^2> by direct summation."""
    e = levels(n_max)
    w = np.exp(-e / (kB * T))                 # unnormalised Boltzmann weights
    Z = w.sum()
    U = (e * w).sum() / Z
    E2 = (e ** 2 * w).sum() / Z
    return Z, U, E2


# --- closed forms ---------------------------------------------------------
def Z_closed(T):
    x = HW / (kB * T)
    return 1.0 / (2.0 * np.sinh(x / 2.0))


def U_closed(T):
    x = HW / (kB * T)
    return HW * (0.5 + 1.0 / (np.exp(x) - 1.0))


def C_closed(T):
    x = HW / (kB * T)
    ex = np.exp(x)
    return kB * x ** 2 * ex / (ex - 1.0) ** 2


def main():
    print("=" * 72)
    print("Example 3.3  Harmonic-oscillator partition function by summation")
    print("=" * 72)

    # ---- (1) Truncation convergence at several temperatures ------------
    print("\n(1) Truncation error of Z (|Z_sum - Z_closed|) vs number of levels:")
    print(f"    {'n_max':>6}", end="")
    Ts = [0.3, 1.0, 3.0]
    for T in Ts:
        print(f"  T={T:>4.1f}", end="")
    print()
    nmaxes = [2, 5, 10, 20, 40, 80]
    for nm in nmaxes:
        row = f"    {nm:>6}"
        for T in Ts:
            Zs, _, _ = partition_sum(T, nm)
            row += f"  {abs(Zs - Z_closed(T)):>7.1e}"
        print(row)

    # ---- (2) Heat capacity three ways ----------------------------------
    print("\n(2) Heat capacity three independent ways:")
    print(f"    {'T/theta':>8} {'C closed':>11} {'C fluct':>11} "
          f"{'C fin-diff':>11} {'max|diff|':>11}")
    n_max = 200                                # plenty for convergence
    for T in [0.2, 0.5, 1.0, 2.0, 5.0]:
        _, U, E2 = partition_sum(T, n_max)
        C_fluct = (E2 - U ** 2) / (kB * T ** 2)
        dT = 1e-5
        _, Up, _ = partition_sum(T + dT, n_max)
        _, Um, _ = partition_sum(T - dT, n_max)
        C_fd = (Up - Um) / (2 * dT)
        Cc = C_closed(T)
        mx = max(abs(C_fluct - Cc), abs(C_fd - Cc))
        print(f"    {T:>8.2f} {Cc:>11.6f} {C_fluct:>11.6f} {C_fd:>11.6f} "
              f"{mx:>11.2e}")

    # ---- (3) Limits ----------------------------------------------------
    print("\n(3) Limiting behaviour:")
    for T in [0.1, 10.0, 100.0]:
        _, U, _ = partition_sum(T, 2000 if T > 50 else 400)
        print(f"    T={T:>6.1f}:  U={U:8.4f} (hi-T limit T={T:.1f}),  "
              f"C={C_closed(T):.5f} (hi-T limit 1)")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), constrained_layout=True)

    T = np.linspace(0.05, 4.0, 400)

    # (a) heat capacity vs T with limits
    ax = axes[0, 0]
    ax.plot(T, C_closed(T), color="#2c3e50", lw=2, label="closed form")
    Csum = np.array([(lambda r: (r[2]-r[1]**2)/(kB*t**2))(partition_sum(t, 300))
                     for t in T])
    ax.plot(T, Csum, "o", color="#c0392b", ms=3, markevery=12,
            label="fluctuation (summed)")
    ax.axhline(1.0, color="#27ae60", ls="--", lw=1.5, label="Dulong-Petit $C=k_B$")
    ax.set_xlabel(r"$T/\theta$"); ax.set_ylabel(r"$C/k_B$")
    ax.set_title("(a)  heat capacity freezes out"); ax.legend(frameon=False, fontsize=9)

    # (b) truncation error vs n_max (geometric convergence)
    ax = axes[0, 1]
    nm = np.arange(1, 40)
    for T0, c in [(0.5, "#2980b9"), (1.0, "#2c3e50"), (2.0, "#c0392b")]:
        errs = [abs(partition_sum(T0, int(k))[0] - Z_closed(T0)) for k in nm]
        ax.semilogy(nm, errs, "-", color=c, lw=1.8, label=f"$T/\\theta={T0}$")
    ax.set_xlabel("levels retained  $n_{\\max}$")
    ax.set_ylabel(r"$|Z_{\rm sum}-Z_{\rm exact}|$")
    ax.set_title("(b)  geometric truncation convergence")
    ax.legend(frameon=False)

    # (c) mean energy vs T with zero-point and equipartition limits
    ax = axes[1, 0]
    ax.plot(T, U_closed(T), color="#2c3e50", lw=2, label="closed form")
    ax.plot(T, [partition_sum(t, 300)[1] for t in T], "o", color="#c0392b",
            ms=3, markevery=12, label="summed")
    ax.axhline(0.5, color="#27ae60", ls="--", lw=1.5, label=r"zero-point $\hbar\omega/2$")
    ax.plot(T, T, "k:", lw=1.5, label=r"equipartition $k_BT$")
    ax.set_ylim(0, 4)
    ax.set_xlabel(r"$T/\theta$"); ax.set_ylabel(r"$U/\hbar\omega$")
    ax.set_title("(c)  energy: zero-point to equipartition")
    ax.legend(frameon=False, fontsize=9)

    # (d) agreement of the three C methods (verification)
    ax = axes[1, 1]
    Tg = np.linspace(0.15, 4.0, 120)
    err_fluct, err_fd = [], []
    for t in Tg:
        _, U, E2 = partition_sum(t, 300)
        err_fluct.append(abs((E2 - U**2)/(kB*t**2) - C_closed(t)))
        dT = 1e-5
        C_fd = (partition_sum(t+dT,300)[1]-partition_sum(t-dT,300)[1])/(2*dT)
        err_fd.append(abs(C_fd - C_closed(t)))
    ax.semilogy(Tg, np.array(err_fluct)+1e-18, color="#2980b9", lw=2,
                label="fluctuation vs closed")
    ax.semilogy(Tg, np.array(err_fd)+1e-18, color="#c0392b", lw=2,
                label="finite-diff vs closed")
    ax.set_xlabel(r"$T/\theta$"); ax.set_ylabel("absolute error in $C/k_B$")
    ax.set_title("(d)  three methods agree")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Harmonic oscillator: thermodynamics by numerical summation "
                 "and verification", y=1.03, fontsize=13)
    fig.savefig("fig3_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig3_3.png")


if __name__ == "__main__":
    main()
