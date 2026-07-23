#!/usr/bin/env python3
"""
Example 5.3 (advanced, research-grade) -- Hindered internal rotation by matrix
diagonalization, bridging the free rotor and the torsional oscillator.

Internal rotation of one molecular group against another (for example a methyl
top) is governed by the one-dimensional Hamiltonian

    H = -B d^2/dphi^2 + (V0/2) (1 - cos(n phi)),

where B = hbar^2 / 2I_red is the internal rotational constant, n the periodicity
of the barrier (3 for a methyl top), and V0 the barrier height.  We diagonalize H
in the plane-wave (Fourier) basis {exp(i m phi)}, in which the kinetic term is
diagonal (B m^2) and the cosine couples m to m +- n with amplitude -V0/4.

The eigenvalues interpolate between two analytic limits, both used here as
VERIFICATION:

  * Free rotor (V0 -> 0):   E_m = B m^2, doubly degenerate for m != 0.
  * High barrier (V0 large): low levels approach a harmonic torsional oscillator
    of quantum  hbar*omega = n sqrt(B V0),  so spacings -> n sqrt(B V0).

The campaign also checks convergence with basis size and computes the partition
function and heat capacity, showing the crossover from torsional-oscillator
behaviour (C -> R) at low T and high barrier to free-internal-rotor behaviour
(C -> R/2) at high T.  numpy/scipy/matplotlib only; B = 1 sets the energy unit.
"""

import numpy as np
import matplotlib.pyplot as plt

R = 8.314462618
B = 1.0                       # internal rotational constant sets the energy unit


def hindered_eigenvalues(V0, n=3, M=60):
    """Eigenvalues of H in the Fourier basis m = -M..M (units of B)."""
    m = np.arange(-M, M + 1)
    H = np.diag(B * m.astype(float) ** 2 + V0 / 2.0)      # kinetic + constant
    # cosine coupling: -(V0/4) between m and m +- n
    for i, mi in enumerate(m):
        for j, mj in enumerate(m):
            if abs(mi - mj) == n:
                H[i, j] -= V0 / 4.0
    return np.linalg.eigvalsh(H)                           # ascending order


def partition_thermo(evals, T):
    """Z, U, C from a list of eigenvalues at temperature T (energy unit B)."""
    e = evals - evals[0]                                  # shift ground state to 0
    w = np.exp(-e / T)
    Z = w.sum()
    U = (e * w).sum() / Z
    E2 = (e ** 2 * w).sum() / Z
    C = (E2 - U ** 2) / T ** 2
    return Z, U, C


def main():
    print("=" * 72)
    print("Example 5.3  Hindered internal rotor by matrix diagonalization")
    print("=" * 72)

    n = 3                                                 # methyl-top periodicity

    # ---- (1) Free-rotor limit: eigenvalues -> B m^2 --------------------
    M0 = 40
    ev0 = hindered_eigenvalues(0.0, n, M0)
    free = np.sort(B * np.arange(-M0, M0 + 1).astype(float) ** 2)  # exact spectrum
    err_free = np.max(np.abs(ev0 - free)[:20])
    print(f"\n(1) Free rotor (V0=0): max|E_num - B m^2| over lowest 20 levels "
          f"= {err_free:.2e}")

    # ---- (2) Basis-size convergence -----------------------------------
    print("\n(2) Convergence of the lowest 5 levels with basis size (V0=50 B):")
    print(f"    {'M':>4} " + " ".join(f"{'E'+str(k):>10}" for k in range(5)))
    prev = None
    for M in [10, 20, 40, 80]:
        ev = hindered_eigenvalues(50.0, n, M)[:5]
        line = f"    {M:>4} " + " ".join(f"{e:>10.5f}" for e in ev)
        if prev is not None:
            line += f"   dmax={np.max(np.abs(ev-prev)):.1e}"
        print(line)
        prev = ev

    # ---- (3) High-barrier -> harmonic torsion -------------------------
    print("\n(3) High-barrier limit: torsional spacing -> hbar*omega = n sqrt(B V0)")
    print("    (each torsional level is an n-fold near-degenerate tunneling manifold)")
    print(f"    {'V0/B':>7} {'spacing (num)':>14} {'n sqrt(B V0)':>13} {'ratio':>8}")
    for V0 in [200, 800, 3200, 12800]:
        ev = hindered_eigenvalues(V0, n, M=90)
        spacing = ev[3:6].mean() - ev[0:3].mean()      # manifold-center spacing
        harmonic = n * np.sqrt(B * V0)
        print(f"    {V0:>7.0f} {spacing:>14.5f} {harmonic:>13.5f} "
              f"{spacing/harmonic:>8.4f}")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), constrained_layout=True)

    # (a) energy-level correlation diagram vs barrier
    ax = axes[0, 0]
    V0s = np.linspace(0, 60, 60)
    levels = np.array([hindered_eigenvalues(v, n, M=60)[:8] for v in V0s])
    for k in range(8):
        ax.plot(V0s, levels[:, k] - levels[:, 0], color="#2c3e50", lw=1.3)
    ax.set_xlabel(r"barrier height  $V_0/B$")
    ax.set_ylabel(r"$(E_k-E_0)/B$")
    ax.set_title("(a)  levels: free rotor $\\to$ torsional pairs")

    # (b) basis-size convergence of E4
    ax = axes[0, 1]
    Ms = np.arange(6, 60, 2)
    ref = hindered_eigenvalues(50.0, n, M=120)[4]
    errs = [abs(hindered_eigenvalues(50.0, n, M)[4] - ref) + 1e-16 for M in Ms]
    ax.semilogy(Ms, errs, "o-", color="#2c3e50", ms=4)
    ax.set_xlabel("basis half-size  $M$")
    ax.set_ylabel(r"$|E_4(M)-E_4^{\rm ref}|/B$")
    ax.set_title("(b)  spectral convergence")

    # (c) high-barrier spacing vs harmonic prediction
    ax = axes[1, 0]
    V0g = np.linspace(100, 6000, 50)
    spac = []
    for v in V0g:
        ev = hindered_eigenvalues(v, n, M=90)
        spac.append(ev[3:6].mean() - ev[0:3].mean())   # torsional manifold spacing
    ax.plot(np.sqrt(V0g), spac, "o", color="#2c3e50", ms=3, label="numerical")
    ax.plot(np.sqrt(V0g), n * np.sqrt(B * V0g), "-", color="#c0392b", lw=2,
            label=r"$n\sqrt{B V_0}$")
    ax.set_xlabel(r"$\sqrt{V_0/B}$")
    ax.set_ylabel(r"level spacing $/B$")
    ax.set_title("(c)  high barrier $\\to$ harmonic torsion")
    ax.legend(frameon=False)

    # (d) heat capacity crossover for several barriers
    ax = axes[1, 1]
    T = np.linspace(0.2, 60, 300)
    for V0, c in [(0.0, "#95a5a6"), (20.0, "#2980b9"), (100.0, "#c0392b")]:
        ev = hindered_eigenvalues(V0, n, M=80)
        C = np.array([partition_thermo(ev, t)[2] for t in T])
        ax.plot(T, C, color=c, lw=2, label=f"$V_0={V0:.0f}\\,B$")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="free rotor $R/2$")
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5, label="oscillator $R$")
    ax.set_xlabel("temperature  $T$  (units $B/k_B$)")
    ax.set_ylabel(r"$C/R$")
    ax.set_title("(d)  heat-capacity crossover")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Hindered internal rotation: eigenvalues, limits and "
                 "thermodynamics", y=1.03, fontsize=13)
    fig.savefig("fig5_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig5_3.png")


if __name__ == "__main__":
    main()
