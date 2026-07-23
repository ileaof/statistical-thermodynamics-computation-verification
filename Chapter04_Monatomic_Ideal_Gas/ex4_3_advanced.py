#!/usr/bin/env python3
"""
Example 4.3 (advanced, research-grade) -- Chemical potential, extensivity and the
Gibbs paradox for the monatomic ideal gas, with a verification campaign.

The Gibbs correction Z = z^N / N! (indistinguishability) is what makes the
entropy EXTENSIVE and resolves the Gibbs paradox.  Writing z = V/Lambda^3, the
Helmholtz energy and chemical potential are

    F = -N k_B T [ ln( V / (N Lambda^3) ) + 1 ],
    mu = (dF/dN)_{T,V} = -k_B T ln( V / (N Lambda^3) ) = k_B T ln(n Lambda^3),

and the Sackur-Tetrode entropy S = N k_B [ ln(V/(N Lambda^3)) + 5/2 ].  This
script mounts four verifications:

  (1) Extensivity.  With the 1/N! factor S/N depends only on the density and is
      constant along a line of fixed n = N/V; WITHOUT it S/N grows with system
      size -- the non-extensive, paradoxical behaviour.
  (2) Entropy of mixing.  Mixing two DISTINCT gases gives Delta S = 2 N k_B ln2;
      mixing two IDENTICAL gases gives zero with the Gibbs factor but a spurious
      2 N k_B ln2 without it -- the Gibbs paradox and its resolution.
  (3) Chemical potential.  mu from a centred finite difference of F(N) must match
      the closed form.
  (4) Thermodynamic identities.  C_p - C_V = N k_B and gamma = C_p/C_V = 5/3.

numpy/scipy/matplotlib only; fully deterministic.
"""

import numpy as np
import matplotlib.pyplot as plt

h  = 6.62607015e-34
kB = 1.380649e-23
u  = 1.66053906660e-27
M_AR = 39.948 * u


def Lambda(T, m=M_AR):
    return h / np.sqrt(2.0 * np.pi * m * kB * T)


def entropy(N, V, T, gibbs=True):
    """Entropy; gibbs=True gives Sackur-Tetrode, False the non-extensive form."""
    lam3 = Lambda(T) ** 3
    if gibbs:
        return N * kB * (np.log(V / (N * lam3)) + 2.5)     # extensive
    return N * kB * (np.log(V / lam3) + 1.5)               # Boltzmann, non-extensive


def helmholtz(N, V, T):
    """F = -kT ln Z with Z = z^N/N!, Stirling for ln N!."""
    lam3 = Lambda(T) ** 3
    return -N * kB * T * (np.log(V / (N * lam3)) + 1.0)


def mu_closed(N, V, T):
    return -kB * T * np.log(V / (N * Lambda(T) ** 3))


def main():
    print("=" * 72)
    print("Example 4.3  Chemical potential, extensivity and the Gibbs paradox")
    print("=" * 72)

    T = 300.0
    n0 = 2.5e25                     # fixed number density (1/m^3), ~1 bar, 300 K

    # ---- (1) Extensivity along fixed density --------------------------
    print("\n(1) Entropy per particle S/N along fixed density n = N/V:")
    print(f"    {'N':>12} {'S/N (Gibbs)':>14} {'S/N (no 1/N!)':>16}")
    for N in [1e20, 1e22, 1e24, 1e26]:
        V = N / n0
        s_g = entropy(N, V, T, True) / (N * kB)
        s_b = entropy(N, V, T, False) / (N * kB)
        print(f"    {N:>12.0e} {s_g:>14.6f} {s_b:>16.6f}")
    print("    -> Gibbs: constant (extensive);  no 1/N!: grows with N (paradox)")

    # ---- (2) Entropy of mixing ----------------------------------------
    print("\n(2) Entropy of mixing two equal volumes (N each, V each -> 2V):")
    N = 1e23
    V = N / n0
    # distinct gases: each expands V -> 2V
    dS_distinct = (entropy(N, 2*V, T, True) - entropy(N, V, T, True)) * 2 / (N*kB)
    # identical gases, Gibbs: final is 2N in 2V at same density
    S_before = 2 * entropy(N, V, T, True)
    S_after_gibbs = entropy(2*N, 2*V, T, True)
    dS_identical_gibbs = (S_after_gibbs - S_before) / (N * kB)
    # identical gases WITHOUT 1/N!: spurious
    S_before_b = 2 * entropy(N, V, T, False)
    S_after_b = entropy(2*N, 2*V, T, False)
    dS_identical_nob = (S_after_b - S_before_b) / (N * kB)
    print(f"    distinct gases:            Delta S/(N k_B) = {dS_distinct:.5f} "
          f"(expect 2 ln2 = {2*np.log(2):.5f})")
    print(f"    identical, with 1/N!:      Delta S/(N k_B) = "
          f"{dS_identical_gibbs:.5e} (expect 0)")
    print(f"    identical, without 1/N!:   Delta S/(N k_B) = "
          f"{dS_identical_nob:.5f} (spurious paradox = 2 ln2)")

    # ---- (3) Chemical potential: finite difference vs closed form -----
    print("\n(3) Chemical potential mu = (dF/dN)_{T,V}:")
    print(f"    {'N':>12} {'mu closed (eV)':>16} {'mu finite-diff':>16} "
          f"{'rel err':>11}")
    for N in [1e22, 1e23, 1e24]:
        V = N / n0
        dN = N * 1e-6
        mu_fd = (helmholtz(N + dN, V, T) - helmholtz(N - dN, V, T)) / (2 * dN)
        mc = mu_closed(N, V, T)
        print(f"    {N:>12.0e} {mc/1.602e-19:>16.6f} {mu_fd/1.602e-19:>16.6f} "
              f"{abs(mu_fd-mc)/abs(mc):>11.2e}")

    # ---- (4) Heat-capacity identities ---------------------------------
    Cv = 1.5 * kB          # per atom
    Cp = Cv + kB
    print("\n(4) Heat-capacity identities (per atom):")
    print(f"    C_V = {Cv/kB:.3f} k_B,  C_p = {Cp/kB:.3f} k_B,  "
          f"C_p - C_V = {(Cp-Cv)/kB:.3f} k_B (expect 1)")
    print(f"    gamma = C_p/C_V = {Cp/Cv:.5f} (expect 5/3 = {5/3:.5f})")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), constrained_layout=True)

    # (a) extensivity
    ax = axes[0, 0]
    Ns = np.logspace(19, 27, 40)
    ax.semilogx(Ns, [entropy(N, N/n0, T, True)/(N*kB) for N in Ns], "o",
                color="#2c3e50", ms=5, label="with $1/N!$ (Sackur-Tetrode)")
    ax.semilogx(Ns, [entropy(N, N/n0, T, False)/(N*kB) for N in Ns], "-",
                color="#c0392b", lw=2, label="without $1/N!$")
    ax.set_xlabel("system size  $N$")
    ax.set_ylabel(r"$S/(N k_B)$")
    ax.set_title("(a)  extensivity requires $1/N!$")
    ax.legend(frameon=False, fontsize=9)

    # (b) entropy of mixing bar chart
    ax = axes[0, 1]
    labels = ["distinct\n(Gibbs)", "identical\n(Gibbs)", "identical\n(no $1/N!$)"]
    vals = [dS_distinct, dS_identical_gibbs, dS_identical_nob]
    colors = ["#2c3e50", "#27ae60", "#c0392b"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(2*np.log(2), color="k", ls=":", lw=1.2, label=r"$2\ln2$")
    ax.set_ylabel(r"$\Delta S_{\rm mix}/(N k_B)$")
    ax.set_title("(b)  Gibbs paradox and its resolution")
    ax.legend(frameon=False)

    # (c) chemical potential vs temperature
    ax = axes[1, 0]
    Tg = np.linspace(100, 1000, 200)
    N = 1e23; V = N / n0
    ax.plot(Tg, [mu_closed(N, V, t)/1.602e-19 for t in Tg], color="#2c3e50",
            lw=2, label="closed form")
    dN = N*1e-6
    mu_fd = [(helmholtz(N+dN, V, t)-helmholtz(N-dN, V, t))/(2*dN)/1.602e-19
             for t in Tg]
    ax.plot(Tg[::12], np.array(mu_fd)[::12], "o", color="#c0392b", ms=5,
            label=r"$\partial F/\partial N$ (finite diff)")
    ax.set_xlabel("temperature  $T$  (K)")
    ax.set_ylabel(r"chemical potential  $\mu$  (eV)")
    ax.set_title("(c)  chemical potential, two ways")
    ax.legend(frameon=False, fontsize=9)

    # (d) chemical-potential verification error vs N
    ax = axes[1, 1]
    Ns = np.logspace(20, 26, 30)
    errs = []
    for N in Ns:
        V = N/n0; dN = N*1e-6
        mu_fd = (helmholtz(N+dN, V, T)-helmholtz(N-dN, V, T))/(2*dN)
        errs.append(abs(mu_fd - mu_closed(N, V, T))/abs(mu_closed(N, V, T)))
    ax.loglog(Ns, errs, "o-", color="#2c3e50", ms=5)
    ax.set_xlabel("system size  $N$")
    ax.set_ylabel(r"$|\mu_{\rm fd}-\mu_{\rm closed}|/|\mu|$")
    ax.set_title("(d)  $\\mu$ agreement (finite-diff floor)")

    fig.suptitle("Monatomic ideal gas: chemical potential, extensivity, "
                 "Gibbs paradox", y=1.03, fontsize=13)
    fig.savefig("fig4_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig4_3.png")


if __name__ == "__main__":
    main()
