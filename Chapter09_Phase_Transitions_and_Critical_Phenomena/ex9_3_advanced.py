#!/usr/bin/env python3
"""
Example 9.3 (advanced, research-grade) -- Finite-size scaling of the 2-D Ising
model: locating T_c with the Binder cumulant and extracting critical exponents.

A finite lattice has no true singularity, but the way its properties depend on size
L encodes the critical behaviour.  Two tools are used here:

  * The Binder cumulant  U_L = 1 - <m^4> / (3 <m^2>^2)  is dimensionless and, near
    criticality, independent of L exactly at T_c -- so curves for different L cross
    at T_c, a size-independent estimator superior to a shifting peak.
  * Finite-size scaling: at T_c the susceptibility peak grows as chi ~ L^{gamma/nu}
    and the magnetization falls as m ~ L^{-beta/nu}.

The exact 2-D Ising exponents are beta/nu = 1/8 = 0.125, gamma/nu = 7/4 = 1.75,
nu = 1, and T_c = 2 / ln(1+sqrt2) = 2.26919.  This script simulates several sizes,
locates T_c from the Binder crossing, extracts the exponent ratios, and collapses
the susceptibility data.  numpy/scipy/matplotlib only; fixed seed.
"""

import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260723)
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


def neighbour_sum(s):
    return (np.roll(s, 1, 0) + np.roll(s, -1, 0)
            + np.roll(s, 1, 1) + np.roll(s, -1, 1))


def sweep(s, T, mask):
    dE = 2.0 * s * neighbour_sum(s)
    accept = (dE < 0) | (RNG.random(s.shape) < np.exp(-dE / T))
    s[accept & mask] *= -1


def simulate(L, T, n_equil=2500, n_meas=6000):
    s = RNG.choice([-1, 1], size=(L, L)).astype(np.int8)
    idx = np.indices((L, L)).sum(0) % 2
    even, odd = (idx == 0), (idx == 1)
    for _ in range(n_equil):
        sweep(s, T, even); sweep(s, T, odd)
    m2s, m4s = [], []
    absm = []
    for _ in range(n_meas):
        sweep(s, T, even); sweep(s, T, odd)
        M = s.sum()
        m2s.append(M * M); m4s.append(M ** 4); absm.append(abs(M))
    N = L * L
    m2 = np.mean(m2s); m4 = np.mean(m4s)
    U = 1.0 - m4 / (3.0 * m2 ** 2)                     # Binder cumulant
    chi = (m2 - np.mean(absm) ** 2) / (N * T)
    m = np.mean(absm) / N
    return U, chi, m


def main():
    print("=" * 72)
    print("Example 9.3  Finite-size scaling of the 2-D Ising model")
    print("=" * 72)
    print(f"\nExact: T_c = {TC:.5f}, beta/nu = 0.125, gamma/nu = 1.75, nu = 1")

    Ls = [8, 16, 24, 32]
    Ts = np.linspace(2.05, 2.50, 13)
    data = {}
    for L in Ls:
        print(f"  simulating L = {L} ...")
        U, chi, m = [], [], []
        for T in Ts:
            u, x, mm = simulate(L, T)
            U.append(u); chi.append(x); m.append(mm)
        data[L] = (np.array(U), np.array(chi), np.array(m))

    # --- Binder crossing near T_c --------------------------------------
    # locate where the two largest lattices' Binder curves cross
    U_big, U_small = data[Ls[-1]][0], data[Ls[-2]][0]
    diff = U_big - U_small
    sign_change = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change):
        i = sign_change[0]
        # linear interpolation of the crossing temperature
        Tcross = Ts[i] - diff[i] * (Ts[i+1]-Ts[i]) / (diff[i+1]-diff[i])
    else:
        Tcross = Ts[np.argmin(np.abs(diff))]
    print(f"\nBinder crossing T_c estimate = {Tcross:.4f} (exact {TC:.4f})")

    # --- exponent ratios from finite-size scaling ----------------------
    chi_peak = np.array([data[L][1].max() for L in Ls])
    gamma_nu = np.polyfit(np.log(Ls), np.log(chi_peak), 1)[0]
    # magnetization interpolated to exactly T_c for each size
    m_at_Tc = np.array([np.interp(TC, Ts, data[L][2]) for L in Ls])
    beta_nu = -np.polyfit(np.log(Ls), np.log(m_at_Tc), 1)[0]
    print(f"chi_peak ~ L^(gamma/nu):  gamma/nu = {gamma_nu:.3f} (exact 1.75)")
    print(f"m(T_c)   ~ L^(-beta/nu):  beta/nu  = {beta_nu:.3f} (exact 0.125)")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)

    # (a) Binder cumulant crossing
    ax = axes[0, 0]
    for L in Ls:
        ax.plot(Ts, data[L][0], "o-", ms=4, lw=1.5, label=f"L={L}")
    ax.axvline(TC, color="k", ls=":", lw=1.5, label="$T_c$ exact")
    ax.set_xlabel("temperature  $T$"); ax.set_ylabel(r"Binder cumulant  $U_L$")
    ax.set_title("(a)  Binder crossing locates $T_c$")
    ax.legend(frameon=False, fontsize=8)

    # (b) chi_peak vs L
    ax = axes[0, 1]
    ax.loglog(Ls, chi_peak, "o", color="#2c3e50", ms=8, label="peak $\\chi$")
    ax.loglog(Ls, chi_peak[0] * (np.array(Ls) / Ls[0]) ** 1.75, "-",
              color="#c0392b", lw=2, label=r"$L^{7/4}$")
    ax.set_xlabel("size  $L$"); ax.set_ylabel(r"$\chi_{\max}$")
    ax.set_title(rf"(b)  $\gamma/\nu={gamma_nu:.2f}$ (exact 1.75)")
    ax.legend(frameon=False)

    # (c) m(Tc) vs L
    ax = axes[1, 0]
    ax.loglog(Ls, m_at_Tc, "o", color="#2c3e50", ms=8, label="$m(T_c)$")
    ax.loglog(Ls, m_at_Tc[0] * (np.array(Ls) / Ls[0]) ** (-0.125), "-",
              color="#c0392b", lw=2, label=r"$L^{-1/8}$")
    ax.set_xlabel("size  $L$"); ax.set_ylabel(r"$m(T_c)$")
    ax.set_title(rf"(c)  $\beta/\nu={beta_nu:.3f}$ (exact 0.125)")
    ax.legend(frameon=False)

    # (d) data collapse of susceptibility: chi L^{-gamma/nu} vs (T-Tc)L^{1/nu}
    ax = axes[1, 1]
    for L in Ls:
        x = (Ts - TC) * L                              # nu = 1
        y = data[L][1] * L ** (-1.75)
        ax.plot(x, y, "o-", ms=4, lw=1.2, label=f"L={L}")
    ax.set_xlabel(r"$(T-T_c)\,L^{1/\nu}$")
    ax.set_ylabel(r"$\chi\,L^{-\gamma/\nu}$")
    ax.set_title("(d)  finite-size scaling collapse")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Finite-size scaling and critical exponents of the 2-D Ising "
                 "model", y=1.03, fontsize=13)
    fig.savefig("fig9_3.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig9_3.png")


if __name__ == "__main__":
    main()
