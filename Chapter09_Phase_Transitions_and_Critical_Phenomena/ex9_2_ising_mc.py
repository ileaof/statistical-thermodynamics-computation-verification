#!/usr/bin/env python3
"""
Example 9.2 (direct numerical) -- Monte Carlo simulation of the two-dimensional
Ising model with the Metropolis algorithm.

The Ising model places spins s = +-1 on a lattice with energy
E = -J sum_<ij> s_i s_j (nearest neighbours).  In two dimensions it has a
continuous phase transition at the exact Onsager temperature

    k_B T_c / J = 2 / ln(1 + sqrt(2)) = 2.26919,

below which a spontaneous magnetization appears.  This script simulates an L x L
lattice with periodic boundaries using a vectorised checkerboard Metropolis update,
measures the magnetization, susceptibility and heat capacity as functions of
temperature, and VERIFIES the transition against Onsager's exact results: the
critical temperature and the magnetization m = (1 - sinh^{-4}(2/T))^{1/8}.

numpy/scipy/matplotlib only; fixed seed.
"""

import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20260723)
TC_ONSAGER = 2.0 / np.log(1.0 + np.sqrt(2.0))          # 2.26919


def neighbour_sum(s):
    return (np.roll(s, 1, 0) + np.roll(s, -1, 0)
            + np.roll(s, 1, 1) + np.roll(s, -1, 1))


def metropolis_sweep(s, T, mask):
    """One checkerboard half-sweep: update the sites selected by mask."""
    dE = 2.0 * s * neighbour_sum(s)                    # J = 1
    accept = (dE < 0) | (RNG.random(s.shape) < np.exp(-dE / T))
    flip = accept & mask
    s[flip] *= -1
    return s


def simulate(L, T, n_equil=2000, n_meas=4000):
    """Return <|m|>, <m^2>, <m^4>, energy, susceptibility, heat capacity."""
    s = RNG.choice([-1, 1], size=(L, L)).astype(np.int8)
    idx = np.indices((L, L)).sum(0) % 2
    even, odd = (idx == 0), (idx == 1)
    for _ in range(n_equil):
        metropolis_sweep(s, T, even); metropolis_sweep(s, T, odd)
    mags, ens = [], []
    for _ in range(n_meas):
        metropolis_sweep(s, T, even); metropolis_sweep(s, T, odd)
        mags.append(s.sum())
        ens.append(-(s * neighbour_sum(s)).sum() / 2.0)
    mags = np.array(mags, float); ens = np.array(ens, float)
    N = L * L
    m_abs = np.abs(mags).mean() / N
    m2 = (mags ** 2).mean() / N ** 2
    m4 = (mags ** 4).mean() / N ** 4
    chi = (np.mean(mags ** 2) - np.mean(np.abs(mags)) ** 2) / (N * T)
    C = (np.mean(ens ** 2) - np.mean(ens) ** 2) / (N * T ** 2)
    return m_abs, m2, m4, ens.mean() / N, chi, C


def onsager_m(T):
    """Exact spontaneous magnetization of the 2-D Ising model (T < T_c)."""
    x = np.sinh(2.0 / T)
    return np.where(T < TC_ONSAGER, (1.0 - x ** (-4)) ** 0.125, 0.0)


def main():
    print("=" * 70)
    print("Example 9.2  Monte Carlo of the 2-D Ising model (Metropolis)")
    print("=" * 70)
    print(f"\nOnsager T_c = {TC_ONSAGER:.5f}  (J = k_B = 1)")

    L = 32
    Ts = np.linspace(1.6, 3.4, 28)
    print(f"\nL = {L}, sweeping {len(Ts)} temperatures...")
    m_abs, chi, C, ener = [], [], [], []
    for T in Ts:
        ma, m2, m4, e, x, c = simulate(L, T)
        m_abs.append(ma); chi.append(x); C.append(c); ener.append(e)
    m_abs, chi, C, ener = map(np.array, (m_abs, chi, C, ener))

    # --- verify Tc from the susceptibility peak ------------------------
    T_peak = Ts[np.argmax(chi)]
    print(f"\nT of susceptibility peak = {T_peak:.3f} "
          f"(Onsager {TC_ONSAGER:.3f})")
    T_cpeak = Ts[np.argmax(C)]
    print(f"T of heat-capacity peak  = {T_cpeak:.3f}")

    # --- compare magnetization with Onsager below Tc -------------------
    print(f"\n{'T':>6} {'m (MC)':>9} {'m (Onsager)':>12}")
    for T, ma in zip(Ts, m_abs):
        if T < TC_ONSAGER:
            print(f"{T:>6.2f} {ma:>9.4f} {float(onsager_m(T)):>12.4f}")

    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    Tsmooth = np.linspace(1.6, TC_ONSAGER, 200)
    axes[0].plot(Ts, m_abs, "o", color="#2c3e50", ms=5, label=f"MC (L={L})")
    axes[0].plot(Tsmooth, onsager_m(Tsmooth), "-", color="#c0392b", lw=2,
                 label="Onsager exact")
    axes[0].axvline(TC_ONSAGER, color="#27ae60", ls=":", lw=1.5, label="$T_c$")
    axes[0].set_xlabel("temperature  $T$"); axes[0].set_ylabel(r"$\langle|m|\rangle$")
    axes[0].set_title("(a)  magnetization"); axes[0].legend(frameon=False)

    axes[1].plot(Ts, chi, "o-", color="#2c3e50", ms=4)
    axes[1].axvline(TC_ONSAGER, color="#27ae60", ls=":", lw=1.5)
    axes[1].set_xlabel("temperature  $T$"); axes[1].set_ylabel(r"susceptibility  $\chi$")
    axes[1].set_title("(b)  susceptibility peak at $T_c$")

    axes[2].plot(Ts, C, "o-", color="#2c3e50", ms=4)
    axes[2].axvline(TC_ONSAGER, color="#27ae60", ls=":", lw=1.5)
    axes[2].set_xlabel("temperature  $T$"); axes[2].set_ylabel(r"heat capacity  $C$")
    axes[2].set_title("(c)  heat-capacity peak at $T_c$")

    fig.suptitle("Monte Carlo simulation of the 2-D Ising model", y=1.05,
                 fontsize=13)
    fig.savefig("fig9_2.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig9_2.png")


if __name__ == "__main__":
    main()
