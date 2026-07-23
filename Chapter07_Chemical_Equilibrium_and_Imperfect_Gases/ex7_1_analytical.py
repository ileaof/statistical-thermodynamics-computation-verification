#!/usr/bin/env python3
"""
Example 7.1 (analytical) -- Chemical equilibrium from molecular partition
functions: the dissociation H2 <-> 2 H.

The statistical equilibrium constant is a ratio of molecular partition functions
per unit volume times a Boltzmann factor of the dissociation energy D0:

    K_p(T) = [ (q_H / V)^2 / (q_H2 / V) ] * (k_B T / p0) * exp(-D0 / k_B T),

with q_H = q_tr,H * g_H (an atom: translation and electronic degeneracy only) and
q_H2 = q_tr,H2 * q_rot * q_vib * g_H2.  From K_p and the total pressure the degree
of dissociation alpha follows from K_p = (P/p0) * 4 alpha^2 / (1 - alpha^2).

The reaction enthalpy is obtained two ways and checked for agreement:
  * van't Hoff:  Delta H = R T^2 d(ln K_p)/dT   (numerical derivative),
  * direct:      Delta H = 2 H_H - H_H2 + N_A D0  from the molecular enthalpies.

numpy/scipy/matplotlib only.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

h  = 6.62607015e-34
kB = 1.380649e-23
NA = 6.02214076e23
R  = NA * kB
u  = 1.66053906660e-27
eV = 1.602176634e-19
p0 = 1.0e5                        # standard pressure, 1 bar

mH  = 1.007825 * u
mH2 = 2.015650 * u
TH_ROT = 87.6                     # H2 rotational temperature, K
TH_VIB = 6332.0                   # H2 vibrational temperature, K
D0  = 4.478 * eV                  # H2 dissociation energy from v=0, J
gH, gH2, SIGMA = 2, 1, 2          # electronic degeneracies, symmetry number


def q_tr_over_V(m, T):
    return (2.0 * np.pi * m * kB * T / h ** 2) ** 1.5


def q_rot(T):
    r = TH_ROT / T
    return (1.0 / (SIGMA * r)) * (1 + r / 3 + r ** 2 / 15)   # Euler-Maclaurin


def q_vib(T):
    return 1.0 / (1.0 - np.exp(-TH_VIB / T))


def Kp(T):
    qH = q_tr_over_V(mH, T) * gH
    qH2 = q_tr_over_V(mH2, T) * q_rot(T) * q_vib(T) * gH2
    return (qH ** 2 / qH2) * (kB * T / p0) * np.exp(-D0 / (kB * T))


def alpha_of(T, P):
    """Degree of dissociation from Kp = (P/p0) 4 a^2/(1-a^2)."""
    K = Kp(T)
    # solve 4(P/p0) a^2 = K(1-a^2)  ->  a = sqrt(K / (K + 4 P/p0))
    return np.sqrt(K / (K + 4.0 * P / p0))


def main():
    print("=" * 70)
    print("Example 7.1  Chemical equilibrium  H2 <-> 2H  from partition functions")
    print("=" * 70)

    print(f"\nD0 = {D0/eV:.3f} eV = {D0*NA/1000:.1f} kJ/mol")
    print(f"\n{'T (K)':>7} {'Kp':>12} {'alpha (1 bar)':>14} {'alpha (0.1 bar)':>16}")
    for T in [2000, 3000, 4000, 5000, 6000]:
        print(f"{T:>7} {Kp(T):>12.4e} {alpha_of(T,1e5):>14.4f} "
              f"{alpha_of(T,1e4):>16.4f}")

    # --- van't Hoff vs direct enthalpy ---------------------------------
    print("\nReaction enthalpy: van't Hoff vs direct (kJ/mol):")
    print(f"  {'T (K)':>7} {'vantHoff':>12} {'direct':>12} {'diff %':>8}")
    for T in [3000, 4000, 5000]:
        dT = 1.0
        dlnK = (np.log(Kp(T + dT)) - np.log(Kp(T - dT))) / (2 * dT)
        dH_vh = R * T ** 2 * dlnK                              # J/mol
        # direct: H_H = 5/2 RT ; H_H2 = 7/2 RT + vibrational enthalpy
        Evib = R * TH_VIB / (np.exp(TH_VIB / T) - 1.0)         # molar vib energy
        H_H = 2.5 * R * T
        H_H2 = 3.5 * R * T + Evib
        dH_direct = 2 * H_H - H_H2 + NA * D0
        print(f"  {T:>7} {dH_vh/1000:>12.2f} {dH_direct/1000:>12.2f} "
              f"{100*abs(dH_vh-dH_direct)/dH_direct:>8.3f}")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    T = np.linspace(1500, 6500, 300)
    ax1.semilogy(T, [Kp(t) for t in T], color="#2c3e50", lw=2)
    ax1.set_xlabel("temperature  $T$  (K)")
    ax1.set_ylabel(r"$K_p$")
    ax1.set_title(r"(a)  equilibrium constant  $K_p(T)$")

    for P, c in [(1e6, "#2980b9"), (1e5, "#2c3e50"), (1e4, "#c0392b")]:
        ax2.plot(T, [alpha_of(t, P) for t in T], color=c, lw=2,
                 label=f"{P/1e5:.0f} bar" if P >= 1e5 else "0.1 bar")
    ax2.set_xlabel("temperature  $T$  (K)")
    ax2.set_ylabel(r"degree of dissociation  $\alpha$")
    ax2.set_title(r"(b)  dissociation of H$_2$ vs $T$ and $P$")
    ax2.legend(frameon=False, title="total pressure")

    fig.suptitle("Chemical equilibrium of H$_2$ dissociation", y=1.05,
                 fontsize=13)
    fig.savefig("fig7_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig7_1.png")


if __name__ == "__main__":
    main()
