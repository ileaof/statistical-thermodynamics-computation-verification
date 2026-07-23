#!/usr/bin/env python3
"""
Example 4.1 (analytical) -- The Sackur-Tetrode equation: absolute entropy of a
monatomic ideal gas from first principles, tested against experiment.

The translational partition function of one atom in a volume V is
z = V/Lambda^3 with the thermal de Broglie wavelength

    Lambda = h / sqrt(2 pi m k_B T).

Applying the Gibbs-corrected canonical machinery (Z = z^N / N!) gives the
Sackur-Tetrode entropy of N indistinguishable atoms,

    S = N k_B [ ln( V / (N Lambda^3) ) + 5/2 ],

or, per mole at pressure p (using V/N = k_B T / p),

    S_m = R [ ln( (2 pi m)^{3/2} (k_B T)^{5/2} / (h^3 p) ) + 5/2 ].

For the noble gases -- monatomic, with non-degenerate electronic ground states --
this PURELY THEORETICAL entropy should reproduce the experimental standard molar
entropies.  This script computes S_m(298.15 K, 1 bar) for He...Xe and compares.

Only numpy, scipy, matplotlib are used.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants (SI, CODATA)
h    = 6.62607015e-34      # J s
kB   = 1.380649e-23        # J/K
NA   = 6.02214076e23       # 1/mol
R    = NA * kB             # J/(mol K)
u    = 1.66053906660e-27   # kg


def thermal_wavelength(m, T):
    """de Broglie thermal wavelength Lambda = h / sqrt(2 pi m k_B T)."""
    return h / np.sqrt(2.0 * np.pi * m * kB * T)


def sackur_tetrode_molar(M_u, T, p):
    """Molar entropy (J/mol/K) of a monatomic ideal gas, molar mass M_u in u."""
    m = M_u * u
    Lam = thermal_wavelength(m, T)
    v_per_atom = kB * T / p                    # V/N from the ideal-gas law
    return R * (np.log(v_per_atom / Lam ** 3) + 2.5)


def main():
    print("=" * 70)
    print("Example 4.1  Sackur-Tetrode absolute entropy vs experiment")
    print("=" * 70)

    T, p = 298.15, 1.0e5                        # standard conditions, 1 bar
    # Experimental standard molar entropies S deg (J/mol/K), monatomic gases
    gases = {
        "He": (4.002602, 126.15),
        "Ne": (20.1797, 146.33),
        "Ar": (39.948,  154.85),
        "Kr": (83.798,  164.09),
        "Xe": (131.293, 169.68),
    }
    print(f"\nStandard conditions: T = {T} K, p = {p:.3e} Pa (1 bar)")
    print(f"{'gas':>5} {'M (u)':>9} {'S ST (calc)':>12} {'S exp':>9} "
          f"{'diff':>8} {'% err':>7}")
    calc, exp, masses = [], [], []
    for name, (M, S_exp) in gases.items():
        S = sackur_tetrode_molar(M, T, p)
        calc.append(S); exp.append(S_exp); masses.append(M)
        print(f"{name:>5} {M:>9.4f} {S:>12.2f} {S_exp:>9.2f} "
              f"{S-S_exp:>8.2f} {100*(S-S_exp)/S_exp:>7.2f}")
    calc, exp, masses = map(np.array, (calc, exp, masses))
    print(f"\nRMS deviation from experiment = "
          f"{np.sqrt(np.mean((calc-exp)**2)):.2f} J/mol/K")
    # Verify the (3/2)R ln M mass scaling between two gases (Ar vs Ne):
    dS_pred = 1.5 * R * np.log(gases['Ar'][0] / gases['Ne'][0])
    dS_calc = sackur_tetrode_molar(*(gases['Ar'][0], T, p)) \
            - sackur_tetrode_molar(*(gases['Ne'][0], T, p))
    print(f"mass scaling Ar-Ne: (3/2)R ln(M_Ar/M_Ne) = {dS_pred:.3f}, "
          f"direct = {dS_calc:.3f} J/mol/K")

    # -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   constrained_layout=True)

    names = list(gases.keys())
    xpos = np.arange(len(names))
    ax1.bar(xpos - 0.2, calc, 0.4, color="#2c3e50", label="Sackur-Tetrode")
    ax1.bar(xpos + 0.2, exp, 0.4, color="#c0392b", label="experiment")
    ax1.set_xticks(xpos); ax1.set_xticklabels(names)
    ax1.set_ylabel(r"$S^\circ_m$  (J mol$^{-1}$ K$^{-1}$)")
    ax1.set_ylim(100, 180)
    ax1.set_title("(a)  absolute entropy: theory vs experiment")
    ax1.legend(frameon=False)

    # entropy vs temperature for argon at 1 bar, showing the 5/2 R ln T growth
    Tg = np.linspace(100, 1000, 300)
    ax2.plot(Tg, sackur_tetrode_molar(gases['Ar'][0], Tg, p),
             color="#2c3e50", lw=2, label="Ar, $p=1$ bar")
    ax2.plot(Tg, sackur_tetrode_molar(gases['He'][0], Tg, p),
             color="#c0392b", lw=2, label="He, $p=1$ bar")
    ax2.set_xlabel("temperature  $T$  (K)")
    ax2.set_ylabel(r"$S_m$  (J mol$^{-1}$ K$^{-1}$)")
    ax2.set_title(r"(b)  $S_m = \frac{5}{2}R\ln T + \cdots$")
    ax2.legend(frameon=False)

    fig.suptitle("Sackur-Tetrode absolute entropy of monatomic gases",
                 y=1.05, fontsize=13)
    fig.savefig("fig4_1.png", dpi=200, bbox_inches="tight")
    print("\nSaved fig4_1.png")


if __name__ == "__main__":
    main()
