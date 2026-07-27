"""Example 13 — Hydrogen-air combustion: adiabatic flame temperature & flue gas.

ThermoPack is a *non-reacting* thermodynamic engine, so ThermoLab does not do
chemical equilibrium. Combustion is handled as an atom-conserving composition
change: the adiabatic flame temperature is found from a steady-flow first-law
(isenthalpic) balance that pairs ThermoLab gas-phase sensible enthalpies with
standard enthalpies of formation. Hydrogen is used because H2, O2, N2 and H2O
are all GERG2008 components (methane is *not* in this ThermoPack build).

Stoichiometric reaction, per mole of H2, with air modelled as O2 + 3.76 N2::

    H2 + 0.5 O2 + 1.88 N2  ->  H2O + 1.88 N2

For a lean mixture (equivalence ratio phi <= 1, complete combustion) the O2
leftover and the N2 from the air carry through to the products:

    reactants : H2, O2 = 0.5/phi, N2 = 1.88/phi
    products  : H2O = 1, O2 = 0.5*(1/phi - 1), N2 = 1.88/phi

Energy balance (per mol H2, sensible enthalpy relative to 298.15 K + formation):

    sum_react nu_i * [h_i(T_in) - h_i(Tref)] + sum_react nu_i * dhf_i
      = sum_prod nu_j * [h_j(T_ad) - h_j(Tref)] + sum_prod nu_j * dhf_j

Reference-state subtlety: at 298.15 K and 1 atm water is a compressed *liquid*,
so its vapour enthalpy must be read at a pressure below the saturation pressure
(~3.17 kPa) to give a consistent gas-phase reference; otherwise the latent heat
of vaporization leaks into the sensible enthalpy and the flame temperature comes
out ~400 K too low. We therefore evaluate every species' 298.15 K reference
enthalpy at P_ref = 1 kPa (where every species is a stable vapour ~ ideal gas).

Run with::

    python examples/13_combustion.py
    python examples/13_combustion.py --save combustion.png
"""
from __future__ import annotations

import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from thermolab import Gas, Mixture

# --- Reference / operating conditions ---------------------------------------
T_REF = 298.15          # K, standard reference temperature
P_REF = 1.0e3           # Pa, low enough that H2O is a stable vapour at T_REF
P_OP = 1.0e5            # Pa, combustor / flame operating pressure
T_IN = 298.15           # K, reactant inlet temperature

# Standard enthalpies of formation at 298.15 K, gas phase [J/mol].
# Only H2O(g) is non-zero for the H2/O2/N2/H2O system.
DHF = {"H2": 0.0, "O2": 0.0, "N2": 0.0, "H2O": -241826.0}

_GAS: dict[str, Gas] = {}


def _gas(species: str) -> Gas:
    g = _GAS.get(species)
    if g is None:
        g = Gas(species)
        _GAS[species] = g
    return g


def h_molar(species: str, T: float, P: float) -> float:
    """Gas-phase molar enthalpy of a pure species at (T, P) [J/mol]."""
    return _gas(species).state(T=T, P=P, phase="vapor").h * _gas(species).molar_mass


def sensible(species: str, T: float, P: float) -> float:
    """Sensible enthalpy relative to the gas-phase reference at 298.15 K."""
    return h_molar(species, T, P) - h_molar(species, T_REF, P_REF)


def stoich(phi: float) -> tuple[dict[str, float], dict[str, float]]:
    """Return (reactant, product) stoichiometric coefficients per mol H2."""
    if phi > 1.0:
        raise ValueError("This example handles lean/complete combustion (phi<=1).")
    nu_r = {"H2": 1.0, "O2": 0.5 / phi, "N2": 1.88 / phi}
    nu_p = {"H2O": 1.0, "O2": 0.5 * (1.0 / phi - 1.0), "N2": 1.88 / phi}
    return nu_r, nu_p


def flame_temperature(phi: float, *, T_in: float = T_IN, P: float = P_OP) -> float:
    """Adiabatic flame temperature [K] for hydrogen-air at equivalence ratio phi."""
    nu_r, nu_p = stoich(phi)
    # RHS: product sensible target = reactant sensible + (-sum_prod dhf + sum_react dhf)
    rhs = (sum(nu_r[s] * sensible(s, T_in, P) for s in nu_r)
           + sum(nu_r[s] * DHF[s] for s in nu_r)
           - sum(nu_p[s] * DHF[s] for s in nu_p))

    def residual(T):
        return sum(nu_p[s] * sensible(s, T, P) for s in nu_p) - rhs

    return brentq(residual, 800.0, 4000.0, xtol=0.1)


def product_mixture(phi: float) -> Mixture:
    """The flue-gas mixture for equivalence ratio phi (mole fractions)."""
    _, nu_p = stoich(phi)
    comps = [s for s in ("H2O", "O2", "N2") if nu_p[s] > 0]
    total = sum(nu_p[s] for s in comps)
    return Mixture(comps, [nu_p[s] / total for s in comps])


def main() -> None:
    # --- Heating value of hydrogen ---------------------------------------
    lhv_mol = -DHF["H2O"]                       # J per mol H2 (LHV, water stays vapour)
    lhv_mass = lhv_mol / _gas("H2").molar_mass   # J/kg
    print("Hydrogen combustion (H2 + air -> H2O + N2 + excess O2)")
    print(f"  LHV  = {lhv_mol / 1e3:8.3f} kJ/mol_H2")
    print(f"        = {lhv_mass / 1e6:8.3f} MJ/kg_H2")

    # --- Stoichiometric flame temperature ---------------------------------
    T_stoich = flame_temperature(1.0)
    print(f"\nStoichiometric (phi=1.0) adiabatic flame temperature:")
    print(f"  T_ad = {T_stoich:7.1f} K  (frozen, no dissociation)")

    # --- Equivalence-ratio sweep (lean -> stoichiometric) ----------------
    phis = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
    print("\n phi    T_ad [K]   product gas @ T_ad, 1 bar")
    print("       gamma   cp[J/kg.K]   sound_speed[m/s]   rho[kg/m3]")
    rows = []
    for phi in phis:
        T = flame_temperature(phi)
        flue = product_mixture(phi)
        st = flue.state(T=T, P=P_OP)
        rows.append((phi, T, st.gamma, st.cp, st.sound_speed, st.rho))
        print(f" {phi:.2f}  {T:7.1f}   {st.gamma:5.3f}   {st.cp:10.1f}   "
              f"{st.sound_speed:15.1f}   {st.rho:8.4f}")

    # --- Plot T_ad vs equivalence ratio ----------------------------------
    phis_fine = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    Tads = [flame_temperature(p) for p in phis_fine]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(phis_fine, Tads, "-o", color="C3")
    ax.axvline(1.0, color="0.6", ls="--", lw=1.0, label="stoichiometric")
    ax.axvline(0.0, color="0.6", ls=":")
    ax.set_xlabel("Equivalence ratio  phi  (lean < 1)")
    ax.set_ylabel("Adiabatic flame temperature  T_ad  [K]")
    ax.set_title("Hydrogen-air adiabatic flame temperature (frozen, complete combustion)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\nSaved {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()