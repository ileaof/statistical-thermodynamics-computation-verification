"""StatThermoPy — examples for the species required by the specification.

Runs He, Ar, N2, O2, CO2, H2O, CH4 at a common state and prints a compact report, then
demonstrates the two required mixtures (Ar + N2 and CH4 + CO2). Every number is derived purely
from the molecular partition function — no empirical correlations.
"""

from __future__ import annotations

from statthermopy import IdealGasMixture, State, Thermodynamics, get, list_molecules


def report(name: str, T: float, P: float) -> None:
    mol = get(name)
    th = Thermodynamics(mol, State(T=T, P=P)).compute()
    print(f"\n=== {mol.formula}  (T={T} K, P={P:.3g} Pa, geometry={mol.geometry.value}) ===")
    print(f"  Molar mass : {mol.molar_mass_gmol:.4f} g/mol")
    print(f"  U_m  = {th.U_m:12.4f} J/mol     U_s  = {th.U_s:12.2f} J/kg")
    print(f"  H_m  = {th.H_m:12.4f} J/mol     H_s  = {th.H_s:12.2f} J/kg")
    print(f"  S_m  = {th.S_m:12.4f} J/mol/K   S_s  = {th.S_s:12.2f} J/kg/K")
    print(f"  A_m  = {th.A_m:12.4f} J/mol     A_s  = {th.A_s:12.2f} J/kg")
    print(f"  G_m  = {th.G_m:12.4f} J/mol     G_s  = {th.G_s:12.2f} J/kg")
    print(f"  Cv_m = {th.Cv_m:12.4f} J/mol/K  Cv_s = {th.Cv_s:12.2f} J/kg/K")
    print(f"  Cp_m = {th.Cp_m:12.4f} J/mol/K  Cp_s = {th.Cp_s:12.2f} J/kg/K")
    print(f"  gamma= {th.gamma:12.6f}         R*  = {th.R_specific:12.4f} J/kg/K")
    print(f"  mu_m = {th.mu_m:12.4f} J/mol")
    print(f"  Q = {th.Qtotal:.6e}   (ln Q = {th.ln_Qtotal:.4f})")
    print("  modes:")
    for m, c in th.contributions.items():
        if c["ln_q"] != 0.0 or c["U_m"] != 0.0:
            print(f"    {m:13s} lnQ={c['ln_q']:+.4f}  U={c['U_m']:12.4f}  "
                  f"S={c['S_m']:10.4f}  Cv={c['Cv_m']:8.4f}")


def main() -> None:
    print(f"Available gases ({len(list_molecules())}): {', '.join(list_molecules())}")
    T, P = 298.15, 101325.0
    for name in ["He", "Ar", "N2", "O2", "CO2", "H2O"]:
        report(name, T, P)
    # CH4 at the higher temperature/pressure requested in the specification.
    report("CH4", 800.0, 5e5)

    print("\n" + "=" * 60)
    print("Mixture 1: Ar + N2  (mole fractions 0.7 / 0.3) at 298.15 K, 1 atm")
    print("=" * 60)
    mix1 = IdealGasMixture.from_names({"Ar": 0.7, "N2": 0.3})
    r1 = mix1.compute(State(T=298.15, P=101325.0))
    print(f"  M_avg = {r1.M_avg*1e3:.4f} g/mol   R* = {r1.R_specific:.4f} J/kg/K")
    print(f"  Cp_m = {r1.Cp_m:.4f}  Cv_m = {r1.Cv_m:.4f}  gamma = {r1.gamma:.6f}")
    print(f"  S_m  = {r1.S_m:.4f} J/mol/K   G_m = {r1.G_m:.4f} J/mol")

    print("\n" + "=" * 60)
    print("Mixture 2: CH4 + CO2  (mole fractions 0.5 / 0.5) at 800 K, 5 bar")
    print("=" * 60)
    mix2 = IdealGasMixture.from_names({"CH4": 0.5, "CO2": 0.5})
    r2 = mix2.compute(State(T=800.0, P=5e5))
    print(f"  M_avg = {r2.M_avg*1e3:.4f} g/mol   R* = {r2.R_specific:.4f} J/kg/K")
    print(f"  Cp_m = {r2.Cp_m:.4f}  Cv_m = {r2.Cv_m:.4f}  gamma = {r2.gamma:.6f}")
    print(f"  S_m  = {r2.S_m:.4f} J/mol/K   G_m = {r2.G_m:.4f} J/mol")


if __name__ == "__main__":
    main()