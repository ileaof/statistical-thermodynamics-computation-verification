"""N2 deep-dive: properties at 298.15 K, quantum-vs-classical rotation, and Cp(T) curve.

Also saves the full set of property-vs-T plots under ``examples/output/n2_*.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from statthermopy import State, Thermodynamics, get
from statthermopy.plots import plot_all_properties, plot_property


def main() -> None:
    n2 = get("N2")
    st = State(T=298.15, P=101325.0)
    th = Thermodynamics(n2, st).compute()
    print("N2 @ 298.15 K, 1 atm")
    print(f"  Cp_m = {th.Cp_m:.4f} J/mol/K  (literature ~29.10)")
    print(f"  Cv_m = {th.Cv_m:.4f} J/mol/K  (literature ~20.79)")
    print(f"  gamma= {th.gamma:.6f}        (literature ~1.40)")
    print(f"  S_m  = {th.S_m:.4f} J/mol/K  (literature ~191.5)")

    # Compare classical vs exact quantum rigid rotor for N2 across T.
    print("\n  T [K]   Cv_classical   Cv_quantum")
    for T in (50, 100, 300, 1000, 3000):
        c = Thermodynamics(n2, State(T=T, P=1e5), use_quantum_rotation=False).compute().Cv_m
        q = Thermodynamics(n2, State(T=T, P=1e5), use_quantum_rotation=True).compute().Cv_m
        print(f"  {T:5d}   {c:12.6f}   {q:12.6f}")

    # Property-vs-T curves.
    out = Path(__file__).parent / "output"
    plot_all_properties(n2, np.linspace(200, 2000, 100), P=1e5, save_dir=out)
    ax = plot_property(n2, "Cp_m", np.linspace(300, 2000, 80), P=1e5,
                       label="N2 Cp_m (statistical mechanics)")
    ax.figure.savefig(out / "n2_Cp_highlight.png", dpi=120, bbox_inches="tight")
    print(f"\n  saved plots to {out}")


if __name__ == "__main__":
    main()