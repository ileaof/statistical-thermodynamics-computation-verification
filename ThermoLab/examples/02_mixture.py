"""Example 02 — Multicomponent gas mixture (combustion / flue-gas-like).

Builds a four-component mixture and flashes a state from (T, P). ThermoLab
selects the GERG2008 multiparameter EOS automatically when every component is
in the GERG core (as N2, O2, CO2, H2O are).

Run with::

    python examples/02_mixture.py
"""
from __future__ import annotations

from thermolab import Mixture


def main() -> None:
    mix = Mixture(
        ["N2", "O2", "CO2", "H2O"],
        [0.78, 0.21, 0.005, 0.005],
        backend="thermopack",
    )
    print(f"Components : {mix.components}")
    print(f"Fractions  : {mix.fractions}")
    print(f"Molar mass : {mix.molar_mass * 1000:.3f} g/mol")

    st = mix.state(T=1200.0, P=3e5)
    print(f"\nMixture at T = {st.T:.1f} K, P = {st.P:.3e} Pa, phase = {st.phase}")
    print("-" * 52)
    for attr in ("rho", "h", "s", "u", "cp", "cv", "gamma", "Z", "sound_speed",
                 "mu", "k", "prandtl"):
        print(f"{attr:>14s} = {getattr(st, attr):.4g}")

    # Update composition in place (e.g. richer in CO2 / H2O for exhaust gas).
    mix.set_fractions([0.70, 0.10, 0.10, 0.10])
    st2 = mix.state(T=1200.0, P=3e5)
    print(f"\nAfter enriching CO2/H2O: rho = {st2.rho:.4g} kg/m^3, "
          f"cp = {st2.cp:.4g} J/(kg.K)")


if __name__ == "__main__":
    main()