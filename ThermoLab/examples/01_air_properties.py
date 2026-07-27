"""Example 01 — Pure-fluid (air) properties from a (T, P) specification.

This is the canonical quick-start from the ThermoLab specification::

    from thermolab import Gas
    air = Gas("Air", backend="thermopack")
    st = air.state(T=800.0, P=5e5)

Run with::

    python examples/01_air_properties.py
"""
from __future__ import annotations

from thermolab import Gas


def main() -> None:
    air = Gas("Air", backend="thermopack")
    st = air.state(T=800.0, P=5e5)

    print(f"Air at T = {st.T:.1f} K, P = {st.P:.3e} Pa")
    print("-" * 52)
    for attr in (
        "rho", "v", "u", "h", "s", "cp", "cv", "gamma", "Z",
        "sound_speed", "mu", "k", "thermal_diffusivity", "prandtl",
        "joule_thomson", "beta_thermal_expansion", "kappa_t",
    ):
        print(f"{attr:>22s} = {getattr(st, attr):.4g}")

    # A clean tabular view, also available via repr:
    print("-" * 52)
    print(st)

    # Any independent variable pair works for flash:
    st2 = air.state(P=5e5, h=st.h)  # recover the same state from (P, h)
    print(f"\nRound-trip (P, h) -> T = {st2.T:.2f} K  (expected {st.T:.2f} K)")


if __name__ == "__main__":
    main()