"""Mass-based property bundle and formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class PropertyBundle:
    """Complete mass-based thermophysical property set (SI units).

    All extensive quantities are *specific* (per unit mass).

    Units:
        T [K], P [Pa], rho [kg/m^3], v [m^3/kg],
        u, h, g, a [J/kg], s [J/(kg.K)],
        cp, cv [J/(kg.K)], gamma [-], Z [-], sound_speed [m/s],
        mu [Pa.s], k [W/(m.K)], thermal_diffusivity [m^2/s], prandtl [-],
        joule_thomson [K/Pa], beta_thermal_expansion [1/K], kappa_t [1/Pa].
    """

    T: float
    P: float
    rho: float
    v: float
    u: float
    h: float
    s: float
    g: float
    a_helmholtz: float
    cp: float
    cv: float
    gamma: float
    Z: float
    sound_speed: float
    mu: float
    k: float
    thermal_diffusivity: float
    prandtl: float
    joule_thomson: float
    beta_thermal_expansion: float
    kappa_t: float
    # optional diagnostics
    phase: str = "vapor"
    quality: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_series(self):
        """Return a pandas Series indexed by property name."""
        import pandas as pd

        return pd.Series(asdict(self))


# Human-readable table formatting: (attr, label, format spec, unit)
_PROPERTY_TABLE: tuple[tuple[str, str, str, str], ...] = (
    ("T", "Temperature", "{:>14.4g}", "K"),
    ("P", "Pressure", "{:>14.4g}", "Pa"),
    ("rho", "Density", "{:>14.4g}", "kg/m^3"),
    ("v", "Spec. volume", "{:>14.4g}", "m^3/kg"),
    ("u", "Int. energy", "{:>14.4g}", "J/kg"),
    ("h", "Enthalpy", "{:>14.4g}", "J/kg"),
    ("s", "Entropy", "{:>14.4g}", "J/(kg.K)"),
    ("g", "Gibbs energy", "{:>14.4g}", "J/kg"),
    ("a_helmholtz", "Helmholtz energy", "{:>14.4g}", "J/kg"),
    ("cp", "Cp", "{:>14.4g}", "J/(kg.K)"),
    ("cv", "Cv", "{:>14.4g}", "J/(kg.K)"),
    ("gamma", "gamma (Cp/Cv)", "{:>14.4g}", "-"),
    ("Z", "Compressibility", "{:>14.4g}", "-"),
    ("sound_speed", "Speed of sound", "{:>14.4g}", "m/s"),
    ("mu", "Viscosity", "{:>14.4g}", "Pa.s"),
    ("k", "Therm. conductivity", "{:>14.4g}", "W/(m.K)"),
    ("thermal_diffusivity", "Therm. diffusivity", "{:>14.4g}", "m^2/s"),
    ("prandtl", "Prandtl", "{:>14.4g}", "-"),
    ("joule_thomson", "Joule-Thomson", "{:>14.4g}", "K/Pa"),
    ("beta_thermal_expansion", "Therm. expansion", "{:>14.4g}", "1/K"),
    ("kappa_t", "Isoth. compress.", "{:>14.4g}", "1/Pa"),
)


def format_property_table(bundle: PropertyBundle) -> str:
    """Return a nicely aligned text table of all properties."""
    lines = []
    header = f"{'Property':<22}{'Value':>16}  {'Unit':<10}"
    lines.append(header)
    lines.append("-" * len(header))
    for attr, label, spec, unit in _PROPERTY_TABLE:
        val = getattr(bundle, attr, None)
        if val is None:
            lines.append(f"{label:<22}{'--':>16}  {unit:<10}")
        else:
            lines.append(f"{label:<22}{spec.format(float(val))}  {unit:<10}")
    q = bundle.quality
    if q is not None:
        lines.append(f"{'Quality':<22}{q:>16.4f}  {'-':<10}")
    lines.append("-" * len(header))
    lines.append(f"{'Phase':<22}{str(bundle.phase):>16}  {'':<10}")
    return "\n".join(lines)