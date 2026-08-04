"""Predefined fluids — named gas compositions built on the ideal-gas mixture engine.

A *predefined fluid* is a named, documented composition (mole fractions of database species) that
the CLI and GUI can offer as a ready-made selection. The flagship fluid is **atmospheric air**:
standard dry air (N₂, O₂, Ar, CO₂) with an optional water-vapour fraction.

Everything here is pure statistical mechanics downstream: a fluid is just a factory that returns
an :class:`~statthermopy.mixture.IdealGasMixture`, whose properties are computed from each
species' molecular partition function and combined through the ideal-mixture relations
(including the entropy of mixing). No empirical property correlation enters the calculation path.

Design notes (extensibility)
----------------------------
* The registry is open: :func:`register_fluid` adds new named compositions, and users remain free
  to bypass presets entirely and build any custom composition with
  :meth:`IdealGasMixture.from_names`.
* Compositions are stored as plain mole-fraction dictionaries, deliberately decoupled from how the
  mixture is *evaluated*. Today every fluid is evaluated as an ideal-gas mixture; a future
  non-ideal (real-gas) mixture model can be slotted in behind the same :meth:`PredefinedFluid.build`
  factory without changing callers.
* Humidity is specified as a **water mole fraction** (a composition input), never derived from a
  saturation-pressure correlation — keeping the whole path free of empirical curves. A
  relative-humidity front-end would need a saturation model and is intentionally left out.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mixture import IdealGasMixture

__all__ = [
    "STANDARD_DRY_AIR",
    "PredefinedFluid",
    "air",
    "available_fluids",
    "get_fluid",
    "register_fluid",
]

#: Standard dry-air composition (mole fractions). N₂/O₂/Ar/CO₂ per the US Standard Atmosphere /
#: CIPM; the values sum to ~1.00004 and are normalised on use. Gives M̄ ≈ 28.96 g/mol and a
#: specific gas constant R ≈ 287 J/kg/K.
STANDARD_DRY_AIR: dict[str, float] = {
    "N2": 0.78084,
    "O2": 0.20946,
    "AR": 0.00934,
    "CO2": 0.00040,
}


@dataclass(frozen=True)
class PredefinedFluid:
    """A named gas composition that can be built into an :class:`IdealGasMixture`.

    Attributes
    ----------
    name : str
        Display name (e.g. ``"Air"``).
    description : str
        One-line human-readable description.
    composition : dict[str, float]
        Mole fractions of database species (need not be normalised).
    source : str
        Provenance of the composition values.
    humidifiable : bool
        Whether an optional water-vapour fraction is meaningful for this fluid.
    """

    name: str
    description: str
    composition: dict[str, float]
    source: str = ""
    humidifiable: bool = False

    def dry_composition(self) -> dict[str, float]:
        """Return a copy of the (normalised) dry mole-fraction composition."""
        total = sum(self.composition.values())
        return {sp: frac / total for sp, frac in self.composition.items()}

    def build(self, *, water_mole_fraction: float = 0.0) -> IdealGasMixture:
        """Build the :class:`IdealGasMixture` for this fluid.

        Parameters
        ----------
        water_mole_fraction : float, default 0.0
            Optional water-vapour mole fraction in ``[0, 1)``. The dry composition is scaled to
            ``1 - water_mole_fraction`` and H₂O is added at ``water_mole_fraction``. Only allowed
            for a :attr:`humidifiable` fluid.
        """
        if water_mole_fraction and not self.humidifiable:
            raise ValueError(f"Fluid {self.name!r} does not support a water-vapour fraction.")
        return _build_composition(self.dry_composition(), water_mole_fraction)


def _build_composition(
    dry: dict[str, float], water_mole_fraction: float
) -> IdealGasMixture:
    """Assemble an :class:`IdealGasMixture` from a dry composition plus optional water vapour."""
    if not (0.0 <= water_mole_fraction < 1.0):
        raise ValueError("water_mole_fraction must be in [0, 1).")
    total = sum(dry.values())
    comp = {sp: (frac / total) * (1.0 - water_mole_fraction) for sp, frac in dry.items()}
    if water_mole_fraction > 0.0:
        comp["H2O"] = comp.get("H2O", 0.0) + water_mole_fraction
    return IdealGasMixture.from_names(comp, basis="mole")


# --- registry ---------------------------------------------------------------

_FLUIDS: dict[str, PredefinedFluid] = {}


def register_fluid(fluid: PredefinedFluid) -> None:
    """Register (or replace) a predefined fluid, keyed case-insensitively by its name."""
    _FLUIDS[fluid.name.lower()] = fluid


def available_fluids() -> list[str]:
    """Return the registered predefined-fluid display names (sorted)."""
    return sorted(f.name for f in _FLUIDS.values())


def get_fluid(name: str) -> PredefinedFluid:
    """Return the predefined fluid with the given name (case-insensitive)."""
    key = name.lower()
    if key not in _FLUIDS:
        avail = ", ".join(available_fluids())
        raise KeyError(f"Unknown fluid {name!r}. Available: {avail}.")
    return _FLUIDS[key]


def air(*, water_mole_fraction: float = 0.0) -> IdealGasMixture:
    """Build atmospheric air as an :class:`IdealGasMixture`.

    Standard dry air (N₂, O₂, Ar, CO₂) with an optional water-vapour mole fraction.

    Parameters
    ----------
    water_mole_fraction : float, default 0.0
        Water-vapour mole fraction in ``[0, 1)``. ``0.0`` gives dry air; e.g. ``0.01`` adds 1 %
        H₂O by mole and scales the dry constituents to fill the remaining 99 %.
    """
    return _build_composition(STANDARD_DRY_AIR, water_mole_fraction)


# Register the built-in fluids.
register_fluid(
    PredefinedFluid(
        name="Air",
        description="Standard atmospheric air (dry N2/O2/Ar/CO2; optional water vapour)",
        composition=STANDARD_DRY_AIR,
        source="US Standard Atmosphere / CIPM dry-air mole fractions",
        humidifiable=True,
    )
)
