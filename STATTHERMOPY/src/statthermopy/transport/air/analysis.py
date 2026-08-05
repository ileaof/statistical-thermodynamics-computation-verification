"""Air-transport analyses: per-species contribution tables and multi-property sweeps.

Thin data-layer helpers built on :class:`.AirTransport` and
:class:`~statthermopy.humidair.analysis.ComparisonTable`. They produce the tabular inputs used by
the air-transport plots and exporters — no physics here, only orchestration.
"""

from __future__ import annotations

from ...humidair.analysis import ComparisonTable
from .air_transport import AirTransport, AirTransportTable
from .mixture_transport import AIR_TRANSPORT_LABELS, AIR_TRANSPORT_PROPS, AIR_TRANSPORT_UNITS

__all__ = ["AirTransportAnalysis"]


class AirTransportAnalysis:
    """Comparative air-transport analyses: per-species contributions and multi-property sweeps.

    Parameters
    ----------
    model : AirTransport, optional
        The air-transport model. Defaults to a standard :class:`AirTransport`.
    """

    def __init__(self, model: AirTransport | None = None) -> None:
        self.model = model if model is not None else AirTransport()

    # -- per-species contributions --------------------------------------------

    def species_contributions(
        self,
        T: float,
        P: float,
        *,
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
        saturated: bool = False,
    ) -> ComparisonTable:
        """Per-species transport breakdown at ``(T, P)`` for the chosen humidity spec.

        One row per species (dry species, plus H₂O for humid air), with columns: mole fraction,
        molar mass, pure-species viscosity / conductivity, Blanc diffusivity into the mixture, and
        the Wilke / Mason–Saxena contributions to the mixture totals.
        """
        res = self.model.humid(
            T, P,
            relative_humidity=relative_humidity,
            humidity_ratio=humidity_ratio,
            mole_fraction=mole_fraction,
            saturated=saturated,
        )
        names = list(res.components.keys())
        x = names  # the "axis" is the species name
        cols = {
            "x [-]": [res.components[n].x for n in names],
            "M [g/mol]": [res.components[n].molar_mass * 1e3 for n in names],
            "mu_i [Pa·s]": [res.components[n].mu_i for n in names],
            "k_i [W/m/K]": [res.components[n].k_i for n in names],
            "D_im [m^2/s]": [res.components[n].D_im for n in names],
            "mu_contrib [Pa·s]": [res.components[n].mu_contrib for n in names],
            "k_contrib [W/m/K]": [res.components[n].k_contrib for n in names],
        }
        tag = "humid air" if (res.humidity_ratio or 0) > 0 else "dry air"
        return ComparisonTable(
            title=f"Per-species transport contributions — {tag} @ {T:.2f} K, {P:.4g} Pa",
            x_label="Species",
            y_label="value",
            x_unit="",
            x=x,
            x_K=[T] * len(names),
            columns=cols,
            meta={"T": T, "P": P, "label": res.label,
                  "humidity_ratio": res.humidity_ratio},
        )

    # -- all eight headline properties vs T -----------------------------------

    def all_properties_vs_T(
        self,
        T_range,
        P: float = 101325.0,
        *,
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
        saturated: bool = False,
        temperature_unit: str = "K",
    ) -> dict[str, AirTransportTable]:
        """Build the eight headline dry-vs-humid comparison tables (one per property)."""
        out: dict[str, AirTransportTable] = {}
        for prop in AIR_TRANSPORT_PROPS:
            out[prop] = self.model.compare_vs_T(
                prop, T_range, P,
                relative_humidity=relative_humidity,
                humidity_ratio=humidity_ratio,
                mole_fraction=mole_fraction,
                saturated=saturated,
                temperature_unit=temperature_unit,
            )
        return out

    # -- single species record lookup (display) -------------------------------

    def species_record(self, name: str):
        """Return the extended :class:`SpeciesTransportData` record for ``name`` if present."""
        from .species_data import get_species_transport

        try:
            return get_species_transport(name)
        except KeyError:
            return None


# silence unused-import lint: labels/units/props document the analysis contract
_ = (AIR_TRANSPORT_LABELS, AIR_TRANSPORT_UNITS, AIR_TRANSPORT_PROPS)