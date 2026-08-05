"""Dry-air and humid-air transport — the air transport-property facade.

:class:`AirTransport` builds on the statistical :class:`~statthermopy.humidair.HumidAir` model
(which supplies the dry-air background, the water saturation physics and the humidity-ratio
resolution) and the :class:`.MixtureTransportCalculator` (which supplies the Wilke / Mason–Saxena /
Blanc mixing rules). It exposes dry-air and humid-air transport at a state, and dry-vs-humid
comparison tables versus temperature. No physics is reimplemented here — it only wires the
composition (from :class:`HumidAir`) to the mixture transport engine.
"""

from __future__ import annotations

from ...core.state import State
from ...humidair import HumidAir
from ...humidair.analysis import ComparisonTable
from .mixture_transport import (
    AIR_TRANSPORT_LABELS,
    AIR_TRANSPORT_UNITS,
    MixtureTransportCalculator,
)

__all__ = ["AirTransport", "AirTransportTable"]


class AirTransportTable(ComparisonTable):
    """A :class:`~statthermopy.humidair.analysis.ComparisonTable` for air-transport comparisons.

    Inherits ``to_csv`` / ``to_excel`` / ``to_json`` / ``to_pdf`` / ``to_dataframe`` from
    :class:`ComparisonTable`. Carries dry-air, humid-air and (optionally) difference columns.
    """


def _humidity_kwargs(
    relative_humidity: float | None,
    humidity_ratio: float | None,
    mole_fraction: float | None,
    saturated: bool,
) -> dict:
    """Pack the humidity spec into the keyword arguments accepted by :meth:`HumidAir.state`."""
    kw: dict = {}
    if relative_humidity is not None:
        kw["relative_humidity"] = relative_humidity
    if humidity_ratio is not None:
        kw["humidity_ratio"] = humidity_ratio
    if mole_fraction is not None:
        kw["mole_fraction"] = mole_fraction
    if saturated:
        kw["saturated"] = True
    return kw


class AirTransport:
    """Dry-air and humid-air transport properties over a dry-gas background.

    Parameters
    ----------
    humid_air : HumidAir, optional
        The humid-air model (dry background + water saturation). Defaults to the standard
        :class:`HumidAir` (standard dry air N₂/O₂/Ar/CO₂).

    Notes
    -----
    The humid composition at a state is resolved by :meth:`HumidAir.state` from the humidity spec
    (relative humidity, humidity ratio, mole fraction, or the saturation limit), then fed to the
    :class:`.MixtureTransportCalculator`. Every transport property therefore updates
    automatically with temperature, pressure and humidity.
    """

    def __init__(self, humid_air: HumidAir | None = None) -> None:
        self.humid_air = humid_air if humid_air is not None else HumidAir()
        self.dry_air = self.humid_air.dry_air

    # -- point evaluation -----------------------------------------------------

    def dry(self, T: float, P: float, *, label: str = "Dry air"):
        """Dry-air transport properties at ``(T, P)``."""
        calc = MixtureTransportCalculator(self.dry_air)
        return calc.compute(State(T=float(T), P=float(P)), label=label)

    def humid(
        self,
        T: float,
        P: float,
        *,
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
        saturated: bool = False,
        label: str = "Humid air",
    ):
        """Humid-air transport properties at ``(T, P)`` for the given humidity spec.

        The composition is resolved by :meth:`HumidAir.state`; the transport is then computed by
        :class:`.MixtureTransportCalculator` on the humid mixture. The resulting
        :class:`.MixtureTransportProperties` carries the resolved ``humidity_ratio``.
        """
        T = float(T)
        P = float(P)
        ha = self.humid_air.state(
            T,
            P,
            relative_humidity=relative_humidity,
            humidity_ratio=humidity_ratio,
            mole_fraction=mole_fraction,
            saturated=saturated,
            wet_bulb=False,
            dew_point=False,
        )
        mix = self.humid_air._humid_mixture(ha.x_h2o)
        res = MixtureTransportCalculator(mix).compute(State(T=T, P=P), label=label)
        res.humidity_ratio = ha.humidity_ratio
        return res

    # -- comparison vs T ------------------------------------------------------

    def compare_vs_T(
        self,
        prop: str,
        T_range,
        P: float = 101325.0,
        *,
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
        saturated: bool = False,
        temperature_unit: str = "K",
    ) -> AirTransportTable:
        """Dry-vs-humid comparison of one transport property versus temperature.

        At each temperature the dry-air value and the humid-air value (composition resolved from
        the humidity spec at that temperature) are evaluated, plus the absolute difference. The
        result is an :class:`AirTransportTable` (a :class:`ComparisonTable`) ready to plot or
        export to CSV / Excel / JSON / PDF.
        """
        unit = temperature_unit
        Ts = [float(t) for t in T_range]
        dry_vals: list[float] = []
        hum_vals: list[float] = []
        diff_vals: list[float] = []
        for t in Ts:
            d = getattr(self.dry(t, P), prop)
            h = getattr(
                self.humid(
                    t,
                    P,
                    relative_humidity=relative_humidity,
                    humidity_ratio=humidity_ratio,
                    mole_fraction=mole_fraction,
                    saturated=saturated,
                ),
                prop,
            )
            dry_vals.append(d)
            hum_vals.append(h)
            diff_vals.append(h - d)
        x = [(t - 273.15 if unit.upper().startswith("C") else t) for t in Ts]
        label = AIR_TRANSPORT_LABELS.get(prop, prop)
        unit_str = AIR_TRANSPORT_UNITS.get(prop, "")
        return AirTransportTable(
            title=f"{label}: dry vs humid air",
            x_label=f"Temperature ({unit})",
            y_label=f"{label} [{unit_str}]" if unit_str else label,
            x_unit=unit,
            x=x,
            x_K=Ts,
            columns={
                "Dry air": dry_vals,
                "Humid air": hum_vals,
                "Humid - Dry": diff_vals,
            },
            meta={
                "prop": prop,
                "P": P,
                "unit": unit_str,
                "humidity": {
                    "relative_humidity": relative_humidity,
                    "humidity_ratio": humidity_ratio,
                    "mole_fraction": mole_fraction,
                    "saturated": saturated,
                },
            },
        )
