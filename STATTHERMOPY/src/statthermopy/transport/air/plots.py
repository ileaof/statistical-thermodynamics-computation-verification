"""Plots for the air-transport properties.

Curves versus temperature at constant pressure for the eight headline transport properties
(dynamic viscosity, kinematic viscosity, thermal conductivity, thermal diffusivity, water-vapour
diffusivity, Prandtl, Schmidt, Lewis), with dry-air, humid-air and dry-vs-humid comparison modes.
Built on :class:`.AirTransport` and :class:`.AirTransportAnalysis`; matplotlib is imported lazily
so the package core never requires it. The dry/humid palette and the interactive legend/tooltip
helpers are reused from :mod:`statthermopy.humidair.plots`.
"""

from __future__ import annotations

from collections.abc import Iterable

from ...humidair.plots import _DRY, _HUMID, add_hover_tooltip, make_pickable_legend
from .air_transport import AirTransport
from .analysis import AirTransportAnalysis
from .mixture_transport import AIR_TRANSPORT_LABELS, AIR_TRANSPORT_UNITS

__all__ = [
    "plot_air_transport",
    "plot_air_transport_comparison",
    "plot_air_transport_vs_T",
    "AIR_TRANSPORT_PLOTS",
]

plt = None


def _get_pyplot():
    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt


def _new_ax(ax):
    if ax is None:
        _, ax = _get_pyplot().subplots(figsize=(7, 4.5))
    return ax


def _ylabel(prop: str) -> str:
    unit = AIR_TRANSPORT_UNITS.get(prop, "")
    name = AIR_TRANSPORT_LABELS.get(prop, prop)
    return f"{name} [{unit}]" if unit else name


def _resolve_model(model) -> AirTransport:
    """Accept an AirTransport or an AirTransportAnalysis; return the underlying AirTransport."""
    if isinstance(model, AirTransportAnalysis):
        return model.model
    return model if model is not None else AirTransport()


def plot_air_transport(
    model,
    prop: str,
    T_range: Iterable[float],
    P: float = 101325.0,
    *,
    which: str = "comparison",
    relative_humidity: float | None = None,
    humidity_ratio: float | None = None,
    mole_fraction: float | None = None,
    saturated: bool = False,
    temperature_unit: str = "K",
    ax=None,
    interactive: bool = False,
):
    """Plot one air-transport property versus temperature.

    Parameters
    ----------
    model : AirTransport | AirTransportAnalysis | None
        The air-transport model (a default is built if ``None``).
    prop : str
        One of the headline properties (``mu, nu, k, alpha, D_eff, Pr, Sc, Le``) or any attribute
        of :class:`.MixtureTransportProperties`.
    which : str, default ``"comparison"``
        ``"dry"`` (dry air only), ``"humid"`` (humid air only) or ``"comparison"`` (both overlaid).
    humidity spec : relative_humidity / humidity_ratio / mole_fraction / saturated
        Sets the humid composition (see :meth:`AirTransport.humid`).
    interactive : bool
        Add a click-to-toggle legend and hover tooltips (GUI use).

    Returns
    -------
    (table, ax)
        The :class:`.AirTransportTable` (numerical data, for export) and the matplotlib axes.
    """
    air = _resolve_model(model)
    table = air.compare_vs_T(
        prop, T_range, P,
        relative_humidity=relative_humidity,
        humidity_ratio=humidity_ratio,
        mole_fraction=mole_fraction,
        saturated=saturated,
        temperature_unit=temperature_unit,
    )
    ax = _new_ax(ax)
    xs = table.x
    if which in ("dry", "comparison"):
        ax.plot(xs, table.columns["Dry air"], color=_DRY, lw=1.8, label="Dry air")
    if which in ("humid", "comparison"):
        ax.plot(xs, table.columns["Humid air"], color=_HUMID, lw=1.8, ls="--",
                label="Humid air")
    ax.set_xlabel(table.x_label)
    ax.set_ylabel(_ylabel(prop))
    ax.set_title(table.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if interactive:
        make_pickable_legend(ax)
        add_hover_tooltip(ax)
    return table, ax


def plot_air_transport_comparison(model, prop, T_range, P=101325.0, *, interactive=False,
                                  temperature_unit="K", **humidity):
    """Alias for :func:`plot_air_transport` with ``which="comparison"`` (dry vs humid overlay)."""
    return plot_air_transport(
        model, prop, T_range, P, which="comparison", interactive=interactive,
        temperature_unit=temperature_unit, **humidity,
    )


def plot_air_transport_vs_T(model, prop, T_range, P=101325.0, *, which="dry",
                             interactive=False, temperature_unit="K", **humidity):
    """Alias for :func:`plot_air_transport` defaulting to a single curve (``which="dry"``)."""
    return plot_air_transport(
        model, prop, T_range, P, which=which, interactive=interactive,
        temperature_unit=temperature_unit, **humidity,
    )


#: Dispatch from the eight headline property names to the plot function (GUI/CLI convenience).
AIR_TRANSPORT_PLOTS: dict[str, callable] = {prop: plot_air_transport for prop in
                                            ("mu", "nu", "k", "alpha", "D_eff", "Pr", "Sc", "Le")}