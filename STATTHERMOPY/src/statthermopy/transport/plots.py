"""Plots for the statistical transport properties.

Curves versus temperature (constant pressure), versus pressure (constant temperature),
two-dimensional T×P maps (contour fill), and multi-property overlays — all from the
Chapman–Enskog / Lennard–Jones transport engine in :mod:`statthermopy.transport.transport`.
Matplotlib is imported lazily so the package core never requires it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..core.state import State
from ..database import get
from .transport import TRANSPORT_PROPS, TRANSPORT_UNITS, TransportCalculator

__all__ = [
    "plot_transport_vs_T",
    "plot_transport_vs_P",
    "plot_transport_map",
    "plot_transport_multi",
]

# Lazy matplotlib (headless Agg) so the core import stays light.
plt = None


def _get_pyplot():
    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt


def _ensure_molecule(molecule):
    return get(molecule) if isinstance(molecule, str) else molecule


def _ylabel(prop: str) -> str:
    unit = TRANSPORT_UNITS.get(prop, "")
    return f"{prop} [{unit}]" if unit else prop


def plot_transport_vs_T(
    molecule,
    prop: str,
    T_range: Iterable[float],
    P: float = 101325.0,
    *,
    ax=None,
    label: str | None = None,
    logy: bool = False,
    color: str | None = None,
):
    """Plot a transport property versus temperature at constant pressure."""
    mol = _ensure_molecule(molecule)
    Ts, vals = TransportCalculator(mol, State(T=300.0, P=P)).property_vs_T(prop, T_range, P=P)
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ts, vals, label=label or f"{mol.name}: {prop}", color=color)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(_ylabel(prop))
    ax.set_title(f"{mol.formula} — {prop} vs T @ {P/1e3:.1f} kPa")
    ax.legend()
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return ax


def plot_transport_vs_P(
    molecule,
    prop: str,
    P_range: Iterable[float],
    T: float = 300.0,
    *,
    ax=None,
    label: str | None = None,
    logy: bool = False,
    logx: bool = True,
    color: str | None = None,
):
    """Plot a transport property versus pressure at constant temperature."""
    mol = _ensure_molecule(molecule)
    Ps, vals = TransportCalculator(mol, State(T=T, P=101325.0)).property_vs_P(prop, P_range, T=T)
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ps, vals, label=label or f"{mol.name}: {prop}", color=color)
    ax.set_xlabel("Pressure (Pa)")
    ax.set_ylabel(_ylabel(prop))
    ax.set_title(f"{mol.formula} — {prop} vs P @ {T:.1f} K")
    ax.legend()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return ax


def plot_transport_map(
    molecule,
    prop: str,
    T_range: tuple[float, float],
    P_range: tuple[float, float],
    n: int = 60,
    *,
    ax=None,
    log_P: bool = True,
):
    """Two-dimensional map (filled contour) of a property over the (T, P) plane."""
    import numpy as np

    mol = _ensure_molecule(molecule)
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    Ts = np.linspace(T_range[0], T_range[1], n)
    if log_P:
        Ps = np.logspace(math.log10(P_range[0]), math.log10(P_range[1]), n)
    else:
        Ps = np.linspace(P_range[0], P_range[1], n)
    Z = np.empty((n, n))
    calc = TransportCalculator(mol, State(T=300.0, P=101325.0))
    for i, T in enumerate(Ts):
        for j, P in enumerate(Ps):
            Z[j, i] = getattr(calc.__class__(mol, State(T=float(T), P=float(P))).compute(), prop)
    mesh = ax.pcolormesh(Ts, Ps, Z, shading="auto", cmap="viridis")
    if log_P:
        ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title(f"{mol.formula} — {prop} map")
    # add the colorbar to the axes' own figure (works whether the caller is the headless Agg
    # CLI or an embedded QtAgg canvas — avoids mixing the lazy ``plt`` figure manager).
    cb = ax.figure.colorbar(mesh, ax=ax)
    cb.set_label(_ylabel(prop))
    return ax


def plot_transport_multi(
    molecule,
    props: Iterable[str],
    T_range: Iterable[float],
    P: float = 101325.0,
    *,
    ax=None,
):
    """Overlay several transport properties versus temperature on shared axes.

    Properties with widely different magnitudes are drawn together for *qualitative*
    comparison (each curve carries its own unit in the legend); for quantitative work use the
    single-property plots.
    """
    mol = _ensure_molecule(molecule)
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for prop in props:
        if prop not in TRANSPORT_PROPS:
            raise ValueError(f"{prop!r} is not a transport property; choose one of {TRANSPORT_PROPS}.")
        plot_transport_vs_T(mol, prop, T_range, P=P, ax=ax, label=f"{prop} [{TRANSPORT_UNITS[prop]}]")
    ax.set_title(f"{mol.formula} — transport properties vs T @ {P/1e3:.1f} kPa")
    return ax
