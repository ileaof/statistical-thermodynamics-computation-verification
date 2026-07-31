"""Property-vs-temperature plots.

Generates the curves requested by the specification — Cp, Cv, H, S, G, A, U, γ and the four
partition-function factors Qt, Qr, Qv, Qe — all versus temperature, from the first-principles
engine. Matplotlib is used; figures are returned so the caller can save or show them.

A non-interactive backend is selected by default so plots can be generated headless; the caller
can switch backends before calling :func:`show`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..core.state import State
from ..database import get
from ..thermodynamics import Thermodynamics

# Matplotlib is imported lazily so the package core (and the CLI) does not require it.
# A non-interactive backend is selected on first import so plots can be generated headless.
plt = None


def _get_pyplot():
    """Lazily import matplotlib.pyplot with a headless backend."""
    global plt
    if plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt

__all__ = [
    "plot_property",
    "plot_mixture_property",
    "plot_all_properties",
    "MOLAR_PROPS",
    "PARTITION_PROPS",
    "MIXTURE_PROPS",
]

#: Molar thermodynamic properties available for plotting.
MOLAR_PROPS: list[str] = ["U_m", "H_m", "S_m", "A_m", "G_m", "Cv_m", "Cp_m", "gamma"]
#: Partition-function factors available for plotting.
PARTITION_PROPS: list[str] = ["Qt", "Qr", "Qv", "Qe", "Qtotal"]
#: Properties available for an ideal-gas *mixture*. The partition-function factors are
#: per-species quantities and are therefore not defined for a mixture.
MIXTURE_PROPS: list[str] = list(MOLAR_PROPS)


def _ensure_molecule(molecule):
    if isinstance(molecule, str):
        return get(molecule)
    return molecule


def plot_property(
    molecule,
    prop: str,
    T_range: Iterable[float],
    P: float = 101325.0,
    *,
    ax=None,
    label: str | None = None,
    logy: bool = False,
):
    """Plot a single property versus temperature.

    Parameters
    ----------
    molecule : str | Molecule
        Molecule name or instance.
    prop : str
        Attribute of :class:`~statthermopy.thermodynamics.ThermoProperties`
        (e.g. ``"Cp_m"``, ``"Qtotal"``).
    T_range : iterable of float
        Temperatures (K).
    P : float, default 101325.0
        Pressure (Pa).
    ax : matplotlib.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional
        Line label.
    logy : bool, default False
        Use a logarithmic y-axis (sensible for partition functions).
    """
    mol = _ensure_molecule(molecule)
    Ts = list(T_range)
    vals = []
    for T in Ts:
        th = Thermodynamics(mol, State(T=float(T), P=P)).compute()
        vals.append(getattr(th, prop))
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ts, vals, label=label or f"{mol.name}: {prop}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(prop)
    ax.set_title(f"{mol.formula} — {prop} vs T @ {P/1e3:.1f} kPa")
    if label:
        ax.legend()
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return ax


def plot_mixture_property(
    mixture,
    prop: str,
    T_range: Iterable[float],
    P: float = 101325.0,
    *,
    ax=None,
    label: str | None = None,
    logy: bool = False,
):
    """Plot a single property of an ideal-gas mixture versus temperature.

    Mirrors :func:`plot_property` but operates on an
    :class:`~statthermopy.mixture.IdealGasMixture`, evaluating the mixture at each
    temperature (each component at its partial pressure ``P_i = x_i P``).

    Parameters
    ----------
    mixture : IdealGasMixture
        The mixture to evaluate.
    prop : str
        A molar property in :data:`MIXTURE_PROPS` (e.g. ``"Cp_m"``, ``"gamma"``).
        Partition-function factors are per-species and are not available here.
    T_range : iterable of float
        Temperatures (K).
    P : float, default 101325.0
        Total pressure (Pa).
    ax, label, logy
        As in :func:`plot_property`.
    """
    if prop not in MIXTURE_PROPS:
        raise ValueError(
            f"{prop!r} is not available for a mixture; choose one of {MIXTURE_PROPS}."
        )
    Ts = list(T_range)
    vals = []
    for T in Ts:
        mp = mixture.compute(State(T=float(T), P=P))
        vals.append(getattr(mp, prop))
    plt = _get_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    comp = ", ".join(f"{mol.name} {xi:.2f}" for mol, xi in mixture.x.items())
    ax.plot(Ts, vals, label=label or f"mixture: {prop}")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(prop)
    ax.set_title(f"{comp} — {prop} vs T @ {P/1e3:.1f} kPa")
    if label:
        ax.legend()
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return ax


def plot_all_properties(
    molecule,
    T_range: Iterable[float],
    P: float = 101325.0,
    *,
    save_dir: str | Path | None = None,
):
    """Generate and optionally save every requested property-vs-T plot.

    Returns a dict mapping property name to matplotlib :class:`~matplotlib.axes.Axes`.
    """
    mol = _ensure_molecule(molecule)
    axes: dict[str, object] = {}
    plt = _get_pyplot()
    for prop in MOLAR_PROPS + PARTITION_PROPS:
        logy = prop in PARTITION_PROPS
        ax = plot_property(mol, prop, T_range, P=P, logy=logy)
        axes[prop] = ax
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(save_dir / f"{mol.name}_{prop}.png", dpi=120, bbox_inches="tight")
    return axes


def show() -> None:
    """Display all open figures (call after plotting in an interactive backend)."""
    _get_pyplot().show()