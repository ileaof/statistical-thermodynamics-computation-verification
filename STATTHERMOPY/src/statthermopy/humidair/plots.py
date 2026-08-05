"""Curves and surfaces for the Statistical Humid Air module.

All generators return matplotlib ``Axes`` (2-D) or ``Axes3D`` (surface) so the caller can save,
show or restyle them. Matplotlib is imported lazily with a headless backend, mirroring
:mod:`statthermopy.plots`.
"""

from __future__ import annotations

from collections.abc import Iterable

from .analysis import PsychrometricAnalysis
from .humidair import HumidAir

__all__ = [
    "plot_saturation_pressure_vs_T",
    "plot_max_solubility_vs_T",
    "plot_max_solubility_vs_P",
    "plot_humidity_ratio_vs_T",
    "plot_relative_humidity_vs_T",
    "plot_dew_point_vs_P",
    "plot_solubility_surface",
    "plot_water_vapor_content_vs_T",
    "plot_property_comparison",
    "plot_thermal_fields_comparison",
    "make_pickable_legend",
    "add_hover_tooltip",
]

# Colour-blind-safe palette: dry vs humid by hue, isobaric vs isochoric by line style.
_DRY = "#0072B2"
_HUMID = "#D55E00"

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


def _model(model: HumidAir | None) -> HumidAir:
    return model if model is not None else HumidAir()


def plot_saturation_pressure_vs_T(T_range: Iterable[float], *, model=None, ax=None, logy=True):
    """Water saturation pressure ``P_sat(T)`` (Pa) versus temperature (K)."""
    m = _model(model)
    Ts = [float(t) for t in T_range]
    ps = [m.saturation_pressure(t) for t in Ts]
    ax = _new_ax(ax)
    ax.plot(Ts, ps, color="#0072B2", label="P_sat (statistical vapour)")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Saturation pressure [Pa]")
    ax.set_title("Water saturation pressure vs temperature")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_max_solubility_vs_T(T_range: Iterable[float], P: float = 101325.0, *, model=None, ax=None):
    """Maximum H₂O mole fraction (solubility) versus temperature at fixed total pressure."""
    m = _model(model)
    Ts = [float(t) for t in T_range]
    x = [m.max_mole_fraction(t, P) for t in Ts]
    ax = _new_ax(ax)
    ax.plot(Ts, x, color="#D55E00", label=f"x_H2O,max @ {P/1e3:.1f} kPa")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Max H2O mole fraction [-]")
    ax.set_title("Maximum water-vapour solubility vs temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_max_solubility_vs_P(P_range: Iterable[float], T: float = 298.15, *, model=None, ax=None):
    """Maximum H₂O mole fraction versus total pressure at fixed temperature."""
    m = _model(model)
    Ps = [float(p) for p in P_range]
    x = [m.max_mole_fraction(T, p) for p in Ps]
    ax = _new_ax(ax)
    ax.plot([p / 1e3 for p in Ps], x, color="#009E73", label=f"x_H2O,max @ {T:.1f} K")
    ax.set_xlabel("Total pressure [kPa]")
    ax.set_ylabel("Max H2O mole fraction [-]")
    ax.set_title("Maximum water-vapour solubility vs pressure")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_humidity_ratio_vs_T(T_range: Iterable[float], P: float = 101325.0, *, model=None, ax=None):
    """Saturation humidity ratio ``w_s(T)`` (kg vapour / kg dry air) versus temperature.

    Computed directly from ``w_s = ε P_sat/(P − P_sat)`` and reported as a dimensionless mass ratio
    (**g/kg dry air**); above the boiling point (``P_sat ≥ P``) no saturated humid air exists, so
    those points are blank.
    """
    m = _model(model)
    eps = m.epsilon
    Ts = [float(t) for t in T_range]
    w = []
    for t in Ts:
        ps = m.saturation_pressure(t)
        w.append(eps * ps / (P - ps) if ps < P else float("nan"))
    ax = _new_ax(ax)
    ax.plot(Ts, w, color="#CC79A7", label=f"w_sat @ {P/1e3:.1f} kPa")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Saturation humidity ratio [g/kg dry air]")
    ax.set_title("Maximum humidity ratio vs temperature")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_relative_humidity_vs_T(
    T_range: Iterable[float], humidity_ratio: float, P: float = 101325.0, *, model=None, ax=None
):
    """Relative humidity versus temperature at a fixed humidity ratio (constant water content).

    At fixed water content the vapour partial pressure ``P_v`` is constant, so
    ``RH(T) = P_v / P_sat(T)`` (capped at 1); computed directly from ``P_sat(T)``.
    """
    m = _model(model)
    eps = m.epsilon
    r = humidity_ratio / eps                 # P_v/(P - P_v)
    P_v = r * P / (1.0 + r)                   # fixed water partial pressure
    Ts = [float(t) for t in T_range]
    rh = [min(P_v / m.saturation_pressure(t), 1.0) for t in Ts]
    ax = _new_ax(ax)
    ax.plot(Ts, rh, color="#0072B2", label=f"RH @ w={humidity_ratio*1e3:.1f} g/kg")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Relative humidity [-]")
    ax.set_title("Relative humidity vs temperature (fixed water content)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_dew_point_vs_P(
    P_range: Iterable[float], mole_fraction: float, *, model=None, ax=None
):
    """Dew-point temperature versus total pressure at a fixed H₂O mole fraction."""
    m = _model(model)
    Ps = [float(p) for p in P_range]
    dew = [m.saturation.dew_point(mole_fraction * p) for p in Ps]
    ax = _new_ax(ax)
    ax.plot([p / 1e3 for p in Ps], dew, color="#D55E00",
            label=f"dew point @ x_H2O={mole_fraction:.4f}")
    ax.set_xlabel("Total pressure [kPa]")
    ax.set_ylabel("Dew point [K]")
    ax.set_title("Dew point vs pressure")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_solubility_surface(
    T_range: Iterable[float], P_range: Iterable[float], *, model=None, ax=None, quantity="x"
):
    """3-D surface of the maximum water solubility over the (T, P) plane.

    ``quantity="x"`` plots the maximum H₂O mole fraction; ``"w"`` the saturation humidity ratio.
    """
    import numpy as np

    m = _model(model)
    Ts = np.asarray([float(t) for t in T_range], dtype=float)
    Ps = np.asarray([float(p) for p in P_range], dtype=float)
    TT, PP = np.meshgrid(Ts, Ps)
    Z = np.empty_like(TT)
    eps = m.epsilon
    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            t, p = float(TT[i, j]), float(PP[i, j])
            if quantity == "w":
                # saturation humidity ratio in g/kg dry air (matches the rest of the module),
                # computed directly from P_sat (fast; no full state() per grid point)
                ps = m.saturation_pressure(t)
                Z[i, j] = eps * ps / (p - ps) * 1e3 if ps < p else float("nan")
            else:
                Z[i, j] = m.max_mole_fraction(t, p)
    pyplot = _get_pyplot()
    if ax is None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
        fig = pyplot.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(TT, PP / 1e3, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_zlabel("w_sat [g/kg]" if quantity == "w" else "x_H2O,max [-]")
    ax.set_title("Maximum water solubility over (T, P)")
    return ax


# ---------------------------------------------------------------------------
# Comparative analysis plots (return the ComparisonTable so callers can export it)
# ---------------------------------------------------------------------------

def _analysis(source) -> PsychrometricAnalysis:
    if isinstance(source, PsychrometricAnalysis):
        return source
    if isinstance(source, HumidAir):
        return PsychrometricAnalysis(source)
    return PsychrometricAnalysis(source)


def plot_water_vapor_content_vs_T(
    source, T_range: Iterable[float], P: float = 101325.0, *,
    ax=None, temperature_unit: str = "K", interactive: bool = False, **humidity,
):
    """Water-vapour content (g/kg dry air) vs temperature: actual and saturation curves.

    ``source`` is a :class:`~statthermopy.humidair.HumidAir` or
    :class:`~statthermopy.humidair.analysis.PsychrometricAnalysis`. The actual content is fixed by
    the humidity keyword (``relative_humidity`` / ``humidity_ratio`` / ``mole_fraction``); the
    saturation curve is the maximum. The dew point (onset of condensation) is marked and the
    condensation region shaded. Returns the
    :class:`~statthermopy.humidair.analysis.ComparisonTable` (the numerical data, for export).
    """
    table = _analysis(source).water_vapor_content(
        T_range, P, temperature_unit=temperature_unit, **humidity
    )
    ax = _new_ax(ax)
    x = table.x
    w_act = table.columns["actual w [g/kg]"]
    w_sat = table.columns["saturation w_sat [g/kg]"]
    ax.plot(x, w_sat, color=_HUMID, ls="--", label="saturation w_sat (max)")
    ax.plot(x, w_act, color=_DRY, lw=2.0, label="actual w")
    dew = table.meta.get("dew_point")
    if dew is not None and dew == dew:  # not NaN
        ax.axvline(dew, color="0.5", ls=":", lw=1.2)
        ymax = max((v for v in w_sat if v == v), default=1.0)
        ax.annotate("onset of condensation\n(dew point)", xy=(dew, 0.0),
                    xytext=(dew, 0.35 * ymax), fontsize=8, ha="center", color="0.35",
                    arrowprops={"arrowstyle": "->", "color": "0.5"})
        ax.axvspan(min(x), dew, color="#0072B2", alpha=0.06)
    ax.set_xlabel(table.x_label)
    ax.set_ylabel(table.y_label)
    ax.set_title(table.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if interactive:
        make_pickable_legend(ax)
        add_hover_tooltip(ax)
    return table, ax


def plot_property_comparison(
    source, prop_field: str, T_range: Iterable[float], P: float = 101325.0, *,
    ax=None, temperature_unit: str = "K", isobaric: bool = True, isochoric: bool = True,
    interactive: bool = False, **humidity,
):
    """Dry vs humid air comparison of a property vs temperature, isobaric and isochoric.

    Up to four curves (dry/humid × const-P/const-V); dry vs humid distinguished by colour,
    isobaric vs isochoric by line style. For temperature-only properties the const-P and const-V
    curves coincide (a note is added). Returns ``(ComparisonTable, ax)``.
    """
    table = _analysis(source).property_comparison(
        prop_field, T_range, P, temperature_unit=temperature_unit,
        isobaric=isobaric, isochoric=isochoric, **humidity,
    )
    ax = _new_ax(ax)
    styles = {
        "Dry air — const P": {"color": _DRY, "ls": "-", "lw": 1.8},
        "Humid air — const P": {"color": _HUMID, "ls": "-", "lw": 1.8},
        "Dry air — const V": {"color": _DRY, "ls": "--", "lw": 1.8},
        "Humid air — const V": {"color": _HUMID, "ls": "--", "lw": 1.8},
    }
    for label, ys in table.columns.items():
        ax.plot(table.x, ys, label=label, **styles.get(label, {}))
    ax.set_xlabel(table.x_label)
    ax.set_ylabel(table.y_label)
    title = table.title
    if table.meta.get("pressure_independent"):
        title += "  (const-P ≡ const-V: temperature-only)"
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if interactive:
        make_pickable_legend(ax)
        add_hover_tooltip(ax)
    return table, ax


def plot_thermal_fields_comparison(
    source, T_range: Iterable[float], P: float = 101325.0, *,
    ax=None, temperature_unit: str = "K", interactive: bool = False, **humidity,
):
    """Thermal fields of dry vs humid air: constant-volume ``T_v`` and constant-pressure ``T_p``.

    Four curves in kelvin — ``T_v = U_m/Cv_m`` (const V) and ``T_p = H_m/Cp_m`` (const P) for dry
    and humid air, **each from its own mixture's properties**. Dry vs humid are distinguished by
    colour, ``T_v`` vs ``T_p`` by line style. Returns ``(ComparisonTable, ax)``.
    """
    table = _analysis(source).thermal_fields_comparison(
        T_range, P, temperature_unit=temperature_unit, **humidity
    )
    ax = _new_ax(ax)
    styles = {
        "Dry air T_v (const V)": {"color": _DRY, "ls": "-", "lw": 1.8},
        "Humid air T_v (const V)": {"color": _HUMID, "ls": "-", "lw": 1.8},
        "Dry air T_p (const P)": {"color": _DRY, "ls": "--", "lw": 1.8},
        "Humid air T_p (const P)": {"color": _HUMID, "ls": "--", "lw": 1.8},
    }
    for label, ys in table.columns.items():
        ax.plot(table.x, ys, label=label, **styles.get(label, {}))
    ax.set_xlabel(table.x_label)
    ax.set_ylabel(table.y_label)
    ax.set_title(table.title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    # the dry/humid fields differ by only a fraction of a K over a wide T range, so quantify the
    # spread on the graph (they are genuinely distinct — each from its own mixture's properties)
    dtv = table.meta.get("max_diff_Tv")
    dtp = table.meta.get("max_diff_Tp")
    if dtv is not None and dtp is not None:
        ax.text(0.02, 0.98,
                f"max |humid − dry|:  T_v {dtv:.3f} K,  T_p {dtp:.3f} K\n"
                "(each field from its own mixture's U_m/H_m/Cv_m/Cp_m)",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color="0.3",
                bbox={"boxstyle": "round,pad=0.3", "fc": "#f4f4f4", "ec": "0.7"})
    if interactive:
        make_pickable_legend(ax)
        add_hover_tooltip(ax)
    return table, ax


# ---------------------------------------------------------------------------
# Interactivity: click-to-toggle legend + hover tooltip (live canvases only)
# ---------------------------------------------------------------------------

def make_pickable_legend(ax) -> None:
    """Make the legend clickable: clicking an entry toggles that curve's visibility."""
    leg = ax.get_legend()
    if leg is None:
        return
    by_label = {}
    for line in ax.get_lines():
        by_label.setdefault(line.get_label(), line)
    legline_to_line = {}
    for legline in leg.get_lines():
        legline.set_picker(True)
        legline.set_pickradius(8)
        legline_to_line[legline] = by_label.get(legline.get_label())

    def on_pick(event):  # pragma: no cover - requires a live canvas / user interaction
        legline = event.artist
        line = legline_to_line.get(legline)
        if line is None:
            return
        visible = not line.get_visible()
        line.set_visible(visible)
        legline.set_alpha(1.0 if visible else 0.25)
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("pick_event", on_pick)


def add_hover_tooltip(ax) -> None:
    """Attach a hover tooltip that snaps to the nearest data point of the visible curves."""
    fig = ax.figure
    ann = ax.annotate(
        "", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.4", "fc": "#ffffe0", "ec": "0.5", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "0.5"}, fontsize=8, zorder=10,
    )
    ann.set_visible(False)

    def on_move(event):  # pragma: no cover - requires a live canvas / mouse motion
        if event.inaxes is not ax:
            if ann.get_visible():
                ann.set_visible(False)
                fig.canvas.draw_idle()
            return
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            hit, info = line.contains(event)
            if hit:
                xd, yd = line.get_data()
                i = int(info["ind"][0])
                ann.xy = (xd[i], yd[i])
                ann.set_text(f"{line.get_label()}\nT={xd[i]:.4g}, y={yd[i]:.4g}")
                ann.set_visible(True)
                fig.canvas.draw_idle()
                return
        if ann.get_visible():
            ann.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
