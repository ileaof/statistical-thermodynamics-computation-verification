"""Property and phase-diagram plotting (matplotlib, lazily imported).

Provides T–s, P–h, P–v and Mollier (h–s) diagrams with isotherms, isobars,
isochores and saturation curves. Curves are built by sampling
:class:`~thermolab.state.State` objects and stitching the single-phase branches
with the two-phase dome segment so that isobars/isotherms cross the dome
correctly.

Every function returns the matplotlib ``Axes`` and does not call ``show()``
unless asked, so figures compose cleanly inside larger applications.
"""

from __future__ import annotations

import numpy as np

from .backends.base import Phase
from .state import State

# Diagram axis definitions: name -> (x_attr, y_attr, log_x, log_y)
_DIAGRAMS = {
    "ts":      ("s",   "T", False, False),
    "ph":      ("h",   "P", False, True),
    "pv":      ("v",   "P", True,  True),
    "mollier": ("s",   "h", False, False),
}


def _get_axes(ax=None, log_x=False, log_y=False):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    return ax


def _label_axes(ax, diagram):
    x_attr, y_attr, _, _ = _DIAGRAMS[diagram]
    labels = {"s": "s [J/(kg.K)]", "h": "h [J/kg]", "T": "T [K]",
              "P": "P [Pa]", "v": "v [m^3/kg]"}
    ax.set_xlabel(labels[x_attr])
    ax.set_ylabel(labels[y_attr])
    titles = {"ts": "T–s diagram", "ph": "P–h diagram",
              "pv": "P–v diagram", "mollier": "Mollier (h–s) diagram"}
    ax.set_title(titles[diagram])


def _single_phase_state(fluid, T, P, phase):
    """Build a single-phase State directly (no flash)."""
    return State(fluid.backend, fluid.fractions, T, P, phase)


def _sat_endpoints(fluid, T, P):
    """Return (liquid_state, vapor_state) on the saturation curve at (T, P)."""
    liq = _single_phase_state(fluid, T, P, Phase.LIQUID)
    vap = _single_phase_state(fluid, T, P, Phase.VAPOR)
    return liq, vap


def _xy(state, x_attr, y_attr):
    return getattr(state, x_attr), getattr(state, y_attr)


# ---------------------------------------------------------------------------
# Saturation curve
# ---------------------------------------------------------------------------
def plot_saturation(fluid, T_range=None, *, diagram="ts", ax=None, n=80, **kw):
    """Plot the saturated liquid/vapor boundary (dome) for a pure fluid."""
    x_attr, y_attr, log_x, log_y = _DIAGRAMS[diagram]
    ax = _get_axes(ax, log_x, log_y)
    be = fluid.backend
    if fluid.fractions.size != 1:
        # mixtures: approximate dome via bubble/dew at fixed T
        return _plot_saturation_mixture(fluid, T_range, diagram, ax, n, kw)
    Tc = be.critical_temperature(fluid.fractions)
    if T_range is None:
        T_range = (max(100.0, 0.5 * Tc), 0.999 * Tc)
    T_lo, T_hi = T_range
    # The saturation solver Fortran-STOPS (uncatchable from Python) for T >= Tc,
    # so the upper bound must stay strictly below the critical temperature.
    T_hi = min(T_hi, 0.999 * Tc)
    T_lo = min(T_lo, T_hi)
    Ts = np.linspace(T_lo, T_hi, n)
    xf, yf, xg, yg = [], [], [], []
    for T in Ts:
        try:
            sat = be.saturation_state(float(T), fluid.fractions)
            liq = _single_phase_state(fluid, float(T), sat.P, Phase.LIQUID)
            vap = _single_phase_state(fluid, float(T), sat.P, Phase.VAPOR)
            xf.append(getattr(liq, x_attr)); yf.append(getattr(liq, y_attr))
            xg.append(getattr(vap, x_attr)); yg.append(getattr(vap, y_attr))
        except Exception:
            continue
    kwf = {**dict(color="C0", lw=1.5), **kw}
    ax.plot(xf, yf, **kwf, label="saturated liquid")
    ax.plot(xg[::-1], yg[::-1], **{**kwf, "color": "C1"}, label="saturated vapor")
    ax.legend(fontsize=8)
    _label_axes(ax, diagram)
    return ax


def _plot_saturation_mixture(fluid, T_range, diagram, ax, n, kw):
    x_attr, y_attr, log_x, log_y = _DIAGRAMS[diagram]
    be = fluid.backend
    if T_range is None:
        Tc = be.critical_temperature(fluid.fractions)
        T_range = (0.5 * Tc, 0.999 * Tc)
    Ts = np.linspace(T_range[0], T_range[1], n)
    xb, yb, xd, yd = [], [], [], []
    z = fluid.fractions
    for T in Ts:
        try:
            Pb = float(be._engine.bubble_pressure(float(T), z)[0])
            Pd = float(be._engine.dew_pressure(float(T), z)[0])
            lb = _single_phase_state(fluid, float(T), Pb, Phase.LIQUID)
            vd = _single_phase_state(fluid, float(T), Pd, Phase.VAPOR)
            xb.append(getattr(lb, x_attr)); yb.append(getattr(lb, y_attr))
            xd.append(getattr(vd, x_attr)); yd.append(getattr(vd, y_attr))
        except Exception:
            continue
    ax.plot(xb, yb, color="C0", lw=1.5, label="bubble")
    ax.plot(xd, yd, color="C1", lw=1.5, label="dew")
    ax.legend(fontsize=8)
    _label_axes(ax, diagram)
    return ax


# ---------------------------------------------------------------------------
# Isobars / isotherms / isochores
# ---------------------------------------------------------------------------
def _isobar_points(fluid, P, T_range, x_attr, y_attr, n=60):
    """Sample an isobar (constant P) stitching liquid + dome + vapor."""
    be = fluid.backend
    z = fluid.fractions
    pts = []
    try:
        Tsat = be.saturation_temperature(P, z) if be.critical_pressure(z) > P else None
    except Exception:
        Tsat = None
    T_lo, T_hi = T_range
    if Tsat is not None and T_lo < Tsat < T_hi:
        # liquid branch
        for T in np.linspace(T_lo, Tsat, n // 2):
            try:
                st = _single_phase_state(fluid, float(T), P, Phase.LIQUID)
                pts.append(_xy(st, x_attr, y_attr))
            except Exception:
                pass
        # dome segment
        try:
            sat = be.saturation_state(Tsat, z)
            liq = _single_phase_state(fluid, Tsat, P, Phase.LIQUID)
            vap = _single_phase_state(fluid, Tsat, P, Phase.VAPOR)
            pts.append(_xy(liq, x_attr, y_attr))
            pts.append(_xy(vap, x_attr, y_attr))
        except Exception:
            pass
        # vapor branch
        for T in np.linspace(Tsat, T_hi, n // 2):
            try:
                st = _single_phase_state(fluid, float(T), P, Phase.VAPOR)
                pts.append(_xy(st, x_attr, y_attr))
            except Exception:
                pass
    else:
        phase = Phase.VAPOR if Tsat is None else (Phase.VAPOR if T_lo > Tsat else Phase.LIQUID)
        for T in np.linspace(T_lo, T_hi, n):
            try:
                st = _single_phase_state(fluid, float(T), P, phase)
                pts.append(_xy(st, x_attr, y_attr))
            except Exception:
                pass
    return tuple(zip(*pts)) if pts else ([], [])


def _isotherm_points(fluid, T, P_range, x_attr, y_attr, n=60):
    """Sample an isotherm (constant T) stitching compressed-liquid + dome + vapor."""
    be = fluid.backend
    z = fluid.fractions
    pts = []
    try:
        Psat = be.saturation_pressure(T, z) if be.critical_temperature(z) > T else None
    except Exception:
        Psat = None
    P_lo, P_hi = P_range
    if Psat is not None and P_lo < Psat < P_hi:
        # vapor branch (P < Psat)
        for P in np.linspace(P_lo, Psat, n // 2):
            try:
                st = _single_phase_state(fluid, T, float(P), Phase.VAPOR)
                pts.append(_xy(st, x_attr, y_attr))
            except Exception:
                pass
        try:
            liq = _single_phase_state(fluid, T, Psat, Phase.LIQUID)
            vap = _single_phase_state(fluid, T, Psat, Phase.VAPOR)
            pts.append(_xy(vap, x_attr, y_attr))
            pts.append(_xy(liq, x_attr, y_attr))
        except Exception:
            pass
        # compressed liquid branch (P > Psat)
        for P in np.linspace(Psat, P_hi, n // 2):
            try:
                st = _single_phase_state(fluid, T, float(P), Phase.LIQUID)
                pts.append(_xy(st, x_attr, y_attr))
            except Exception:
                pass
    else:
        phase = Phase.VAPOR if (Psat is None or P_hi < Psat) else Phase.LIQUID
        for P in np.linspace(P_lo, P_hi, n):
            try:
                st = _single_phase_state(fluid, T, float(P), phase)
                pts.append(_xy(st, x_attr, y_attr))
            except Exception:
                pass
    return tuple(zip(*pts)) if pts else ([], [])


def plot_isobars(fluid, pressures, T_range=None, *, diagram="ts", ax=None, n=60, **kw):
    """Plot a family of isobars (constant-pressure lines)."""
    x_attr, y_attr, log_x, log_y = _DIAGRAMS[diagram]
    ax = _get_axes(ax, log_x, log_y)
    if T_range is None:
        T_range = (250.0, 1500.0)
    for i, P in enumerate(pressures):
        xs, ys = _isobar_points(fluid, P, T_range, x_attr, y_attr, n=n)
        if len(xs):
            ax.plot(xs, ys, color=f"C{i % 10}", lw=1.0,
                    label=f"{P/1e5:g} bar", **kw)
    ax.legend(fontsize=7, title="isobars")
    _label_axes(ax, diagram)
    return ax


def plot_isotherms(fluid, temperatures, P_range=None, *, diagram="ts", ax=None, n=60, **kw):
    """Plot a family of isotherms (constant-temperature lines)."""
    x_attr, y_attr, log_x, log_y = _DIAGRAMS[diagram]
    ax = _get_axes(ax, log_x, log_y)
    if P_range is None:
        P_range = (1e4, 1e7)
    for i, T in enumerate(temperatures):
        xs, ys = _isotherm_points(fluid, T, P_range, x_attr, y_attr, n=n)
        if len(xs):
            ax.plot(xs, ys, color=f"C{i % 10}", lw=1.0, ls="--",
                    label=f"{T:g} K", **kw)
    ax.legend(fontsize=7, title="isotherms")
    _label_axes(ax, diagram)
    return ax


def plot_isochores(fluid, densities, T_range=None, *, diagram="ts", ax=None, n=60, **kw):
    """Plot a family of isochores (constant-density lines) via ``(T, rho)`` specs."""
    x_attr, y_attr, log_x, log_y = _DIAGRAMS[diagram]
    ax = _get_axes(ax, log_x, log_y)
    if T_range is None:
        T_range = (250.0, 1500.0)
    for i, rho in enumerate(densities):
        xs, ys = [], []
        for T in np.linspace(T_range[0], T_range[1], n):
            try:
                st = fluid.state(T=float(T), rho=float(rho))
                xs.append(getattr(st, x_attr)); ys.append(getattr(st, y_attr))
            except Exception:
                continue
        if len(xs):
            ax.plot(xs, ys, color=f"C{i % 10}", lw=1.0, ls=":",
                    label=f"rho={rho:g}", **kw)
    ax.legend(fontsize=7, title="isochores")
    _label_axes(ax, diagram)
    return ax


# ---------------------------------------------------------------------------
# High-level diagrams
# ---------------------------------------------------------------------------
def plot_ts(fluid, *, isobars=None, isotherms=None, T_range=None, ax=None, **kw):
    """Draw a T–s diagram (with optional isobars/isotherms and saturation)."""
    ax = _get_axes(ax)
    try:
        plot_saturation(fluid, T_range, diagram="ts", ax=ax)
    except Exception:
        pass
    if isobars is not None:
        plot_isobars(fluid, isobars, T_range, diagram="ts", ax=ax)
    if isotherms is not None:
        plot_isotherms(fluid, isotherms, diagram="ts", ax=ax)
    _label_axes(ax, "ts")
    return ax


def plot_ph(fluid, *, isotherms=None, T_range=None, ax=None, **kw):
    """Draw a P–h diagram (with optional isotherms and saturation)."""
    ax = _get_axes(ax, log_y=True)
    try:
        plot_saturation(fluid, T_range, diagram="ph", ax=ax)
    except Exception:
        pass
    if isotherms is not None:
        plot_isotherms(fluid, isotherms, diagram="ph", ax=ax)
    _label_axes(ax, "ph")
    return ax


def plot_pv(fluid, *, isotherms=None, T_range=None, ax=None, **kw):
    """Draw a P–v diagram (log–log, with optional isotherms and saturation)."""
    ax = _get_axes(ax, log_x=True, log_y=True)
    try:
        plot_saturation(fluid, T_range, diagram="pv", ax=ax)
    except Exception:
        pass
    if isotherms is not None:
        plot_isotherms(fluid, isotherms, diagram="pv", ax=ax)
    _label_axes(ax, "pv")
    return ax


def plot_mollier(fluid, *, isobars=None, T_range=None, ax=None, **kw):
    """Draw a Mollier h–s diagram (with optional isobars and saturation)."""
    ax = _get_axes(ax)
    try:
        plot_saturation(fluid, T_range, diagram="mollier", ax=ax)
    except Exception:
        pass
    if isobars is not None:
        plot_isobars(fluid, isobars, T_range, diagram="mollier", ax=ax)
    _label_axes(ax, "mollier")
    return ax


def _sample_isobar(backend, z, P, T_lo, T_hi, x_attr, y_attr, n=40):
    """Sample an isobar (constant ``P``) from ``T_lo`` to ``T_hi``.

    Each intermediate temperature is flashed at ``(T, P)`` so the curve follows
    the real thermodynamic path — compressed liquid up to the saturated-liquid
    line, across the dome, then superheated vapour — instead of a straight
    chord between the endpoints.
    """
    from .flash import flash
    if not np.isfinite(T_lo) or not np.isfinite(T_hi) or T_lo <= 0 or T_hi <= 0:
        return []
    T_lo, T_hi = sorted((float(T_lo), float(T_hi)))
    pts = []
    for T in np.linspace(T_lo, T_hi, n):
        try:
            fr = flash(backend, z, {"T": float(T), "P": float(P)})
            st = State(backend, z, fr.T, fr.P, fr.phase,
                       two_phase=fr.two_phase, quality=fr.quality, sat=fr.sat)
            x, y = float(getattr(st, x_attr)), float(getattr(st, y_attr))
            if np.isfinite(x) and np.isfinite(y):
                pts.append((x, y))
        except Exception:
            continue
    return pts


def plot_cycle(cycle_result, *, diagram="ts", ax=None, **kw):
    """Overlay a :class:`~thermolab.cycles.CycleResult` on a diagram.

    Each process segment is drawn as its real thermodynamic curve where the
    constraint is known: constant-pressure legs (boiler, condenser) are sampled
    along the isobar through the saturation dome, while isentropic/isothermal
    legs are straight. A faint saturation dome is added for pure-fluid context.
    """
    x_attr, y_attr, log_x, log_y = _DIAGRAMS[diagram]
    ax = _get_axes(ax, log_x, log_y)

    # --- Faint saturation dome for context (pure fluid) -----------------
    pts = [p for p in cycle_result.points if p.state is not None]
    if pts:
        ref = pts[0].state
        try:
            if ref._z.size == 1:
                plot_saturation(_StateAsFluid(ref), diagram=diagram, ax=ax,
                                n=60, color="0.7", lw=1.0)
        except Exception:
            pass

    # --- Cycle path: sample each segment by its constraint --------------
    xs, ys = [], []
    for i in range(len(pts)):
        a = pts[i].state
        b = pts[(i + 1) % len(pts)].state
        seg = _segment_curve(a, b, x_attr, y_attr)
        if i == 0 and seg:
            xs.append(seg[0][0]); ys.append(seg[0][1])
        xs.extend(p[0] for p in seg[1:])
        ys.extend(p[1] for p in seg[1:])

    if xs:
        ax.plot(xs, ys, "-o", color="red", lw=2, ms=5, label=cycle_result.name)
        for pt in pts:
            ax.annotate(pt.label, (getattr(pt.state, x_attr), getattr(pt.state, y_attr)),
                        fontsize=9, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points")
        ax.legend(fontsize=8)
    _label_axes(ax, diagram)
    return ax


def _segment_curve(a, b, x_attr, y_attr, n_iso=2):
    """Interpolate one process segment between corner states ``a`` and ``b``.

    Detects the constraint from the endpoint values:
    * equal pressure  -> isobar, sampled through the dome;
    * equal entropy   -> isentrope (straight);
    * equal temperature -> isotherm (straight);
    * otherwise       -> straight chord.
    Returns a list of ``(x, y)`` points (excluding a duplicate of ``a`` when
    sampling, so callers can stitch segments cleanly).
    """
    def _close(u, v, rel=1e-4):
        return abs(u - v) <= rel * max(abs(u), abs(v), 1.0)

    if _close(a.P, b.P) and not _close(a.T, b.T):
        seg = _sample_isobar(a._backend, a._z, a.P, a.T, b.T, x_attr, y_attr)
        if len(seg) >= 2:
            # ensure the segment starts exactly at `a`'s coordinates
            seg[0] = (float(getattr(a, x_attr)), float(getattr(a, y_attr)))
            seg[-1] = (float(getattr(b, x_attr)), float(getattr(b, y_attr)))
            return seg
    # isentrope / isotherm / unknown -> straight line through the two corners
    xa, ya = float(getattr(a, x_attr)), float(getattr(a, y_attr))
    xb, yb = float(getattr(b, x_attr)), float(getattr(b, y_attr))
    return [(xa, ya)] + [(xa + (xb - xa) * k / n_iso,
                          ya + (yb - ya) * k / n_iso) for k in range(1, n_iso + 1)]


class _StateAsFluid:
    """Adapter exposing the minimal fluid interface ``plot_saturation`` needs
    (``backend`` and ``fractions``) from a single :class:`State`."""

    def __init__(self, state):
        self.backend = state.backend
        self.fractions = state._z