"""Plotting smoke tests (headless Agg)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thermolab import Gas
from thermolab import plotting as P


def _has_lines(ax):
    return sum(len(ln.get_data()[0]) for ln in ax.lines) > 0


def test_plot_ts(air):
    ax = P.plot_ts(air, isobars=[1e5, 5e5], T_range=(250, 1000))
    assert _has_lines(ax)
    plt.close(ax.figure)


def test_plot_ph(water):
    ax = P.plot_ph(water, isotherms=[400, 500], T_range=(300, 600))
    assert _has_lines(ax)
    plt.close(ax.figure)


def test_plot_pv(water):
    ax = P.plot_pv(water, isotherms=[400, 500], T_range=(300, 600))
    assert _has_lines(ax)
    plt.close(ax.figure)


def test_plot_mollier(water):
    ax = P.plot_mollier(water, isobars=[1e5, 5e5], T_range=(300, 600))
    assert _has_lines(ax)
    plt.close(ax.figure)


def test_plot_isotherms(air):
    ax = P.plot_isotherms(air, [300, 500, 800], P_range=(1e4, 1e6))
    assert _has_lines(ax)
    plt.close(ax.figure)


def test_plot_isochores(air):
    ax = P.plot_isochores(air, [1.0, 5.0], T_range=(300, 800))
    assert _has_lines(ax)
    plt.close(ax.figure)


def test_plot_saturation(water):
    ax = P.plot_saturation(water, T_range=(300, 600), diagram="ts")
    assert _has_lines(ax)
    plt.close(ax.figure)