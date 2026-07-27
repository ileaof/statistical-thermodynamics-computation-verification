"""Thermodynamic cycle tests."""

from __future__ import annotations

import pytest

from thermolab import cycles as C


def test_rankine_efficiency():
    res = C.rankine(P_boiler=8e6, P_condenser=1e4, T_superheat=773.0)
    assert 0.25 < res.eta < 0.50
    assert res.back_work_ratio < 0.05
    assert res.net_work > 0
    assert len(res.points) == 4


def test_brayton_efficiency():
    res = C.brayton(pressure_ratio=12, T3=1400)
    assert 0.3 < res.eta < 0.6
    assert res.net_work > 0
    assert 0 < res.back_work_ratio < 0.7


def test_joule_alias():
    assert C.joule is C.brayton


def test_refrigeration_cop():
    res = C.refrigeration(T_evap=263.15, T_cond=313.15)
    assert 2.0 < res.cop < 6.0
    assert res.net_work > 0


def test_otto_efficiency():
    res = C.otto(compression_ratio=8, T3=2500)
    assert 0.35 < res.eta < 0.6
    assert res.q["q_in"] > 0


def test_diesel_efficiency():
    res = C.diesel(compression_ratio=18, cutoff_ratio=2.0)
    assert 0.4 < res.eta < 0.7
    assert res.q["q_in"] > 0
    assert res.points[2].state.T > res.points[1].state.T  # T3 > T2


def test_cycle_plot():
    import matplotlib
    matplotlib.use("Agg")
    res = C.brayton(pressure_ratio=10, T3=1300)
    ax = res.plot(diagram="ts")
    assert sum(len(ln.get_data()[0]) for ln in ax.lines) > 0
    import matplotlib.pyplot as plt
    plt.close(ax.figure)


def test_nonideal_brayton_lower_eta():
    ideal = C.brayton(pressure_ratio=12, T3=1400, eta_compressor=1.0, eta_turbine=1.0)
    real = C.brayton(pressure_ratio=12, T3=1400, eta_compressor=0.85, eta_turbine=0.9)
    assert real.eta < ideal.eta