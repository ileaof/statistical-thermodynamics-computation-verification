"""State-level tests: the spec's user example and basic sanity."""

from __future__ import annotations

import math

import pytest

from thermolab import Gas
from thermolab.exceptions import TwoPhaseError


def test_spec_example(air):
    """The exact example from the project prompt."""
    st = air.state(T=800.0, P=5e5)
    # air at 800 K, 5 bar: ideal-gas rho = P/(R*T) ~ 2.17 kg/m^3
    assert st.rho == pytest.approx(2.17, rel=0.05)
    assert st.cp > 1000           # J/(kg.K)
    assert 1.2 < st.gamma < 1.5
    assert st.sound_speed > 500   # m/s
    # all requested attributes exist and are finite
    for a in ("rho", "cp", "cv", "h", "s", "mu", "k", "gamma", "sound_speed"):
        v = getattr(st, a)
        assert math.isfinite(v), a


def test_property_caching(air):
    st = air.state(T=500.0, P=1e5)
    cp1 = st.cp
    assert st.cp is cp1  # cached_property returns same object


def test_two_phase_raises_on_cp(water):
    st = water.state(P=5e4, h=2e6)
    assert st.two_phase
    assert st.quality is not None
    with pytest.raises(TwoPhaseError):
        _ = st.cp


def test_state_repr(air):
    st = air.state(T=400.0, P=1e5)
    txt = repr(st)
    assert "ThermoLab State" in txt
    assert "Density" in txt


def test_to_dict_and_series(air):
    st = air.state(T=450.0, P=2e5)
    d = st.to_dict()
    assert {"T", "P", "rho", "cp", "h", "s", "mu", "k"}.issubset(d.keys())
    s = st.to_series()
    assert pytest.approx(s["T"], rel=1e-9) == 450.0


def test_liquid_state(water):
    st = water.state(P=1e5, h=1e5)   # compressed liquid
    assert st.phase == "liquid"
    assert st.rho > 900              # water-like density
    assert st.cp == pytest.approx(4180, rel=0.1)