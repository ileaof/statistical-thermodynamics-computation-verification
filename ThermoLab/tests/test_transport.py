"""Transport correlation tests (gas-phase)."""

from __future__ import annotations

import math

import pytest

from thermolab import Gas


def test_air_viscosity(air):
    st = air.state(T=300.0, P=1e5)
    # air viscosity at 300 K ~ 1.85e-5 Pa.s
    assert st.mu == pytest.approx(1.85e-5, rel=0.25)
    assert st.mu > 0


def test_air_conductivity(air):
    st = air.state(T=300.0, P=1e5)
    # air k at 300 K ~ 0.026 W/(m.K)
    assert st.k == pytest.approx(0.026, rel=0.3)
    assert st.k > 0


def test_prandtl_air(air):
    st = air.state(T=300.0, P=1e5)
    # air Pr ~ 0.71
    assert st.prandtl == pytest.approx(0.71, rel=0.25)
    assert st.prandtl > 0


def test_thermal_diffusivity(air):
    st = air.state(T=350.0, P=1e5)
    alpha = st.k / (st.rho * st.cp)
    assert st.thermal_diffusivity == pytest.approx(alpha, rel=1e-6)


def test_viscosity_increases_with_T(air):
    mu_lo = air.state(T=300.0, P=1e5).mu
    mu_hi = air.state(T=800.0, P=1e5).mu
    assert mu_hi > mu_lo  # gas viscosity rises with T (Sutherland)


def test_transport_nan_in_two_phase(water):
    st = water.state(P=5e4, h=2e6)
    assert math.isnan(st.mu)
    assert math.isnan(st.k)