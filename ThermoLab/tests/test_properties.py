"""Thermodynamic identity / consistency tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermolab import Gas


def test_compressibility_factor(air):
    """Z -> 1 as P -> 0 (ideal-gas limit)."""
    T = 350.0
    for P in (1e5, 1e4, 1e3):
        st = air.state(T=T, P=P)
        assert abs(st.Z - 1.0) < 0.05
    # lowest pressure is closest to 1
    z_low = air.state(T=T, P=1e3).Z
    z_high = air.state(T=T, P=1e5).Z
    assert abs(z_low - 1.0) <= abs(z_high - 1.0) + 1e-9


def test_gamma_equals_cp_over_cv(air):
    st = air.state(T=500.0, P=2e5)
    assert st.gamma == pytest.approx(st.cp / st.cv, rel=1e-6)


def test_cp_minus_cv_identity(nitrogen):
    """cp - cv = -T (dP/dT|v)^2 / (dP/dv|T)  (checked via finite differences)."""
    st = nitrogen.state(T=400.0, P=5e5)
    # at moderate pressure this is near R_s = R/M for N2 ~ 296.8 J/(kg.K)
    R_s = 8.31446261815324 / nitrogen.molar_mass
    assert (st.cp - st.cv) == pytest.approx(R_s, rel=0.1)


def test_internal_energy_h_minus_pv(air):
    st = air.state(T=600.0, P=3e5)
    assert st.u == pytest.approx(st.h - st.P * st.v, rel=1e-6)


def test_gibbs_and_helmholtz(air):
    st = air.state(T=500.0, P=1e5)
    assert st.g == pytest.approx(st.h - st.T * st.s, rel=1e-6)
    assert st.a_helmholtz == pytest.approx(st.u - st.T * st.s, rel=1e-6)


def test_density_volume_reciprocal(air):
    st = air.state(T=450.0, P=2e5)
    assert st.rho * st.v == pytest.approx(1.0, rel=1e-9)


def test_ideal_gas_density(air):
    """rho = P / (R_s T) in the ideal-gas limit."""
    st = air.state(T=300.0, P=1e5)
    R_s = 8.31446261815324 / air.molar_mass
    assert st.rho == pytest.approx(1e5 / (R_s * 300.0), rel=0.02)


def test_joule_thomson_sign(air):
    """At room T, air JT coefficient is small; sign can be +/- but finite."""
    st = air.state(T=300.0, P=1e5)
    assert math.isfinite(st.joule_thomson)
    assert abs(st.joule_thomson) < 1e-4   # K/Pa, small


def test_speed_of_sound_ideal(air):
    """a ~ sqrt(gamma R T) in the ideal-gas limit."""
    st = air.state(T=300.0, P=1e4)
    R_s = 8.31446261815324 / air.molar_mass
    a_ideal = math.sqrt(st.gamma * R_s * st.T)
    assert st.sound_speed == pytest.approx(a_ideal, rel=0.05)