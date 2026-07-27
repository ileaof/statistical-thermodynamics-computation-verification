"""Tables and interpolation tests."""

from __future__ import annotations

import numpy as np
import pytest

from thermolab.tables import PropertyTable, SaturationTable


def test_property_table_shape(air):
    pt = PropertyTable(air, (300, 500, 5), (1e5, 5e5, 4))
    assert pt.df.shape[0] == 5 * 4
    assert {"T", "P", "rho", "cp", "h"}.issubset(pt.df.columns)


def test_property_table_interpolation(air):
    pt = PropertyTable(air, (300, 500, 9), (1e5, 5e5, 9))
    f = pt.interpolate()
    # exact at a grid node
    r = f(400.0, 3e5)
    st = air.state(T=400.0, P=3e5)
    assert r["rho"] == pytest.approx(st.rho, rel=1e-2)
    assert r["cp"] == pytest.approx(st.cp, rel=1e-2)


def test_saturation_table(water):
    stb = SaturationTable(water, (300, 370, 8))
    assert "h_g" in stb.df.columns
    # saturated vapor enthalpy at ~100 C should be ~ 2676 kJ/kg (row near 370 K)
    last = stb.df.dropna().iloc[-1]
    assert last["h_g"] == pytest.approx(2.65e6, rel=0.05)


def test_cfd_bulk(air):
    from thermolab.cfd import bulk_properties
    states = [air.state(T=T, P=1e5) for T in (300, 400, 500)]
    df = bulk_properties(states)
    assert len(df) == 3
    assert {"rho", "cp", "gamma", "mu", "k"}.issubset(df.columns)