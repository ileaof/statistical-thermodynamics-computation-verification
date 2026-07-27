"""Flash solver tests: every variable pair round-trips; two-phase handling."""

from __future__ import annotations

import pytest

from thermolab import Gas
from thermolab.exceptions import FlashSpecificationError


PAIRS_AIR = [
    dict(T=500, P=2e5),
    dict(T=700, rho=3.5),
    dict(T=650, v=0.3),
    dict(P=1e5, rho=1.2),
    dict(P=1e5, h=1e6),
    dict(P=5e5, s=700),
    dict(rho=1.5, h=5e5),
    dict(T=600, s=650),
]


@pytest.mark.parametrize("specs", PAIRS_AIR)
def test_air_pair_roundtrip(air, specs):
    st = air.state(**specs)
    # The defining variables should be recovered.
    for k, v in specs.items():
        got = getattr(st, k)
        assert got == pytest.approx(v, rel=1e-3 or 1e-6), f"{k}: {got} vs {v}"


def test_invalid_spec_count(air):
    with pytest.raises(FlashSpecificationError):
        air.state(T=500)
    with pytest.raises(FlashSpecificationError):
        air.state(T=500, P=1e5, h=1e5)


def test_two_phase_quality(water):
    st = water.state(P=5e4, h=2e6)
    assert st.two_phase
    assert 0.0 < st.quality < 1.0
    assert st.h == pytest.approx(2e6, rel=1e-6)
    # T equals the saturation temperature at this pressure
    Tsat = water.saturation_temperature(5e4)
    assert st.T == pytest.approx(Tsat, rel=1e-4)


def test_superheated_steam(water):
    st = water.state(P=5e4, h=3e6)
    assert st.phase == "vapor"
    assert not st.two_phase
    assert st.h == pytest.approx(3e6, rel=1e-4)


def test_compressed_liquid(water):
    st = water.state(P=5e4, h=1e5)
    assert st.phase == "liquid"
    assert st.rho > 900


def test_phase_hint(air):
    st = air.state(T=300, P=1e5, phase="vapor")
    assert st.phase == "vapor"


def test_quality_zero_and_one(water):
    Tsat = water.saturation_temperature(5e4)
    sat = water.backend.saturation_state(Tsat, water.fractions)
    M = water.molar_mass
    hf, hg = sat.h_f / M, sat.h_g / M
    s0 = water.state(P=5e4, h=hf)
    s1 = water.state(P=5e4, h=hg)
    assert s0.quality == pytest.approx(0.0, abs=1e-6)
    assert s1.quality == pytest.approx(1.0, abs=1e-6)