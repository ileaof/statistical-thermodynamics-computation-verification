"""Tests for the predefined-fluids module (atmospheric air) and the mixture component breakdown."""

from __future__ import annotations

import math

import pytest

from statthermopy import (
    STANDARD_DRY_AIR,
    PredefinedFluid,
    air,
    available_fluids,
    get_fluid,
    register_fluid,
)
from statthermopy.constants import R
from statthermopy.core.state import State


def _x_by_name(mix) -> dict:
    return {m.name: x for m, x in mix.x.items()}


# -- composition -------------------------------------------------------------

def test_air_is_registered_and_case_insensitive():
    assert "Air" in available_fluids()
    assert get_fluid("air").name == "Air"
    assert get_fluid("AIR").humidifiable is True


def test_unknown_fluid_raises():
    with pytest.raises(KeyError):
        get_fluid("unobtainium")


def test_dry_air_composition_and_molar_mass():
    a = air()
    x = _x_by_name(a)
    assert set(x) == {"N2", "O2", "Ar", "CO2"}
    # normalised standard fractions
    assert x["N2"] == pytest.approx(0.78084 / sum(STANDARD_DRY_AIR.values()), abs=1e-6)
    # canonical dry-air molar mass ~28.96 g/mol and R ~287 J/kg/K
    assert a.M_avg * 1e3 == pytest.approx(28.96, abs=0.05)
    assert R / a.M_avg == pytest.approx(287.0, abs=0.5)


def test_humid_air_adds_and_scales():
    w = 0.02
    a = air(water_mole_fraction=w)
    x = _x_by_name(a)
    assert x["H2O"] == pytest.approx(w, abs=1e-9)
    # dry constituents scaled by (1 - w); ratios among them preserved
    dry = air()
    xd = _x_by_name(dry)
    assert x["N2"] == pytest.approx(xd["N2"] * (1 - w), rel=1e-9)
    # humid air is lighter (water is lighter than air) and so has a larger specific R
    assert a.M_avg < dry.M_avg
    assert R / a.M_avg > R / dry.M_avg


def test_water_fraction_bounds():
    with pytest.raises(ValueError):
        air(water_mole_fraction=1.0)
    with pytest.raises(ValueError):
        air(water_mole_fraction=-0.1)


def test_predefined_fluid_registry_extensible():
    """A custom fluid can be registered and built — the registry is open for extension."""
    register_fluid(PredefinedFluid("TestGasMix", "50/50 N2/O2", {"N2": 0.5, "O2": 0.5}))
    try:
        assert "TestGasMix" in available_fluids()
        mix = get_fluid("testgasmix").build()
        assert _x_by_name(mix) == pytest.approx({"N2": 0.5, "O2": 0.5}, abs=1e-9)
        # a non-humidifiable fluid rejects a water fraction
        with pytest.raises(ValueError):
            get_fluid("testgasmix").build(water_mole_fraction=0.01)
    finally:
        from statthermopy import fluids as _f
        _f._FLUIDS.pop("testgasmix", None)


# -- per-component breakdown + mixing entropy --------------------------------

def test_component_contributions_sum_to_totals():
    res = air().compute(State(T=298.15, P=101325.0))
    assert set(res.components) == {"N2", "O2", "Ar", "CO2"}
    for prop in ("U", "H", "S", "A", "G", "Cv", "Cp"):
        summed = sum(getattr(c, f"{prop}_contrib") for c in res.components.values())
        assert summed == pytest.approx(getattr(res, f"{prop}_m"), rel=1e-9, abs=1e-6)
    # mole fractions sum to 1
    assert sum(c.x for c in res.components.values()) == pytest.approx(1.0, abs=1e-9)


def test_entropy_of_mixing():
    a = air()
    res = a.compute(State(T=300.0, P=101325.0))
    xs = [x for _, x in a.x.items()]
    expected = -R * sum(x * math.log(x) for x in xs)
    assert res.S_mixing == pytest.approx(expected, rel=1e-12)
    assert res.S_mixing > 0.0  # mixing distinct species raises entropy


def test_component_contribution_reports_pure_values():
    """Each component carries its own pure-species molar properties plus its weighted share."""
    res = air().compute(State(T=500.0, P=101325.0))
    n2 = res.components["N2"]
    assert n2.molar_mass == pytest.approx(0.0280134, abs=1e-6)
    assert n2.U_contrib == pytest.approx(n2.x * n2.U_m, rel=1e-12)
    assert n2.Cp_contrib == pytest.approx(n2.x * n2.Cp_m, rel=1e-12)


def test_mixture_as_dict_includes_components_and_is_serialisable():
    import json

    res = air(water_mole_fraction=0.01).compute(State(T=298.15, P=101325.0))
    d = res.as_dict()
    assert "S_mixing" in d and "components" in d
    assert "H2O" in d["components"]
    # nested ComponentContribution dataclasses become plain dicts -> JSON round-trips
    assert json.loads(json.dumps(d))["components"]["N2"]["x"] > 0.0
