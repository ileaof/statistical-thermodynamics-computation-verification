"""Validation against reference / tabulated values and backend consistency."""

from __future__ import annotations

import pytest

from thermolab import Gas, list_fluids


def test_supported_fluid_list():
    fluids = list_fluids()
    assert "H2O" in fluids
    assert "AIR" in fluids or "Air" in fluids
    assert "R134a" in fluids or "R134A" in fluids


def test_unsupported_fluid_raises():
    from thermolab.exceptions import UnsupportedFluidError
    with pytest.raises(UnsupportedFluidError):
        Gas("Methane")  # not in this ThermoPack build


def test_n2_critical_point(nitrogen):
    assert nitrogen.critical_temperature() == pytest.approx(126.2, rel=2e-3)
    assert nitrogen.critical_pressure() == pytest.approx(3.3958e6, rel=5e-3)


def test_water_saturation_100c(water):
    Psat = water.saturation_pressure(373.15)
    assert Psat == pytest.approx(101325.0, rel=1e-3)


def test_water_sat_vapor_enthalpy(water):
    """Saturated vapor enthalpy of water at 100 C ~ 2676 kJ/kg."""
    Tsat = water.saturation_temperature(101325.0)
    sat = water.backend.saturation_state(Tsat, water.fractions)
    hg = sat.h_g / water.molar_mass
    assert hg == pytest.approx(2.676e6, rel=2e-2)


def test_air_cp_reference(air):
    """Air cp at 300 K ~ 1005 J/(kg.K)."""
    st = air.state(T=300.0, P=1e5)
    assert st.cp == pytest.approx(1005.0, rel=2e-2)


def test_air_gamma_reference(air):
    """Air gamma at 300 K ~ 1.4."""
    st = air.state(T=300.0, P=1e5)
    assert st.gamma == pytest.approx(1.4, rel=2e-2)


def test_available_backends():
    from thermolab import available_backends
    assert "thermopack" in available_backends()


def test_property_bundle_complete(air):
    st = air.state(T=500.0, P=1e5)
    b = st.bundle()
    for attr in ("T", "P", "rho", "v", "u", "h", "s", "g", "a_helmholtz",
                 "cp", "cv", "gamma", "Z", "sound_speed", "mu", "k",
                 "thermal_diffusivity", "prandtl", "joule_thomson",
                 "beta_thermal_expansion", "kappa_t"):
        assert hasattr(b, attr), attr