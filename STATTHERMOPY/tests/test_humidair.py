"""Tests for the Statistical Humid Air module.

Physics is validated against reference water/psychrometric data; the psychrometric relations and
the statistical-mechanics vapour/liquid coupling are checked for internal consistency.
"""

from __future__ import annotations

import pytest

from statthermopy.constants import R
from statthermopy.humidair import (
    ConstantCpLiquid,
    HumidAir,
    IAPWSLiquid,
    SaturationCalculator,
)
from statthermopy.humidair.liquid import P_TRIPLE, T_TRIPLE

# -- saturation pressure (statistical vapour, no vapour-pressure correlation) -

#: reference water saturation pressures (Pa)
_PSAT_REF = {273.16: 611.657, 293.15: 2339.0, 298.15: 3169.9, 323.15: 12344.0, 373.15: 101325.0}


@pytest.mark.parametrize("liquid", [ConstantCpLiquid(), IAPWSLiquid()])
def test_saturation_pressure_matches_reference(liquid):
    sc = SaturationCalculator(liquid=liquid)
    for T, ref in _PSAT_REF.items():
        ps = sc.saturation_pressure(T)
        # anchored exactly at the triple point; within ~2 % up to the boiling point
        tol = 1e-3 if T == T_TRIPLE else 2.0
        assert abs(ps - ref) / ref * 100.0 < tol


def test_triple_point_is_exact_anchor():
    sc = SaturationCalculator(liquid=ConstantCpLiquid())
    assert sc.saturation_pressure(T_TRIPLE) == pytest.approx(P_TRIPLE, rel=1e-6)


def test_dew_point_inverts_saturation_pressure():
    sc = SaturationCalculator(liquid=ConstantCpLiquid())
    for T in (283.15, 300.0, 340.0):
        P = sc.saturation_pressure(T)
        assert sc.dew_point(P) == pytest.approx(T, abs=1e-3)


def test_enthalpy_of_vaporisation_reasonable():
    sc = SaturationCalculator(liquid=ConstantCpLiquid())
    # literature Δh_vap(298 K) ≈ 43.99 kJ/mol
    assert sc.enthalpy_of_vaporisation(298.15) / 1e3 == pytest.approx(44.0, abs=0.3)


def test_vapour_entropy_is_absolute_and_correct():
    """The statistical vapour reproduces the standard molar entropy of steam (~188.8 J/mol/K)."""
    sc = SaturationCalculator(liquid=ConstantCpLiquid())
    s = sc.vapour_properties(298.15, 1e5).S_m
    assert s == pytest.approx(188.8, abs=0.5)


# -- psychrometric relations -------------------------------------------------

def test_saturated_state_headline_numbers():
    """Maximum solubility at 25 C / 1 atm matches textbook psychrometrics."""
    st = HumidAir().state(298.15, 101325.0)  # default = saturated
    assert st.saturated
    assert st.relative_humidity == pytest.approx(1.0, abs=1e-9)
    assert st.dew_point == pytest.approx(298.15, abs=1e-3)      # dew point = dry bulb
    assert st.wet_bulb == pytest.approx(298.15, abs=0.05)       # wet bulb = dry bulb
    assert st.humidity_ratio_max * 1e3 == pytest.approx(20.0, abs=0.4)   # ~20 g/kg
    assert st.absolute_humidity_max * 1e3 == pytest.approx(23.0, abs=0.5)  # ~23 g/m^3


def test_max_mole_fraction_is_psat_over_P():
    ha = HumidAir()
    T, P = 310.0, 90000.0
    assert ha.max_mole_fraction(T, P) == pytest.approx(ha.saturation_pressure(T) / P, rel=1e-12)


def test_humidity_ratio_and_absolute_humidity_definitions():
    ha = HumidAir()
    st = ha.state(298.15, 101325.0, relative_humidity=0.5, wet_bulb=False)
    eps = ha.epsilon
    P_v = st.P_vapor
    assert st.humidity_ratio == pytest.approx(eps * P_v / (st.P - P_v), rel=1e-12)
    assert st.absolute_humidity == pytest.approx(P_v * ha.M_water / (R * st.T), rel=1e-12)
    assert st.vapor_concentration == pytest.approx(P_v / (R * st.T), rel=1e-12)
    # relative humidity is P_v / P_sat
    assert st.relative_humidity == pytest.approx(0.5, rel=1e-9)


def test_humidity_ratio_input_round_trips_to_relative_humidity():
    ha = HumidAir()
    ref = ha.state(298.15, 101325.0, relative_humidity=0.7, wet_bulb=False)
    got = ha.state(298.15, 101325.0, humidity_ratio=ref.humidity_ratio, wet_bulb=False)
    assert got.relative_humidity == pytest.approx(0.7, rel=1e-9)
    assert got.x_h2o == pytest.approx(ref.x_h2o, rel=1e-9)


def test_degree_of_saturation_below_relative_humidity():
    st = HumidAir().state(298.15, 101325.0, relative_humidity=0.5, wet_bulb=False)
    # μ = w/w_s = RH·(P - P_sat)/(P - P_v) < RH for unsaturated air
    assert st.degree_of_saturation < st.relative_humidity


def test_wet_bulb_between_dew_and_dry_bulb():
    st = HumidAir().state(298.15, 101325.0, relative_humidity=0.5)
    assert st.dew_point < st.wet_bulb < st.T
    # 25 C / 50 % RH -> wet-bulb ~ 17.9 C, dew point ~ 13.9 C
    assert st.wet_bulb - 273.15 == pytest.approx(17.9, abs=0.4)
    assert st.dew_point - 273.15 == pytest.approx(13.9, abs=0.4)


def test_reject_multiple_humidity_specs():
    with pytest.raises(ValueError):
        HumidAir().state(298.15, 101325.0, relative_humidity=0.5, humidity_ratio=0.01)


# -- partition-function breakdown & mixture consistency ----------------------

def test_vapor_mode_contributions_sum_to_gibbs():
    st = HumidAir().state(298.15, 101325.0, relative_humidity=0.5, wet_bulb=False)
    g_from_modes = sum(c["G_m"] for c in st.vapor_mode_contributions.values())
    from statthermopy import Thermodynamics, get
    from statthermopy.core.state import State
    g_vapor = Thermodynamics(get("H2O"), State(T=298.15, P=st.P_vapor)).compute().G_m
    assert g_from_modes == pytest.approx(g_vapor, rel=1e-9)
    # translational + rotational dominate; vibration nearly frozen; electronic inert
    modes = st.vapor_mode_contributions
    assert modes["electronic"]["Cv_m"] == pytest.approx(0.0, abs=1e-9)
    assert abs(modes["vibrational"]["S_m"]) < abs(modes["rotational"]["S_m"])


def test_mixture_bulk_properties():
    st = HumidAir().state(298.15, 101325.0, relative_humidity=0.4, wet_bulb=False)
    assert st.R_specific == pytest.approx(R / st.M_avg, rel=1e-12)
    assert st.density == pytest.approx(st.P * st.M_avg / (R * st.T), rel=1e-12)
    # adding light water vapour lowers M_avg below dry air (28.96 g/mol)
    assert st.M_avg < 0.028966


def test_custom_dry_background_extensible():
    """The dry background is swappable (extensibility toward trace gases / other mixtures)."""
    from statthermopy import IdealGasMixture
    pure_n2 = IdealGasMixture.from_names({"N2": 1.0})
    ha = HumidAir(dry_air=pure_n2)
    st = ha.state(300.0, 101325.0, relative_humidity=0.5, wet_bulb=False)
    assert st.M_avg > 0.0 and st.P_sat > 0.0


def test_state_as_dict_is_serialisable():
    import json
    st = HumidAir().state(298.15, 101325.0, relative_humidity=0.5, wet_bulb=False)
    d = st.as_dict()
    assert d["P_sat"] > 0 and "vapor_mode_contributions" in d and "components" in d
    assert json.loads(json.dumps(d))["humidity_ratio_max"] > 0.0


# -- plots -------------------------------------------------------------------

def test_humidair_plots_return_axes():
    import numpy as np

    from statthermopy.humidair import plots as hp
    Ts = np.linspace(280.0, 360.0, 12)
    assert hp.plot_saturation_pressure_vs_T(Ts).has_data()
    assert hp.plot_max_solubility_vs_T(Ts).has_data()
    ax3 = hp.plot_solubility_surface(Ts[:6], np.linspace(80e3, 120e3, 6))
    assert ax3.get_zlabel() != ""


def test_humidity_ratio_plot_matches_definition_and_handles_boiling():
    """The humidity-ratio curve is computed directly from P_sat (fast path), equals
    ε·P_sat/(P−P_sat) as a dimensionless kg/kg dry air mass ratio, and leaves the above-boiling
    region (P_sat ≥ P) blank."""
    import numpy as np

    from statthermopy.humidair import HumidAir
    from statthermopy.humidair import plots as hp
    ha = HumidAir()
    P = 101325.0
    # range that crosses the boiling point at 1 atm
    Ts = np.linspace(283.15, 380.0, 15)
    ax = hp.plot_humidity_ratio_vs_T(Ts, P=P, model=ha)
    assert "humidity ratio" in ax.get_ylabel().lower()
    ys = ax.lines[0].get_ydata()
    for t, y in zip(Ts, ys, strict=False):
        ps = ha.saturation_pressure(float(t))
        if ps < P:
            # reported as the dimensionless mass ratio w_s = ε·P_sat/(P−P_sat) (kg/kg dry air)
            assert y == pytest.approx(ha.epsilon * ps / (P - ps), rel=1e-9)
        else:  # above boiling: no saturated humid air -> blank
            assert np.isnan(y)


def test_relative_humidity_plot_is_direct():
    """RH-vs-T at fixed humidity ratio equals P_v/P_sat(T), capped at 1."""
    import numpy as np

    from statthermopy.humidair import HumidAir
    from statthermopy.humidair import plots as hp
    ha = HumidAir()
    P, w = 101325.0, 0.01
    Ts = np.linspace(280.0, 340.0, 10)
    ax = hp.plot_relative_humidity_vs_T(Ts, w, P=P, model=ha)
    ys = ax.lines[0].get_ydata()
    r = w / ha.epsilon
    P_v = r * P / (1.0 + r)
    for t, y in zip(Ts, ys, strict=False):
        assert y == pytest.approx(min(P_v / ha.saturation_pressure(float(t)), 1.0), rel=1e-9)


def test_default_liquid_model_uses_iapws_when_available():
    # iapws is installed in this environment -> the default calculator picks it up
    assert SaturationCalculator().liquid.name == "iapws95"


# -- comparative analysis (water-vapour content + dry/humid isobaric/isochoric) ----

def test_water_vapor_content_actual_capped_by_saturation():
    """The actual content follows saturation below the dew point (condensation) and plateaus at
    the fixed value above it; both are ≤ their saturation values, and the dew point is where they
    meet."""
    import numpy as np

    from statthermopy.humidair import PsychrometricAnalysis
    an = PsychrometricAnalysis()
    Ts = np.linspace(273.16, 320.0, 24)
    tbl = an.water_vapor_content(Ts, relative_humidity=0.5)
    act = tbl.columns["actual w [g/kg]"]
    sat = tbl.columns["saturation w_sat [g/kg]"]
    w_fixed = tbl.meta["w_fixed_g_per_kg"]
    dew = tbl.meta["dew_point_K"]
    for t, a, s in zip(tbl.x_K, act, sat, strict=False):
        assert a <= s + 1e-9                       # actual never exceeds saturation
        if t <= dew:
            assert a == pytest.approx(s, rel=1e-6)  # below dew point: saturated (condensation)
        else:
            assert a == pytest.approx(w_fixed, rel=1e-6)  # above: fixed content


def test_property_comparison_four_curves_and_pressure_dependence():
    """Entropy gives four distinct curves (P-dependent: const-P ≠ const-V); Cp gives four curves
    that coincide by constraint (temperature-only)."""
    import numpy as np

    from statthermopy.humidair import PsychrometricAnalysis
    an = PsychrometricAnalysis()
    Ts = np.linspace(280.0, 340.0, 12)

    s = an.property_comparison("S_m", Ts, relative_humidity=0.5)
    assert set(s.columns) == {
        "Dry air — const P", "Humid air — const P", "Dry air — const V", "Humid air — const V",
    }
    assert not np.allclose(s.columns["Dry air — const P"], s.columns["Dry air — const V"])
    # humid air has higher entropy than dry air (extra component + mixing) at const P
    assert np.all(np.array(s.columns["Humid air — const P"])
                  > np.array(s.columns["Dry air — const P"]))

    cp = an.property_comparison("Cp_m", Ts, relative_humidity=0.5)
    assert cp.meta["pressure_independent"]
    assert np.allclose(cp.columns["Dry air — const P"], cp.columns["Dry air — const V"])


def test_thermal_fields_computed_independently_per_mixture():
    """T_v = U_m/Cv_m and T_p = H_m/Cp_m must come from each mixture's OWN properties — dry-air
    values are never reused for humid air (or vice versa) — so the fields genuinely differ."""
    import numpy as np

    from statthermopy.core.state import State
    from statthermopy.humidair import HumidAir, PsychrometricAnalysis
    ha = HumidAir()
    an = PsychrometricAnalysis(ha)
    Ts = np.linspace(290.0, 350.0, 12)
    P = 101325.0
    tbl = an.thermal_fields_comparison(Ts, P, relative_humidity=0.5)
    assert set(tbl.columns) == {
        "Dry air T_v (const V)", "Humid air T_v (const V)",
        "Dry air T_p (const P)", "Humid air T_p (const P)",
    }
    # dry and humid differ at every temperature (distinct thermodynamic properties)
    dtv = np.array(tbl.columns["Dry air T_v (const V)"])
    htv = np.array(tbl.columns["Humid air T_v (const V)"])
    dtp = np.array(tbl.columns["Dry air T_p (const P)"])
    htp = np.array(tbl.columns["Humid air T_p (const P)"])
    assert np.all(dtv != htv) and np.all(dtp != htp)
    assert tbl.meta["max_diff_Tv"] > 0.0 and tbl.meta["max_diff_Tp"] > 0.0

    # each field equals its own mixture's U_m/Cv_m and H_m/Cp_m (no cross-use)
    dry = ha.dry_air
    humid, _, _ = an._humid_mixture_for(0.5 * (Ts[0] + Ts[-1]), P, 0.5, None, None)
    for k, t in enumerate(Ts):
        pd = dry.compute(State(T=float(t), P=P))
        ph = humid.compute(State(T=float(t), P=P))
        assert dtv[k] == pytest.approx(pd.U_m / pd.Cv_m, rel=1e-12)
        assert htv[k] == pytest.approx(ph.U_m / ph.Cv_m, rel=1e-12)
        assert dtp[k] == pytest.approx(pd.H_m / pd.Cp_m, rel=1e-12)
        assert htp[k] == pytest.approx(ph.H_m / ph.Cp_m, rel=1e-12)
        # explicit no-reuse check: humid field must NOT match a dry-based ratio
        assert htv[k] != pytest.approx(ph.U_m / pd.Cv_m, rel=1e-9)


def test_comparison_table_exports(tmp_path):
    from statthermopy.humidair import PsychrometricAnalysis
    tbl = PsychrometricAnalysis().water_vapor_content([280.0, 300.0, 320.0], relative_humidity=0.5)
    csv = tbl.to_csv(tmp_path / "wv.csv")
    xlsx = tbl.to_excel(tmp_path / "wv.xlsx")
    from pathlib import Path
    assert Path(csv).exists() and "saturation" in Path(csv).read_text(encoding="utf-8")
    assert Path(xlsx).stat().st_size > 0


def test_comparison_plots_return_table_and_axes():
    import numpy as np

    from statthermopy.humidair import HumidAir
    from statthermopy.humidair import plots as hp
    ha = HumidAir()
    Ts = np.linspace(280.0, 330.0, 15)
    tbl, ax = hp.plot_water_vapor_content_vs_T(ha, Ts, relative_humidity=0.5)
    assert ax.has_data() and "actual w [g/kg]" in tbl.columns
    tbl2, ax2 = hp.plot_property_comparison(ha, "S_m", Ts, relative_humidity=0.5)
    # four plotted curves + a legend
    assert len(list(ax2.get_lines())) == 4
    assert ax2.get_legend() is not None
    # temperature unit conversion to Celsius
    tbl3, _ = hp.plot_property_comparison(ha, "H_m", Ts, relative_humidity=0.5, temperature_unit="C")
    assert tbl3.x[0] == pytest.approx(Ts[0] - 273.15, rel=1e-9)
