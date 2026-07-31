"""Tests for PartitionFunction, Thermodynamics and Mixture."""

from __future__ import annotations

import math

import numpy as np
import pytest

from statthermopy import State, Thermodynamics, get


# -- PartitionFunction -------------------------------------------------------

def test_partition_factors_product():
    n2 = get("N2")
    pv = Thermodynamics(n2, State(T=298.15, P=101325.0)).partition.evaluate(
        State(T=298.15, P=101325.0)
    )
    assert math.isclose(math.log(pv.Qt) + math.log(pv.Qr) + math.log(pv.Qv) + math.log(pv.Qe),
                        math.log(pv.Qtotal), rel_tol=1e-9)
    assert math.isclose(pv.ln_Qtotal, math.log(pv.Qtotal), rel_tol=1e-9)


def test_partition_contributions_keys():
    n2 = get("N2")
    pf = Thermodynamics(n2, State(T=500.0, P=1e5)).partition
    contribs = pf.contributions(State(T=500.0, P=1e5))
    assert set(contribs.keys()) == {"translational", "rotational", "vibrational", "electronic"}


# -- Thermodynamics: literature benchmarks -----------------------------------

def test_n2_298K_properties():
    n2 = get("N2")
    th = Thermodynamics(n2, State(T=298.15, P=101325.0)).compute()
    # Cp ~ 29.10, Cv ~ 20.79, gamma ~ 1.40, S ~ 191.5 J/mol/K
    assert math.isclose(th.Cp_m, 29.12, abs_tol=0.05)
    assert math.isclose(th.Cv_m, 20.79, abs_tol=0.05)
    assert math.isclose(th.gamma, 1.3998, abs_tol=0.005)
    assert math.isclose(th.S_m, 191.5, abs_tol=0.2)


def test_h2o_300K_cp():
    h2o = get("H2O")
    th = Thermodynamics(h2o, State(T=300.0, P=1e5)).compute()
    # nonlinear: 3 R trans + 1.5 R rot + small vib -> Cp ~ 33.6
    assert math.isclose(th.Cp_m, 33.6, abs_tol=0.3)


def test_co2_298K_cp():
    co2 = get("CO2")
    th = Thermodynamics(co2, State(T=298.15, P=1e5)).compute()
    # linear: 2.5 R + small vib -> Cp ~ 37.1 J/mol/K
    assert math.isclose(th.Cp_m, 37.2, abs_tol=0.5)


def test_ar_monoatomic():
    ar = get("Ar")
    th = Thermodynamics(ar, State(T=298.15, P=1e5)).compute()
    # monoatomic ideal gas: Cv = 1.5 R, Cp = 2.5 R, gamma = 5/3
    assert math.isclose(th.Cv_m, 1.5 * 8.314462618, abs_tol=0.01)
    assert math.isclose(th.Cp_m, 2.5 * 8.314462618, abs_tol=0.01)
    assert math.isclose(th.gamma, 5.0 / 3.0, abs_tol=0.005)


def test_ch4_800K_cp():
    ch4 = get("CH4")
    th = Thermodynamics(ch4, State(T=800.0, P=5e5)).compute()
    # NIST Cp(CH4,800K) ~ 63.5 J/mol/K
    assert math.isclose(th.Cp_m, 63.5, abs_tol=2.0)


def test_o2_electronic_ground_degeneracy():
    o2 = get("O2")
    th = Thermodynamics(o2, State(T=298.15, P=1e5)).compute()
    # ground triplet g=3 contributes R ln3 to entropy; check it's included.
    pf = th.contributions["electronic"]
    assert math.isclose(pf["ln_q"], math.log(3.0), abs_tol=1e-6)


def test_molar_and_massic_consistency():
    n2 = get("N2")
    th = Thermodynamics(n2, State(T=500.0, P=1e5)).compute()
    M = n2.molar_mass
    assert math.isclose(th.U_s, th.U_m / M, rel_tol=1e-12)
    assert math.isclose(th.Cp_s, th.Cp_m / M, rel_tol=1e-12)
    assert math.isclose(th.R_specific, 8.314462618 / M, rel_tol=1e-9)


def test_extensive_scales_with_n():
    n2 = get("N2")
    th1 = Thermodynamics(n2, State(T=500.0, P=1e5, n=1.0)).compute()
    th2 = Thermodynamics(n2, State(T=500.0, P=1e5, n=3.0)).compute()
    assert math.isclose(th2.U, 3 * th1.U, rel_tol=1e-9)
    assert math.isclose(th2.S, 3 * th1.S, rel_tol=1e-9)


def test_H_and_G_relations():
    n2 = get("N2")
    th = Thermodynamics(n2, State(T=500.0, P=1e5)).compute()
    assert math.isclose(th.H_m, th.U_m + 8.314462618 * 500.0, rel_tol=1e-9)
    assert math.isclose(th.G_m, th.A_m + 8.314462618 * 500.0, rel_tol=1e-9)
    assert math.isclose(th.mu_m, th.G_m, rel_tol=1e-12)
    assert math.isclose(th.Cp_m, th.Cv_m + 8.314462618, rel_tol=1e-9)


def test_cv_via_finite_difference_matches_analytic():
    # dU/dT|V should equal Cv_m (sum of mode Cv).
    n2 = get("N2")
    T = 600.0
    dT = 1e-2
    u0 = Thermodynamics(n2, State(T=T - dT, P=1e5, V=None, n=1.0)).compute().U_m
    u1 = Thermodynamics(n2, State(T=T + dT, P=1e5, n=1.0)).compute().U_m
    cv_num = (u1 - u0) / (2 * dT)
    cv = Thermodynamics(n2, State(T=T, P=1e5)).compute().Cv_m
    assert math.isclose(cv, cv_num, rel_tol=1e-3)


def test_entropy_pressure_dependence():
    # S should decrease with increasing P (S = ... + ln(kT/P)).
    n2 = get("N2")
    s_low = Thermodynamics(n2, State(T=300.0, P=1e4)).compute().S_m
    s_high = Thermodynamics(n2, State(T=300.0, P=1e6)).compute().S_m
    assert s_low > s_high
    assert math.isclose(s_low - s_high, 8.314462618 * math.log(100.0), rel_tol=1e-6)


def test_property_vs_T():
    n2 = get("N2")
    th = Thermodynamics(n2, State(T=300.0, P=1e5))
    Ts, cps = th.property_vs_T("Cp_m", np.linspace(300, 1000, 8), P=1e5)
    assert len(Ts) == len(cps) == 8
    # Cp should increase with T (vibrational activation)
    assert cps[-1] > cps[0]


def test_quantum_rotation_option():
    n2 = get("N2")
    th_c = Thermodynamics(n2, State(T=1000.0, P=1e5), use_quantum_rotation=False).compute()
    th_q = Thermodynamics(n2, State(T=1000.0, P=1e5), use_quantum_rotation=True).compute()
    # high T: quantum ~ classical
    assert math.isclose(th_c.Cv_m, th_q.Cv_m, abs_tol=0.2)


# -- Mixture -----------------------------------------------------------------

def test_mixture_mole_fractions_normalized():
    from statthermopy import IdealGasMixture
    mix = IdealGasMixture.from_names({"Ar": 0.7, "N2": 0.3})
    assert math.isclose(sum(mix.x.values()), 1.0, rel_tol=1e-12)


def test_mixture_mass_to_mole_conversion():
    from statthermopy import IdealGasMixture
    mix = IdealGasMixture.from_names({"Ar": 0.5, "H2": 0.5}, basis="mass")
    # H2 lighter -> higher mole fraction
    assert mix.x[get("H2")] > mix.x[get("Ar")]


def test_mixture_average_molar_mass():
    from statthermopy import IdealGasMixture
    mix = IdealGasMixture.from_names({"Ar": 0.7, "N2": 0.3})
    expected = 0.7 * get("Ar").molar_mass + 0.3 * get("N2").molar_mass
    assert math.isclose(mix.M_avg, expected, rel_tol=1e-12)


def test_mixture_cp_weighted():
    from statthermopy import IdealGasMixture
    mix = IdealGasMixture.from_names({"Ar": 0.5, "N2": 0.5})
    res = mix.compute(State(T=298.15, P=1e5))
    cp_ar = Thermodynamics(get("Ar"), State(T=298.15, P=1e5)).compute().Cp_m
    cp_n2 = Thermodynamics(get("N2"), State(T=298.15, P=1e5)).compute().Cp_m
    assert math.isclose(res.Cp_m, 0.5 * cp_ar + 0.5 * cp_n2, rel_tol=1e-9)


def test_mixture_entropy_includes_mixing():
    from statthermopy import IdealGasMixture
    mix = IdealGasMixture.from_names({"Ar": 0.5, "N2": 0.5})
    res = mix.compute(State(T=298.15, P=1e5))
    s_ar = Thermodynamics(get("Ar"), State(T=298.15, P=1e5)).compute().S_m
    s_n2 = Thermodynamics(get("N2"), State(T=298.15, P=1e5)).compute().S_m
    unmixed = 0.5 * s_ar + 0.5 * s_n2
    dsmix = -8.314462618 * (0.5 * math.log(0.5) + 0.5 * math.log(0.5))
    assert math.isclose(res.S_m, unmixed + dsmix, abs_tol=1e-6)


def test_mixture_validation_errors():
    from statthermopy import IdealGasMixture
    with pytest.raises(ValueError):
        IdealGasMixture.from_names({}, basis="mole")
    with pytest.raises(ValueError):
        IdealGasMixture.from_names({"N2": 1.0}, basis="invalid")


def test_mixture_as_dict():
    from statthermopy import IdealGasMixture
    mix = IdealGasMixture.from_names({"Ar": 0.7, "N2": 0.3})
    res = mix.compute(State(T=300.0, P=1e5))
    d = res.as_dict()
    assert "Cp_m" in d and "x" in d