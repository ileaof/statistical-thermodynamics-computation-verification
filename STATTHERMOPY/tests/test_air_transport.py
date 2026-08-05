"""Tests for the Air Transport Properties Database (``statthermopy.transport.air``).

Covers the mixing rules (Wilke viscosity, Mason–Saxena conductivity, Blanc diffusion), the
:class:`AirTransport` dry/humid facade, the per-species contribution breakdown, internal
consistency of the derived dimensionless groups, the dry-vs-humid vs-T comparison table, the
four export formats (CSV / Excel / JSON / PDF), and the extended per-species transport database
(critical properties + acentric factor stored as inputs, with ``Molecule`` still free of any
``critical`` attribute).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from statthermopy import State, get
from statthermopy.transport.air import (
    AIR_TRANSPORT_PROPS,
    AirTransport,
    AirTransportAnalysis,
    AirTransportExporter,
    MixtureTransportCalculator,
    blanc_diffusion,
    get_species_transport,
    list_species_transport,
    mason_saxena_conductivity,
    wilke_viscosity,
)

T0 = 300.0
P0 = 101325.0


# -- mixing-rule helpers --------------------------------------------------------


def test_wilke_viscosity_binary_n2_o2():
    """Wilke viscosity of an equimolar N2/O2 mixture is ~2e-5 Pa·s at 300 K and recovers the pure."""
    n2, o2 = get("N2"), get("O2")
    from statthermopy.transport import TransportCalculator

    mu_n2 = TransportCalculator(n2, State(T=T0, P=P0)).compute().mu
    mu_o2 = TransportCalculator(o2, State(T=T0, P=P0)).compute().mu
    # equimolar mixture
    mu_mix, contribs = wilke_viscosity([(0.5, mu_n2, n2.molar_mass), (0.5, mu_o2, o2.molar_mass)])
    # order of magnitude and within the pure-species bracket
    assert 1.5e-5 < mu_mix < 2.5e-5
    assert min(mu_n2, mu_o2) <= mu_mix <= max(mu_n2, mu_o2)
    # contributions reconstruct the total
    assert sum(contribs) == pytest.approx(mu_mix, rel=1e-12)


def test_wilke_viscosity_recovers_pure():
    """A single-species "mixture" gives back the pure viscosity."""
    n2 = get("N2")
    from statthermopy.transport import TransportCalculator

    mu = TransportCalculator(n2, State(T=T0, P=P0)).compute().mu
    mu_mix, contribs = wilke_viscosity([(1.0, mu, n2.molar_mass)])
    assert mu_mix == pytest.approx(mu, rel=1e-12)
    assert contribs == [pytest.approx(mu, rel=1e-12)]


def test_mason_saxena_conductivity_dry_air():
    """Mason–Saxena conductivity of dry air is ~0.026 W/m/K at 300 K (within ~5 %)."""
    air = AirTransport()
    res = air.dry(T0, P0)
    assert res.k == pytest.approx(0.026, rel=0.05)


def test_mason_saxena_recovers_pure():
    n2 = get("N2")
    from statthermopy.transport import TransportCalculator

    k = TransportCalculator(n2, State(T=T0, P=P0)).compute().k
    k_mix, _ = mason_saxena_conductivity([(1.0, k, n2.molar_mass)])
    assert k_mix == pytest.approx(k, rel=1e-12)


def test_blanc_diffusion_h2o_in_air():
    """D_H2O-air via Blanc ~2.6e-5 m^2/s at 300 K, 1 atm (loose tolerance — H2O polar-LJ caveat)."""
    air = AirTransport()
    res = air.dry(T0, P0)  # dry air: H2O is an external trace diffusing into the background
    assert res.D_eff == pytest.approx(2.6e-5, rel=0.20)


def test_blanc_diffusion_recovers_binary():
    """For a 2-species mixture, Blanc reduces to the binary D_tj / (1 - x_t) form."""
    # trace index 0, species 0 with x=0.3 diffusing into species 1 (x=0.7), D_01 = 2e-5
    D = 2.0e-5
    D_pairs = [[0.0, D], [D, 0.0]]
    x = [0.3, 0.7]
    d = blanc_diffusion(0, x, D_pairs)
    # (1 - 0.3) / (0.7 / D) = 0.3 / 0.7 * D ... = D * (1-0.3)/0.7
    assert d == pytest.approx((1.0 - 0.3) / (0.7 / D), rel=1e-12)


# -- dry-air headline values ----------------------------------------------------


def test_dry_air_headline_values():
    """Dry air at 300 K, 1 atm: mu ~1.85e-5, Pr ~0.71 (polar-H2O properties looser)."""
    res = AirTransport().dry(T0, P0)
    assert res.mu == pytest.approx(1.85e-5, rel=0.05)
    assert res.Pr == pytest.approx(0.71, rel=0.05)
    # Schmidt/Lewis involve the polar H2O LJ approximation -> wider tolerance
    assert 0.45 < res.Sc < 0.80
    assert 0.65 < res.Le < 1.10
    # composition
    assert set(res.x.keys()) == {"N2", "O2", "Ar", "CO2"}
    assert res.humidity_ratio is None
    assert pytest.approx(1.0) == res.Z


def test_dry_air_speed_of_sound():
    res = AirTransport().dry(T0, P0)
    assert res.a == pytest.approx(347.0, rel=0.02)  # ~347 m/s at 300 K


# -- humid-air update -----------------------------------------------------------


def test_humid_air_differs_from_dry():
    """Humid (saturated) air has a lower density than dry air at the same (T, P)."""
    air = AirTransport()
    dry = air.dry(T0, P0)
    hum = air.humid(T0, P0, saturated=True)
    assert hum.rho < dry.rho
    assert hum.humidity_ratio is not None and hum.humidity_ratio > 0
    # H2O now appears in the composition
    assert "H2O" in hum.x
    assert hum.M_avg < dry.M_avg  # lighter mixture


def test_humid_transport_monotonic_with_humidity():
    """As humidity rises (RH 0 -> 0.5 -> 1.0), density falls monotonically."""
    air = AirTransport()
    rhos = [air.humid(T0, P0, relative_humidity=rh).rho for rh in (1e-6, 0.5, 1.0)]
    assert rhos[0] > rhos[1] > rhos[2]


# -- per-species contributions --------------------------------------------------


def test_contributions_sum_to_mixture():
    """The Wilke/Mason–Saxena contributions sum to the mixture mu and k."""
    res = AirTransport().humid(T0, P0, saturated=True)
    mu_sum = sum(c.mu_contrib for c in res.components.values())
    k_sum = sum(c.k_contrib for c in res.components.values())
    assert mu_sum == pytest.approx(res.mu, rel=1e-9)
    assert k_sum == pytest.approx(res.k, rel=1e-9)
    # humid air has the 4 dry species + H2O
    assert set(res.components.keys()) == {"N2", "O2", "Ar", "CO2", "H2O"}


def test_species_contributions_table():
    ana = AirTransportAnalysis()
    tbl = ana.species_contributions(T0, P0, relative_humidity=0.5)
    assert tbl.x == ["N2", "O2", "Ar", "CO2", "H2O"]
    # columns present
    for key in (
        "x [-]",
        "M [g/mol]",
        "mu_i [Pa·s]",
        "k_i [W/m/K]",
        "D_im [m^2/s]",
        "mu_contrib [Pa·s]",
        "k_contrib [W/m/K]",
    ):
        assert key in tbl.columns
        assert len(tbl.columns[key]) == 5


# -- internal consistency -------------------------------------------------------


def test_dimensionless_group_consistency():
    """Pr = mu*cp_s/k, Sc = nu/D_eff, Le = Sc/Pr, a = sqrt(gamma*R_s*T)."""
    res = AirTransport().humid(T0, P0, relative_humidity=0.5)
    assert res.Pr == pytest.approx(res.mu * res.cp_s / res.k, rel=1e-6)
    assert res.Sc == pytest.approx(res.nu / res.D_eff, rel=1e-6)
    assert res.Le == pytest.approx(res.Sc / res.Pr, rel=1e-6)
    assert res.a == pytest.approx(math.sqrt(res.gamma * res.R_specific * res.T), rel=1e-6)
    # ideal-gas EOS
    assert res.rho == pytest.approx(res.P * res.M_avg / (8.314462618 * res.T), rel=1e-5)
    assert res.beta == pytest.approx(1.0 / res.T, rel=1e-12)
    assert res.kappa_T == pytest.approx(1.0 / res.P, rel=1e-12)


# -- vs-T comparison table ------------------------------------------------------


def test_compare_vs_T_table():
    air = AirTransport()
    Ts = np.linspace(280.0, 360.0, 10)
    tbl = air.compare_vs_T("mu", Ts, P=P0, relative_humidity=0.5)
    assert list(tbl.columns.keys()) == ["Dry air", "Humid air", "Humid - Dry"]
    n = len(Ts)
    for col in tbl.columns.values():
        assert len(col) == n
    # the difference column is exactly humid - dry
    for i in range(n):
        assert tbl.columns["Humid - Dry"][i] == pytest.approx(
            tbl.columns["Humid air"][i] - tbl.columns["Dry air"][i], rel=1e-12, abs=1e-20
        )
    # x_K tracks the kelvin axis
    assert tbl.x_K == pytest.approx(list(Ts))


def test_compare_vs_T_all_props_run():
    """Every headline property produces a finite dry + humid column over a T sweep."""
    air = AirTransport()
    Ts = np.linspace(250.0, 400.0, 8)
    for prop in AIR_TRANSPORT_PROPS:
        tbl = air.compare_vs_T(prop, Ts, P=P0, relative_humidity=0.5)
        for col in ("Dry air", "Humid air"):
            assert all(math.isfinite(v) for v in tbl.columns[col]), prop


# -- export round-trip ----------------------------------------------------------


def test_export_round_trips(tmp_path):
    """CSV / Excel / JSON / PDF all produce non-empty files."""
    air = AirTransport()
    res = air.humid(T0, P0, relative_humidity=0.5)
    Ts = np.linspace(280.0, 360.0, 6)
    table = air.compare_vs_T("mu", Ts, P=P0, relative_humidity=0.5)
    exp = AirTransportExporter(res, table)

    csv_p = tmp_path / "air.csv"
    xlsx_p = tmp_path / "air.xlsx"
    json_p = tmp_path / "air.json"
    pdf_p = tmp_path / "air.pdf"
    exp.to_csv(csv_p)
    exp.to_excel(xlsx_p)
    exp.to_json(json_p)
    exp.to_pdf(pdf_p)
    for p in (csv_p, xlsx_p, json_p, pdf_p):
        assert p.exists() and p.stat().st_size > 0
    # JSON is structured: properties + components (+ table)
    data = json.loads(json_p.read_text(encoding="utf-8"))
    assert "properties" in data and "components" in data
    assert "table" in data  # the comparison table was attached


def test_comparison_table_direct_exports(tmp_path):
    """The AirTransportTable itself exports to JSON/PDF (inherited from ComparisonTable)."""
    air = AirTransport()
    Ts = np.linspace(280.0, 360.0, 5)
    tbl = air.compare_vs_T("k", Ts, P=P0, relative_humidity=0.5)
    json_p = tmp_path / "k.json"
    pdf_p = tmp_path / "k.pdf"
    tbl.to_json(json_p)
    tbl.to_pdf(pdf_p)
    assert json_p.stat().st_size > 0 and pdf_p.stat().st_size > 0
    data = json.loads(json_p.read_text(encoding="utf-8"))
    assert "columns" in data and "Dry air" in data["columns"]


# -- extended species-transport database ----------------------------------------


def test_species_database_has_five_air_species():
    names = list_species_transport()
    for required in ("N2", "O2", "AR", "CO2", "H2O"):
        assert required in names, f"{required} missing from transport database"


def test_species_record_has_critical_and_acentric():
    rec = get_species_transport("N2")
    assert rec.critical is not None
    assert rec.critical.Tc == pytest.approx(126.19, rel=1e-4)
    assert rec.critical.Pc == pytest.approx(3.3958e6, rel=1e-4)
    assert rec.acentric_factor is not None
    assert 0.03 < rec.acentric_factor < 0.05
    # LJ mirrored from the molecular database
    assert rec.sigma_angstrom > 0
    assert rec.epsilon_over_k > 0
    assert rec.molar_mass_gmol == pytest.approx(28.0134, rel=1e-4)


def test_molecule_has_no_critical_attribute():
    """The extended transport fields live outside Molecule (regression guard)."""
    mol = get("N2")
    assert not hasattr(mol, "critical")
    assert not hasattr(mol, "has_critical")
    # but the extended record is reachable via the transport registry
    rec = get_species_transport("N2")
    assert rec.has_critical


def test_species_record_as_dict():
    rec = get_species_transport("H2O")
    d = rec.as_dict()
    assert d["name"] == "H2O"
    assert d["critical"] is not None
    assert {"Tc", "Pc", "Vc", "Zc"} <= set(d["critical"].keys())


# -- mixture transport calculator direct ---------------------------------------


def test_mixture_transport_calculator_on_dry_air():
    """The calculator on the dry-air mixture matches the facade."""
    air = AirTransport()
    calc = MixtureTransportCalculator(air.dry_air)
    res = calc.compute(State(T=T0, P=P0), label="dry")
    assert res.mu == pytest.approx(air.dry(T0, P0).mu, rel=1e-12)
    assert res.label == "dry"


def test_property_vs_T_vectorised():
    air = AirTransport()
    calc = MixtureTransportCalculator(air.dry_air)
    Ts = [280.0, 300.0, 320.0]
    xs, vals = calc.property_vs_T("mu", Ts, P0)
    assert xs == Ts
    # viscosity rises with temperature for a dilute gas
    assert vals[0] < vals[1] < vals[2]


# -- CLI: REPL `airtransport` and one-shot subcommand ---------------------------


def test_cli_repl_airtransport_mixed_case_props(tmp_path, capsys):
    """``airtransport <prop>`` accepts the mixed-case property keys (Pr/Sc/Le/D_eff).

    Regression for the case-sensitivity bug where ``parts[0].lower()`` was compared against the
    mixed-case ``_AIR_TRANSPORT_PROPS`` keys, rejecting Pr/Sc/Le/D_eff.
    """
    from statthermopy.cli.app import StatThermoPyShell

    sh = StatThermoPyShell()
    for prop in ("Pr", "Sc", "Le", "D_eff"):
        out = tmp_path / f"air_{prop}.png"
        sh.onecmd(f"airtransport {prop} 250 400 4 {out}")
        result = capsys.readouterr().out
        assert f"saved plot -> {out}" in result, f"prop {prop} rejected by REPL: {result!r}"
        assert out.exists() and out.stat().st_size > 0


def test_cli_repl_airtransport_point_and_db(capsys):
    """``airtransport`` (dry point) and ``airtransport db`` print the expected tables."""
    from statthermopy.cli.app import StatThermoPyShell

    sh = StatThermoPyShell()
    sh.onecmd("T = 300")
    sh.onecmd("P = 101325")
    sh.onecmd("airtransport")
    out = capsys.readouterr().out
    assert "Dry air" in out and "Dynamic viscosity" in out
    assert "Per-species contributions" in out and "N2" in out
    sh.onecmd("airtransport db")
    db = capsys.readouterr().out
    assert "Air-transport species database" in db
    for sp in ("Ar", "CO2", "H2O", "N2", "O2"):
        assert sp in db


def test_cli_one_shot_airtransport_full_report(capsys):
    """The one-shot ``airtransport`` subcommand prints the full dry-air report by default."""
    from statthermopy.cli.app import main

    rc = main(["airtransport", "--T", "300", "--P", "101325"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dry air" in out and "mixing rules: mu=Wilke" in out
    assert "Dynamic viscosity" in out and "Prandtl number" in out


def test_cli_one_shot_airtransport_prop_humid(capsys):
    """``--prop`` reports a single property for the humid-air composition at (T, P)."""
    from statthermopy.cli.app import main

    rc = main(["airtransport", "--T", "300", "--rh", "0.5", "--prop", "Pr"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pr(Humid air)" in out and "0.7" in out


def test_cli_one_shot_airtransport_db_and_species(capsys):
    """``--db`` dumps the species database; ``--species`` dumps the contribution table."""
    from statthermopy.cli.app import main

    rc = main(["airtransport", "--db"])
    assert rc == 0
    assert "Air-transport species database" in capsys.readouterr().out

    rc = main(["airtransport", "--T", "300", "--rh", "0.5", "--species"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Per-species transport contributions" in out and "N2" in out


def test_cli_one_shot_airtransport_export_and_png(tmp_path, capsys):
    """``--export`` writes the point evaluation and ``--png`` writes a comparison plot."""
    from statthermopy.cli.app import main

    jpath = tmp_path / "air.json"
    rc = main(["airtransport", "--T", "300", "--rh", "0.5", "--export", "json", str(jpath)])
    assert rc == 0
    data = json.loads(jpath.read_text())
    assert "properties" in data and "components" in data
    assert "exported" in capsys.readouterr().out

    png = tmp_path / "air_Pr.png"
    rc = main(["airtransport", "--T", "300", "--prop", "Pr", "--png", str(png)])
    assert rc == 0
    assert png.exists() and png.stat().st_size > 0
    assert "saved plot" in capsys.readouterr().out


def test_cli_one_shot_airtransport_bad_prop(capsys):
    """An unknown ``--prop`` is rejected with a helpful message and a non-zero-free exit."""
    from statthermopy.cli.app import main

    main(["airtransport", "--prop", "notaprop"])
    out = capsys.readouterr().out
    assert "unknown property" in out and "Pr" in out


# -- performance: skip the dew-point root-find in transport sweeps ---------------


def test_humid_state_dew_point_flag():
    """``HumidAir.state(dew_point=False)`` skips the dew-point solve but keeps the composition.

    Regression for the GUI plot hang: the dew-point root-find (via the IAPWS liquid model) costs
    ~0.7 s per point, which made a 100-point air-transport plot take ~77 s. AirTransport only needs
    the composition, so it now passes ``dew_point=False``. This test pins the contract: the
    composition / humidity-ratio are unchanged, the dew point is NaN when skipped, and the
    transport facade produces identical values either way.
    """
    import math

    from statthermopy.humidair import HumidAir

    ha = HumidAir()
    full = ha.state(T0, P0, relative_humidity=0.5, wet_bulb=False, dew_point=True)
    fast = ha.state(T0, P0, relative_humidity=0.5, wet_bulb=False, dew_point=False)
    # composition + humidity ratio must match exactly (they don't depend on the dew point)
    assert fast.x_h2o == full.x_h2o
    assert fast.humidity_ratio == full.humidity_ratio
    assert fast.relative_humidity == full.relative_humidity
    # the skipped dew point is NaN; the full one is finite
    assert math.isfinite(full.dew_point) and not math.isfinite(fast.dew_point)


def test_airtransport_humid_matches_full_state_path():
    """``AirTransport.humid`` (fast path) matches a humid point built from the full state."""
    from statthermopy.core.state import State
    from statthermopy.humidair import HumidAir
    from statthermopy.transport.air.mixture_transport import MixtureTransportCalculator

    air = AirTransport()
    fast = air.humid(T0, P0, relative_humidity=0.5)
    # reference: full state (with dew point), same composition -> same transport
    ha = HumidAir().state(T0, P0, relative_humidity=0.5, wet_bulb=False, dew_point=True)
    mix = HumidAir()._humid_mixture(ha.x_h2o)
    ref = MixtureTransportCalculator(mix).compute(State(T=T0, P=P0))
    for prop in ("mu", "nu", "k", "alpha", "D_eff", "Pr", "Sc", "Le", "rho"):
        assert getattr(fast, prop) == pytest.approx(getattr(ref, prop), rel=1e-12)
    assert fast.humidity_ratio == pytest.approx(ha.humidity_ratio, rel=1e-12)
