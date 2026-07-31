"""Tests for the embedded NIST/JANAF reference data and the ``validate`` convenience."""

from __future__ import annotations

import pandas as pd
import pytest

from statthermopy.validation import (
    EmbeddedReferenceSource,
    NistJanafReference,
    ReferenceRegistry,
    list_references,
    validate,
)

# Core subset shipped in Phase 2.
_EXPECTED = {"AR", "CH4", "CO", "CO2", "H2", "H2O", "N2", "NO", "O2"}
# Extended set added in Phase 4 (He/Ne/Kr/Xe/Cl2 from NIST Shomate; C2H6/C3H8 from NASA Glenn
# 7-coefficient polynomials because NIST WebBook has no Shomate for them).
_EXTENDED = {
    "HE",
    "NE",
    "KR",
    "XE",
    "CL2",
    "NH3",
    "SO2",
    "H2S",
    "N2O",
    "C2H2",
    "C2H4",
    "C2H6",
    "C3H8",
}
# Rigid-rotor/harmonic-oscillator model departs from experiment by up to ~4% (H2 at high T);
# 5% mean abs error is the documented validation tolerance.
_MAE_TOL = 5.0


def test_list_references_contains_core_subset():
    refs = set(list_references())
    assert refs >= _EXPECTED


def test_list_references_contains_extended_set():
    refs = set(list_references())
    assert refs >= _EXTENDED
    # together they cover every species in the molecular database
    assert refs == set(_EXPECTED) | set(_EXTENDED)


@pytest.mark.parametrize("species", sorted(_EXPECTED))
def test_reference_load_returns_dataframe_with_T_Cp_S(species):
    src = NistJanafReference(species)
    df = src.load()
    assert isinstance(df, pd.DataFrame)
    assert "T" in df.columns
    assert "Cp" in df.columns
    assert "S" in df.columns
    assert len(df) >= 5  # H2O has 6 points; the rest have 7
    assert (df["T"] > 0).all()
    assert (df["Cp"] > 0).all()
    assert (df["S"] > 0).all()


def test_reference_pressure_is_one_bar():
    assert NistJanafReference("N2").pressure == pytest.approx(100000.0)


def test_reference_source_and_available_properties():
    src = NistJanafReference("N2")
    assert src.available_properties() == ("Cp", "S")
    assert "NIST" in src.source or "JANAF" in src.source


def test_nistjanaf_is_embedded_subclass():
    assert issubclass(NistJanafReference, EmbeddedReferenceSource)


@pytest.mark.parametrize("species", sorted(_EXPECTED))
@pytest.mark.parametrize("prop", ["Cp", "S"])
def test_validate_within_tolerance(species, prop):
    report = validate(species, prop)
    assert report.species.upper() == species
    assert report.property_name == prop
    assert len(report.T) >= 5
    assert (
        report.mean_abs_error_percent < _MAE_TOL
    ), f"{species}/{prop}: MAE={report.mean_abs_error_percent:.3f}% exceeds {_MAE_TOL}%"


@pytest.mark.parametrize("species", sorted(_EXTENDED))
@pytest.mark.parametrize("prop", ["Cp", "S"])
def test_validate_extended_within_tolerance(species, prop):
    report = validate(species, prop)
    assert report.species.upper() == species
    assert report.property_name == prop
    assert len(report.T) >= 5
    assert (
        report.mean_abs_error_percent < _MAE_TOL
    ), f"{species}/{prop}: MAE={report.mean_abs_error_percent:.3f}% exceeds {_MAE_TOL}%"


def test_validate_ar_is_near_exact():
    # Monatomic Ar: Cp = 5R/2, S from Sackur-Tetrode — the engine is essentially exact.
    for prop in ("Cp", "S"):
        report = validate("Ar", prop)
        assert report.mean_abs_error_percent < 0.1


@pytest.mark.parametrize("species", ["He", "Ne", "Kr", "Xe"])
def test_validate_noble_gases_near_exact(species):
    # Monatomic noble gases: Cp = 5R/2, S from Sackur-Tetrode — engine is essentially exact.
    for prop in ("Cp", "S"):
        report = validate(species, prop)
        assert (
            report.mean_abs_error_percent < 0.1
        ), f"{species}/{prop}: MAE={report.mean_abs_error_percent:.4f}% not near-exact"


def test_validate_uses_reference_pressure_for_entropy():
    # Entropy depends on pressure (translational term). validate() must default to the
    # reference's standard-state pressure (1 bar), not the ValidationRunner's 1-atm default.
    report = validate("N2", "S")
    assert report.mean_abs_error_percent < 1.0  # would be off by ~1.3% if run at 1 atm


def test_validate_explicit_pressure_overrides_reference():
    r_default = validate("N2", "S")
    r_override = validate("N2", "S", pressure=101325.0)  # 1 atm instead of 1 bar
    # the two runs use different pressures -> different predicted S -> different errors
    assert r_override.predicted[0] != pytest.approx(r_default.predicted[0], rel=1e-4)


def test_registry_get_and_list():
    reg = ReferenceRegistry()
    assert set(reg.list_references()) == set(list_references())
    src = reg.get("n2")  # case-insensitive
    assert isinstance(src, EmbeddedReferenceSource)
    assert src.load()["T"].iloc[0] == pytest.approx(298.15)


def test_validate_missing_species_raises():
    # n-decane is in neither the molecule registry nor the reference set.
    with pytest.raises(KeyError):
        validate("C10H22", "Cp")


def test_reference_missing_species_raises():
    with pytest.raises(KeyError):
        EmbeddedReferenceSource("XYZZY").load()


def test_validate_returns_report_with_aligned_arrays():
    report = validate("CO2", "Cp")
    assert (
        len(report.T)
        == len(report.predicted)
        == len(report.reference)
        == len(report.errors_percent)
    )
    for p, r in zip(report.predicted, report.reference, strict=False):
        assert isinstance(p, float) and isinstance(r, float)
