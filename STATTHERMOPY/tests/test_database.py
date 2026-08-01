"""Tests for the molecular database."""

from __future__ import annotations

import math

import pytest

from statthermopy import get, list_molecules
from statthermopy.core.molecule import Geometry


EXPECTED = [
    "HE", "NE", "AR", "KR", "XE",
    "H2", "N2", "O2", "CL2", "CO", "NO",
    "H2O", "CO2", "NH3", "CH4", "SO2", "H2S", "N2O", "C2H2", "C2H4", "C2H6", "C3H8",
]


def test_all_22_molecules_present():
    names = set(list_molecules())
    for n in EXPECTED:
        assert n in names, f"missing {n}"


def test_case_insensitive_lookup():
    assert get("n2").name == "N2"
    assert get("Ar").name == "Ar"


def test_unknown_molecule_raises():
    with pytest.raises(KeyError):
        get("unobtainium")


def test_molecule_geometry_classes():
    assert get("He").is_monoatomic
    assert get("N2").is_diatomic
    assert get("CO2").is_linear and not get("CO2").is_diatomic
    assert get("H2O").is_nonlinear


def test_diatomic_symmetry_numbers():
    assert get("N2").symmetry_number == 2   # homonuclear
    assert get("CO").symmetry_number == 1    # heteronuclear
    assert get("NO").symmetry_number == 1


def test_vibrational_oscillator_counts():
    # 3N-5 linear, 3N-6 nonlinear; internal rotors (ethane/propane methyl torsions) count
    # toward the total but are carried separately from the harmonic oscillators.
    assert get("N2").n_vibrational_modes == 1
    assert get("CO2").n_vibrational_modes == 4   # 1+2+1
    assert get("H2O").n_vibrational_modes == 3
    assert get("CH4").n_vibrational_modes == 9   # 1+2+3+3
    # C2H6: 17 harmonic oscillators + 1 internal rotor = 18 = 3N-6
    assert get("C2H6").n_vibrational_modes == 17
    assert get("C2H6").n_internal_rotors == 1
    assert get("C2H6").n_vibrational_modes + get("C2H6").n_internal_rotors == 18
    # C3H8: 25 harmonic oscillators + 2 internal rotors = 27 = 3N-6
    assert get("C3H8").n_vibrational_modes == 25
    assert get("C3H8").n_internal_rotors == 2
    assert get("C3H8").n_vibrational_modes + get("C3H8").n_internal_rotors == 27


def test_moments_of_inertia_counts():
    assert get("Ar").moments_of_inertia == ()
    assert len(get("N2").moments_of_inertia) == 1
    assert len(get("H2O").moments_of_inertia) == 3


def test_electronic_ground_state_degeneracies():
    assert get("Ar").ground_state_degeneracy == 1
    assert get("O2").ground_state_degeneracy == 3   # triplet
    assert get("NO").ground_state_degeneracy == 2


def test_database_is_extensible(tmp_path):
    from statthermopy.database.registry import MoleculeRegistry
    import yaml
    data = {
        "name": "Fake", "formula": "Fake", "molar_mass_gmol": 50.0,
        "geometry": "monoatomic", "n_atoms": 1,
    }
    p = tmp_path / "fake.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    reg = MoleculeRegistry(data_dir=tmp_path)
    mol = reg.get("fake")
    assert mol.name == "Fake"
    assert math.isclose(mol.molar_mass_gmol, 50.0)


def test_rotational_temperatures_to_moments():
    from statthermopy.database.registry import _rotational_temperatures_to_moments
    moments = _rotational_temperatures_to_moments([2.878])
    # round trip: theta = h^2/(8pi^2 I k) -> recompute theta from I
    from statthermopy.modes.rotational import rotational_temperature
    assert math.isclose(rotational_temperature(moments[0]), 2.878, rel_tol=1e-9)