"""Shared pytest fixtures for ThermoLab tests."""

from __future__ import annotations

import warnings

import pytest

warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")  # headless rendering for plotting tests

from thermolab import Gas, Mixture


@pytest.fixture(scope="session")
def air():
    return Gas("Air")


@pytest.fixture(scope="session")
def water():
    return Gas("H2O")


@pytest.fixture(scope="session")
def nitrogen():
    return Gas("N2")


@pytest.fixture(scope="session")
def r134a():
    return Gas("R134a")


@pytest.fixture(scope="session")
def flue_mix():
    return Mixture(["N2", "O2", "CO2", "H2O"], [0.78, 0.21, 0.005, 0.005])


def pytest_collection_modifyitems(config, items):
    # Mark slow integration tests if needed (none currently).
    pass