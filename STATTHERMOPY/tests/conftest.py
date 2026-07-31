"""Shared pytest fixtures and tolerances."""

from __future__ import annotations

import pytest

# Numerical tolerances.
REL_TOL = 1e-4   # 0.01 % — for closed-form cross-checks
LOOSE_TOL = 2e-2  # 2 %   — for literature value comparisons


@pytest.fixture(scope="session")
def constants():
    from statthermopy import constants as c
    return c


@pytest.fixture
def standard_state():
    from statthermopy import State
    return State(T=298.15, P=101325.0)