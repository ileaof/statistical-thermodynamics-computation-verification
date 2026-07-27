"""Mixture tests."""

from __future__ import annotations

import numpy as np
import pytest

from thermolab import Mixture


def test_mixture_basic(flue_mix):
    st = flue_mix.state(T=1200, P=3e5)
    assert st.T == pytest.approx(1200)
    assert st.P == pytest.approx(3e5)
    # ideal-gas rho = P/(R*T) with R~289 J/(kg.K) -> ~0.87 kg/m^3
    assert st.rho == pytest.approx(0.87, rel=0.1)
    assert st.cp > 1100


def test_mixture_fraction_normalization():
    m = Mixture(["N2", "O2"], [0.8, 0.2])
    assert m.fractions.sum() == pytest.approx(1.0)
    m2 = Mixture(["N2", "O2"], [4.0, 1.0])  # unnormalized
    assert m2.fractions[0] == pytest.approx(0.8)


def test_mixture_set_frequencies():
    m = Mixture(["N2", "O2", "CO2"], [0.7, 0.2, 0.1])
    m.set_fractions([0.5, 0.3, 0.2])
    assert m.fractions[2] == pytest.approx(0.2)


def test_mixture_validation():
    with pytest.raises(ValueError):
        Mixture(["N2", "O2"], [0.5])  # length mismatch
    with pytest.raises(ValueError):
        Mixture(["N2", "O2"], [0.5, -0.5])  # negative fraction
    with pytest.raises(ValueError):
        Mixture(["N2", "O2"], [0.0, 0.0])  # zero sum


def test_air_is_alias_mixture():
    from thermolab import Gas
    air = Gas("Air")
    assert air.is_pseudo_mixture
    assert set(air.components) == {"N2", "O2", "Ar", "CO2"}