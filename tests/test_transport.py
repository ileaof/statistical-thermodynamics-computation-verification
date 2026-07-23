"""Tests for elementary transport coefficients."""

import numpy as np

from statistical_thermodynamics import constants as C
from statistical_thermodynamics import transport as tr


def test_viscosity_independent_of_density():
    d, m, T = 3.7e-10, 28.0134 * C.u, 300.0
    eta1 = tr.viscosity(1e25, d, m, T)
    eta2 = tr.viscosity(4e25, d, m, T)
    # Maxwell's result: viscosity is independent of number density.
    assert np.isclose(eta1, eta2, rtol=1e-10)


def test_viscosity_sqrt_T_scaling():
    d, m, n = 3.7e-10, 28.0134 * C.u, 2.5e25
    eta300 = tr.viscosity(n, d, m, 300.0)
    eta1200 = tr.viscosity(n, d, m, 1200.0)
    assert np.isclose(eta1200 / eta300, 2.0, rtol=1e-10)


def test_air_viscosity_order_of_magnitude():
    # Air near room temperature has eta ~ 1.8 x 10^-5 Pa s (within a factor ~2).
    eta = tr.viscosity(2.5e25, 3.7e-10, 28.0134 * C.u, 300.0)
    assert 5e-6 < eta < 5e-5


def test_diffusion_positive_and_scales_with_T():
    d, m = 3.7e-10, 28.0134 * C.u
    D_lo = tr.diffusion_coefficient(2.5e25, d, m, 200.0)
    D_hi = tr.diffusion_coefficient(2.5e25, d, m, 800.0)
    assert D_lo > 0 and D_hi > D_lo
