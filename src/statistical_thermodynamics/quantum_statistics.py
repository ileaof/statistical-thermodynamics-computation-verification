"""Quantum statistics: occupation numbers, Bose functions and blackbody radiation.

Contains the three occupation-number distributions (Maxwell-Boltzmann,
Bose-Einstein, Fermi-Dirac), the Bose functions (polylogarithms) that govern the
ideal Bose gas, and the Planck spectrum with the Stefan-Boltzmann flux.

The occupation-number functions take ``k_B`` as a parameter (default ``1``) so
they can be used either in reduced units or with the SI value from
:mod:`statistical_thermodynamics.constants`.
"""

from __future__ import annotations

from math import gamma as _gamma

import numpy as np

from .constants import c, hbar, k_B, sigma_SB
from .utilities import trapezoid


# --------------------------------------------------------------------------
# Occupation numbers
# --------------------------------------------------------------------------
def maxwell_boltzmann(eps, mu, T, k_B_: float = 1.0) -> np.ndarray:
    r"""Classical mean occupation :math:`\langle n\rangle = e^{-(\varepsilon-\mu)/k_BT}`."""
    x = (np.asarray(eps, float) - mu) / (k_B_ * T)
    return np.exp(-x)


def bose_einstein(eps, mu, T, k_B_: float = 1.0) -> np.ndarray:
    r"""Bose-Einstein occupation :math:`\langle n\rangle = 1/(e^{(\varepsilon-\mu)/k_BT} - 1)`.

    Defined for :math:`\varepsilon > \mu`; diverges as :math:`\varepsilon \to \mu^+`.
    """
    x = (np.asarray(eps, float) - mu) / (k_B_ * T)
    return 1.0 / (np.exp(x) - 1.0)


def fermi_dirac(eps, mu, T, k_B_: float = 1.0) -> np.ndarray:
    r"""Fermi-Dirac occupation :math:`f = 1/(e^{(\varepsilon-\mu)/k_BT} + 1)`.

    Evaluated through the numerically stable logistic form
    :math:`f = \tfrac12[1 - \tanh(x/2)]` with ``x = (eps - mu)/k_B T``, which
    avoids overflow for large positive arguments.
    """
    x = (np.asarray(eps, float) - mu) / (k_B_ * T)
    return 0.5 * (1.0 - np.tanh(0.5 * x))


# --------------------------------------------------------------------------
# Bose functions (polylogarithms)
# --------------------------------------------------------------------------
def bose_function(s: float, z: float, n_grid: int = 6000) -> float:
    r"""Bose function :math:`g_s(z)` via its integral representation.

    .. math::

        g_s(z) = \frac{1}{\Gamma(s)}
                 \int_0^{\infty}\frac{x^{s-1}}{z^{-1}e^{x} - 1}\,dx,

    evaluated with the substitution :math:`x = u^2`, which removes the
    integrable singularity at the origin.  At ``z = 1`` this returns
    :math:`\zeta(s)`.

    Parameters
    ----------
    s : float
        Order of the Bose function.
    z : float
        Fugacity, ``0 < z <= 1``.
    n_grid : int, optional
        Number of quadrature points.

    Returns
    -------
    float
        The value of :math:`g_s(z)`.
    """
    u = np.linspace(1e-6, 9.0, n_grid)
    x = u * u
    integrand = 2.0 * u ** (2 * s - 1) / (np.exp(x) / z - 1.0)
    return float(trapezoid(integrand, u) / _gamma(s))


# --------------------------------------------------------------------------
# Blackbody radiation
# --------------------------------------------------------------------------
def planck_u_omega(omega, T) -> np.ndarray:
    r"""Planck spectral energy density per angular frequency (J s m^-3).

    .. math::

        u(\omega, T) = \frac{\hbar\omega^3}{\pi^2 c^3}
                       \frac{1}{e^{\hbar\omega/k_BT} - 1}.
    """
    omega = np.asarray(omega, float)
    x = hbar * omega / (k_B * T)
    return hbar * omega ** 3 / (np.pi ** 2 * c ** 3) / np.expm1(x)


def planck_u_nu(nu, T) -> np.ndarray:
    r"""Planck spectral energy density per ordinary frequency (J s m^-3).

    .. math::

        u(\nu, T) = \frac{8\pi h\nu^3}{c^3}\frac{1}{e^{h\nu/k_BT} - 1}.
    """
    from .constants import h
    nu = np.asarray(nu, float)
    x = h * nu / (k_B * T)
    return 8.0 * np.pi * h * nu ** 3 / c ** 3 / np.expm1(x)


def stefan_boltzmann_flux(T) -> np.ndarray:
    r"""Blackbody radiant exitance :math:`M = \sigma T^4` (W/m^2)."""
    return sigma_SB * np.asarray(T, float) ** 4


__all__ = [
    "maxwell_boltzmann",
    "bose_einstein",
    "fermi_dirac",
    "bose_function",
    "planck_u_omega",
    "planck_u_nu",
    "stefan_boltzmann_flux",
]
