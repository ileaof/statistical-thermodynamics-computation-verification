"""Heat capacity of solids: the Einstein and Debye models.

Implements the two classic lattice heat-capacity models of Chapter 8, each of
which interpolates between the low-temperature quantum freeze-out and the
high-temperature Dulong-Petit limit ``3R``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .constants import R


def einstein_heat_capacity(T, theta_E, molar: bool = True) -> np.ndarray:
    r"""Einstein heat capacity of a solid.

    .. math::

        C_V = 3 R \left(\frac{\theta_E}{T}\right)^2
              \frac{e^{\theta_E/T}}{(e^{\theta_E/T} - 1)^2}.

    Parameters
    ----------
    T : array_like
        Temperature in kelvin.
    theta_E : float
        Einstein temperature :math:`\hbar\omega_E/k_B` in kelvin.
    molar : bool, optional
        If ``True`` (default) return the molar heat capacity in J/(mol K); if
        ``False`` return the dimensionless ``C_V / 3R``.

    Returns
    -------
    numpy.ndarray
        The heat capacity.
    """
    x = theta_E / np.asarray(T, float)
    emx = np.exp(-x)
    reduced = x ** 2 * emx / (1.0 - emx) ** 2
    return 3.0 * R * reduced if molar else reduced


def debye_integrand(x) -> np.ndarray:
    r"""Debye integrand :math:`x^4 e^x / (e^x - 1)^2`, written stably for large ``x``."""
    x = np.asarray(x, float)
    ex = np.exp(-x)
    return x ** 4 * ex / (1.0 - ex) ** 2


def debye_heat_capacity(T, theta_D, molar: bool = True) -> np.ndarray:
    r"""Debye heat capacity by numerical integration.

    .. math::

        C_V = 9 R \left(\frac{T}{\theta_D}\right)^3
              \int_0^{\theta_D/T}\frac{x^4 e^x}{(e^x - 1)^2}\,dx.

    Parameters
    ----------
    T : array_like
        Temperature in kelvin.
    theta_D : float
        Debye temperature in kelvin.
    molar : bool, optional
        If ``True`` return J/(mol K); otherwise return ``C_V / 3R``.

    Returns
    -------
    numpy.ndarray
        The heat capacity (array with the shape of ``T``).
    """
    T = np.atleast_1d(T).astype(float)
    out = np.empty_like(T)
    for i, t in enumerate(T):
        xD = theta_D / t
        I, _ = quad(debye_integrand, 1e-8, xD, limit=200)
        out[i] = 9.0 * (t / theta_D) ** 3 * I
    reduced = out / 3.0
    return (3.0 * R * reduced) if molar else reduced


def dulong_petit(molar: bool = True) -> float:
    r"""The classical Dulong-Petit heat capacity ``3R`` (or ``1`` if ``molar=False``)."""
    return 3.0 * R if molar else 1.0


def debye_T3_coefficient(theta_D) -> float:
    r"""Low-temperature Debye :math:`T^3` coefficient of ``C_V / R``.

    Returns :math:`(12\pi^4/5)/\theta_D^3`, so that
    :math:`C_V/R \approx (12\pi^4/5)(T/\theta_D)^3` at low temperature.
    """
    return float((12.0 * np.pi ** 4 / 5.0) / theta_D ** 3)


__all__ = [
    "einstein_heat_capacity",
    "debye_integrand",
    "debye_heat_capacity",
    "dulong_petit",
    "debye_T3_coefficient",
]
