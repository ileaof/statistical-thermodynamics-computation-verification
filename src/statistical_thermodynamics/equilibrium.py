"""Chemical equilibrium and imperfect-gas equations of state.

Provides the reduced second virial coefficient of a Lennard-Jones gas, its Boyle
temperature, the Carnahan-Starling hard-sphere equation of state, and the degree
of dissociation of a diatomic gas from an equilibrium constant.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def second_virial_lj(Tstar: float) -> float:
    r"""Reduced second virial coefficient of a Lennard-Jones gas.

    .. math::

        B^*(T^*) = -3\int_0^{\infty}
          \left[e^{-(4/T^*)(r^{-12} - r^{-6})} - 1\right] r^2\, dr,

    in units of :math:`b_0 = \tfrac23\pi N_A\sigma^3`, with
    :math:`T^* = k_B T/\varepsilon` and :math:`r` in units of ``sigma``.

    Parameters
    ----------
    Tstar : float
        Reduced temperature :math:`k_B T/\varepsilon`.

    Returns
    -------
    float
        The reduced second virial coefficient :math:`B/b_0`.
    """
    def integrand(r):
        u = 4.0 * (r ** -12 - r ** -6) / Tstar
        return (np.exp(-u) - 1.0) * r ** 2

    val, _ = quad(integrand, 1e-4, 12.0, limit=400)
    return -3.0 * val


def boyle_temperature_lj(bracket=(2.5, 4.5)) -> float:
    r"""Reduced Boyle temperature of the Lennard-Jones gas, where :math:`B^*(T^*) = 0`.

    Parameters
    ----------
    bracket : tuple of float, optional
        Interval that brackets the root (default ``(2.5, 4.5)``).

    Returns
    -------
    float
        :math:`T^*_B` (accepted value ``3.418``).
    """
    return float(brentq(second_virial_lj, *bracket, xtol=1e-8))


def carnahan_starling(eta) -> np.ndarray:
    r"""Carnahan-Starling hard-sphere compressibility factor.

    .. math:: Z = \frac{1 + \eta + \eta^2 - \eta^3}{(1 - \eta)^3},

    an accurate closed form for the hard-sphere fluid up to the freezing packing
    fraction.

    Parameters
    ----------
    eta : array_like
        Packing fraction :math:`\eta = \tfrac{\pi}{6} n \sigma^3`.

    Returns
    -------
    numpy.ndarray
        The compressibility factor :math:`Z = pV/Nk_BT`.
    """
    eta = np.asarray(eta, float)
    return (1 + eta + eta ** 2 - eta ** 3) / (1 - eta) ** 3


def degree_of_dissociation(Kp, P, p0: float = 1.0e5) -> float:
    r"""Degree of dissociation ``alpha`` for ``A2 <-> 2A``.

    From :math:`K_p = (P/p_0)\, 4\alpha^2/(1 - \alpha^2)` one finds

    .. math:: \alpha = \sqrt{K_p / (K_p + 4 P/p_0)}.

    Parameters
    ----------
    Kp : float
        Equilibrium constant.
    P : float
        Total pressure in pascals.
    p0 : float, optional
        Standard pressure in pascals (default 1 bar).

    Returns
    -------
    float
        The degree of dissociation, between 0 and 1.
    """
    return float(np.sqrt(Kp / (Kp + 4.0 * P / p0)))


# Exact hard-sphere virial coefficients (sigma = 1), useful as test targets.
B2_HARD_SPHERE = (2.0 / 3.0) * np.pi          #: exact :math:`B_2 = \tfrac23\pi\sigma^3`
B3_HARD_SPHERE = (5.0 / 18.0) * np.pi ** 2    #: exact :math:`B_3 = \tfrac{5}{18}\pi^2\sigma^6`


__all__ = [
    "second_virial_lj",
    "boyle_temperature_lj",
    "carnahan_starling",
    "degree_of_dissociation",
    "B2_HARD_SPHERE",
    "B3_HARD_SPHERE",
]
