"""Intermolecular pair potentials and their forces.

The Lennard-Jones and hard-sphere potentials underpin the molecular-dynamics and
virial-coefficient studies of Chapters 2 and 7.  Functions accept reduced units
by default (``epsilon = sigma = 1``) but take explicit parameters so they can be
used with real gases.
"""

from __future__ import annotations

import numpy as np


def lennard_jones(r, epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    r"""Lennard-Jones pair potential.

    .. math:: V(r) = 4\varepsilon\left[(\sigma/r)^{12} - (\sigma/r)^6\right].

    Parameters
    ----------
    r : array_like
        Pair separation (same length unit as ``sigma``).
    epsilon : float, optional
        Well depth.
    sigma : float, optional
        Distance at which the potential is zero.

    Returns
    -------
    numpy.ndarray
        The potential energy.
    """
    sr6 = (sigma / np.asarray(r, float)) ** 6
    return 4.0 * epsilon * (sr6 ** 2 - sr6)


def lennard_jones_force(r, epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    r"""Magnitude of the Lennard-Jones force :math:`f(r) = -dV/dr`.

    .. math:: f(r) = \frac{24\varepsilon}{r}\left[2(\sigma/r)^{12} - (\sigma/r)^6\right].

    Positive values are repulsive (directed to increase ``r``).
    """
    r = np.asarray(r, float)
    sr6 = (sigma / r) ** 6
    return 24.0 * epsilon / r * (2.0 * sr6 ** 2 - sr6)


def lennard_jones_minimum(sigma: float = 1.0):
    r"""Location and depth of the Lennard-Jones minimum.

    Returns
    -------
    tuple of float
        ``(r_min, V_min)`` where :math:`r_{\min} = 2^{1/6}\sigma` and
        :math:`V_{\min} = -\varepsilon` (here reported as ``-1`` in units of
        ``epsilon``).
    """
    return 2.0 ** (1.0 / 6.0) * sigma, -1.0


def hard_sphere(r, sigma: float = 1.0) -> np.ndarray:
    r"""Hard-sphere potential: ``inf`` for ``r < sigma`` and ``0`` otherwise."""
    r = np.asarray(r, float)
    return np.where(r < sigma, np.inf, 0.0)


def mayer_f(r, T, potential=lennard_jones, **kwargs) -> np.ndarray:
    r"""Mayer function :math:`f(r) = e^{-V(r)/k_B T} - 1`.

    Parameters
    ----------
    r : array_like
        Pair separation.
    T : float
        Temperature in the same energy unit as the potential (i.e. ``k_B = 1``
        in reduced units, so ``T`` is ``k_B T / epsilon``).
    potential : callable, optional
        A pair-potential function ``V(r, **kwargs)``.  Defaults to
        :func:`lennard_jones`.
    **kwargs
        Extra parameters forwarded to ``potential``.

    Returns
    -------
    numpy.ndarray
        The Mayer f-function, the integrand of the second virial coefficient.
    """
    V = potential(r, **kwargs)
    return np.expm1(-V / T)


__all__ = [
    "lennard_jones",
    "lennard_jones_force",
    "lennard_jones_minimum",
    "hard_sphere",
    "mayer_f",
]
