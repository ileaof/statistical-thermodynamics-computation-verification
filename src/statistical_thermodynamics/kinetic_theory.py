"""Kinetic theory of gases: speed distributions, collisions and the mean free path.

The functions here reproduce the closed-form results of Chapter 2 -- the
Maxwell-Boltzmann speed distribution, the three characteristic speeds, and the
collision frequency and mean free path of a dilute hard-sphere gas.
"""

from __future__ import annotations

import numpy as np

from .constants import k_B


def maxwell_speed_pdf(v, m, T) -> np.ndarray:
    r"""Maxwell-Boltzmann speed probability density (per m/s), SI units.

    .. math::

        f(v) = 4\pi\left(\frac{m}{2\pi k_B T}\right)^{3/2}
               v^2 \exp\!\left(-\frac{m v^2}{2 k_B T}\right).

    Parameters
    ----------
    v : array_like
        Molecular speed in m/s.
    m : float
        Molecular mass in kilograms.
    T : float
        Temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        The probability density in s/m.
    """
    v = np.asarray(v, dtype=float)
    a = m / (2.0 * np.pi * k_B * T)
    return 4.0 * np.pi * a ** 1.5 * v ** 2 * np.exp(-m * v ** 2 / (2.0 * k_B * T))


def characteristic_speeds(m, T):
    r"""Most-probable, mean and root-mean-square speeds of a Maxwellian gas.

    Returns the closed forms

    .. math::

        v_p = \sqrt{2k_BT/m},\quad
        \langle v\rangle = \sqrt{8k_BT/\pi m},\quad
        v_{\rm rms} = \sqrt{3k_BT/m}.

    Parameters
    ----------
    m : float
        Molecular mass in kilograms.
    T : float
        Temperature in kelvin.

    Returns
    -------
    tuple of float
        ``(v_p, v_avg, v_rms)`` in m/s.
    """
    v_p = np.sqrt(2.0 * k_B * T / m)
    v_avg = np.sqrt(8.0 * k_B * T / (np.pi * m))
    v_rms = np.sqrt(3.0 * k_B * T / m)
    return float(v_p), float(v_avg), float(v_rms)


def mean_speed(m, T) -> float:
    r"""Mean molecular speed :math:`\langle v\rangle = \sqrt{8k_BT/\pi m}` (m/s)."""
    return float(np.sqrt(8.0 * k_B * T / (np.pi * m)))


def collision_cross_section(d: float) -> float:
    r"""Hard-sphere collision cross-section :math:`\sigma = \pi d^2`.

    Parameters
    ----------
    d : float
        Effective molecular diameter in metres.

    Returns
    -------
    float
        The cross-section in square metres.
    """
    return float(np.pi * d ** 2)


def number_density(p, T) -> float:
    r"""Ideal-gas number density :math:`n = p / k_B T` (per cubic metre)."""
    return float(p / (k_B * T))


def collision_frequency(n, d, m, T) -> float:
    r"""Collision frequency :math:`z = \sqrt{2}\, n\sigma\langle v\rangle` (per second).

    The :math:`\sqrt2` accounts for the relative motion of colliding molecules,
    both drawn from the Maxwell-Boltzmann distribution.

    Parameters
    ----------
    n : float
        Number density in m^-3.
    d : float
        Molecular diameter in metres.
    m : float
        Molecular mass in kilograms.
    T : float
        Temperature in kelvin.

    Returns
    -------
    float
        Collisions per second per molecule.
    """
    sigma = collision_cross_section(d)
    return float(np.sqrt(2.0) * n * sigma * mean_speed(m, T))


def mean_free_path(n, d) -> float:
    r"""Mean free path :math:`\lambda = 1/(\sqrt2\, n\sigma)` (metres).

    Parameters
    ----------
    n : float
        Number density in m^-3.
    d : float
        Molecular diameter in metres.

    Returns
    -------
    float
        The mean free path in metres.
    """
    sigma = collision_cross_section(d)
    return float(1.0 / (np.sqrt(2.0) * n * sigma))


__all__ = [
    "maxwell_speed_pdf",
    "characteristic_speeds",
    "mean_speed",
    "collision_cross_section",
    "number_density",
    "collision_frequency",
    "mean_free_path",
]
