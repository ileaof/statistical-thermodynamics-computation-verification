"""Molecular partition functions and the thermodynamics that follow from them.

Each contribution (translation, rotation, vibration, and the generic two-level
and harmonic systems) is provided both as a closed form and, where instructive,
as a direct sum over levels, mirroring the "compute two independent ways and
compare" philosophy of the book.

Unless stated otherwise, characteristic temperatures ``theta`` are in kelvin and
``T`` is in kelvin, so the exponents are dimensionless ratios ``theta / T``.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Two-level system
# --------------------------------------------------------------------------
def two_level_partition(T, eps: float = 1.0, k_B: float = 1.0) -> np.ndarray:
    r"""Two-level partition function :math:`Z = 1 + e^{-\varepsilon/k_BT}`."""
    return 1.0 + np.exp(-eps / (k_B * np.asarray(T, float)))


def two_level_energy(T, eps: float = 1.0, k_B: float = 1.0) -> np.ndarray:
    r"""Mean energy of a two-level system, :math:`U = \varepsilon / (e^{\varepsilon/k_BT} + 1)`."""
    return eps / (np.exp(eps / (k_B * np.asarray(T, float))) + 1.0)


def two_level_heat_capacity(T, eps: float = 1.0, k_B: float = 1.0) -> np.ndarray:
    r"""Two-level (Schottky) heat capacity :math:`C = k_B x^2 e^x / (e^x + 1)^2`."""
    x = eps / (k_B * np.asarray(T, float))
    ex = np.exp(x)
    return k_B * x ** 2 * ex / (ex + 1.0) ** 2


# --------------------------------------------------------------------------
# Harmonic oscillator (Einstein mode)
# --------------------------------------------------------------------------
def harmonic_partition(T, theta: float = 1.0) -> np.ndarray:
    r"""Harmonic-oscillator partition function :math:`Z = 1/[2\sinh(\theta/2T)]`.

    Uses the closed form of :math:`\sum_n e^{-(n+1/2)\theta/T}`, i.e. the
    zero-point-referenced geometric series.
    """
    x = theta / np.asarray(T, float)
    return 1.0 / (2.0 * np.sinh(x / 2.0))


def harmonic_energy(T, theta: float = 1.0) -> np.ndarray:
    r"""Harmonic-oscillator mean energy :math:`U/\hbar\omega = 1/2 + 1/(e^{\theta/T} - 1)`."""
    x = theta / np.asarray(T, float)
    return 0.5 + 1.0 / (np.exp(x) - 1.0)


def harmonic_heat_capacity(T, theta: float = 1.0, k_B: float = 1.0) -> np.ndarray:
    r"""Einstein heat capacity :math:`C/k_B = x^2 e^x / (e^x - 1)^2`, ``x = theta/T``.

    Written with ``e^{-x}`` internally so that it is numerically stable in the
    low-temperature (``x`` large) limit.
    """
    x = theta / np.asarray(T, float)
    emx = np.exp(-x)
    return k_B * x ** 2 * emx / (1.0 - emx) ** 2


def harmonic_partition_sum(T, theta: float = 1.0, n_max: int = 200):
    r"""Harmonic thermodynamics by direct summation over ``n = 0..n_max``.

    Parameters
    ----------
    T : float
        Temperature (units of ``theta``).
    theta : float, optional
        Characteristic temperature :math:`\hbar\omega/k_B`.
    n_max : int, optional
        Highest level retained in the sum.

    Returns
    -------
    tuple of float
        ``(Z, U, E2)`` -- partition function, mean energy and mean-square
        energy, all referenced to the level spacing ``theta``.
    """
    n = np.arange(n_max + 1)
    e = (n + 0.5) * theta
    w = np.exp(-e / T)
    Z = w.sum()
    U = (e * w).sum() / Z
    E2 = (e ** 2 * w).sum() / Z
    return float(Z), float(U), float(E2)


# --------------------------------------------------------------------------
# Rigid rotor
# --------------------------------------------------------------------------
def rotational_partition_sum(T, theta_rot: float, sigma: int = 1,
                             J_max: int = 2000) -> np.ndarray:
    r"""Exact linear-rotor partition function by summation over ``J``.

    .. math::

        z_{\rm rot} = \frac{1}{\sigma}\sum_{J=0}^{\infty}
                       (2J+1)\, e^{-\theta_{\rm rot} J(J+1)/T}.

    Parameters
    ----------
    T : array_like
        Temperature(s).
    theta_rot : float
        Rotational temperature :math:`\hbar^2/2Ik_B`.
    sigma : int, optional
        Symmetry number (1 heteronuclear, 2 homonuclear).
    J_max : int, optional
        Highest rotational level summed.

    Returns
    -------
    numpy.ndarray
        ``z_rot`` evaluated at each temperature.
    """
    J = np.arange(0, J_max + 1)
    g = (2 * J + 1).astype(float)
    T = np.atleast_1d(T).astype(float)
    out = np.empty_like(T)
    for i, t in enumerate(T):
        out[i] = np.sum(g * np.exp(-theta_rot * J * (J + 1) / t)) / sigma
    return out


def rotational_partition_high_T(T, theta_rot: float, sigma: int = 1) -> np.ndarray:
    r"""High-temperature (Euler-Maclaurin) rotational partition function.

    .. math::

        z_{\rm rot} \approx \frac{T}{\sigma\theta_{\rm rot}}
          \left[1 + \tfrac{1}{3}\tfrac{\theta_{\rm rot}}{T}
                + \tfrac{1}{15}\left(\tfrac{\theta_{\rm rot}}{T}\right)^2
                + \tfrac{4}{315}\left(\tfrac{\theta_{\rm rot}}{T}\right)^3\right].
    """
    r = theta_rot / np.asarray(T, float)
    return (1.0 / (sigma * r)) * (1 + r / 3 + r ** 2 / 15 + 4 * r ** 3 / 315)


# --------------------------------------------------------------------------
# Vibration and translation
# --------------------------------------------------------------------------
def vibrational_partition(T, theta_vib: float) -> np.ndarray:
    r"""Vibrational partition function :math:`z_{\rm vib} = 1/(1 - e^{-\theta_{\rm vib}/T})`."""
    return 1.0 / (1.0 - np.exp(-theta_vib / np.asarray(T, float)))


def translational_z1_sum(alpha: float) -> float:
    r"""One-dimensional particle-in-a-box sum :math:`\sum_{n\ge 1} e^{-\alpha n^2}`.

    Parameters
    ----------
    alpha : float
        Level spacing in units of ``k_B T``, :math:`\alpha = h^2/(8mL^2k_BT)`.

    Returns
    -------
    float
        The truncated sum (truncation error is exponentially small).
    """
    n_max = max(50, int(10.0 / np.sqrt(alpha)))
    n = np.arange(1, n_max + 1, dtype=float)
    return float(np.sum(np.exp(-alpha * n ** 2)))


def translational_z1_continuum(alpha: float) -> float:
    r"""Continuum approximation :math:`z_1 = \tfrac12\sqrt{\pi/\alpha} = L/\Lambda`."""
    return 0.5 * np.sqrt(np.pi / alpha)


__all__ = [
    "two_level_partition", "two_level_energy", "two_level_heat_capacity",
    "harmonic_partition", "harmonic_energy", "harmonic_heat_capacity",
    "harmonic_partition_sum",
    "rotational_partition_sum", "rotational_partition_high_T",
    "vibrational_partition",
    "translational_z1_sum", "translational_z1_continuum",
]
