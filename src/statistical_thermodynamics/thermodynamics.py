"""Bulk thermodynamics of the ideal gas and generic canonical ensembles.

This module gathers the results that connect a partition function to measurable
thermodynamics: the thermal de Broglie wavelength, the Sackur-Tetrode entropy of
a monatomic ideal gas, the Helmholtz energy and chemical potential with the
Gibbs ``1/N!`` correction, and the fluctuation route to the heat capacity.

All functions use SI units and the constants from
:mod:`statistical_thermodynamics.constants`.
"""

from __future__ import annotations

import numpy as np

from .constants import h, k_B, R


def thermal_wavelength(m, T) -> np.ndarray:
    r"""Thermal de Broglie wavelength :math:`\Lambda = h/\sqrt{2\pi m k_B T}`.

    Parameters
    ----------
    m : float or array_like
        Particle mass in kilograms.
    T : float or array_like
        Temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        The thermal wavelength in metres.
    """
    return h / np.sqrt(2.0 * np.pi * np.asarray(m, float) * k_B * np.asarray(T, float))


def sackur_tetrode_entropy(N, V, T, m) -> np.ndarray:
    r"""Sackur-Tetrode entropy of a monatomic ideal gas.

    .. math:: S = N k_B\left[\ln\!\frac{V}{N\Lambda^3} + \frac{5}{2}\right].

    Parameters
    ----------
    N : float
        Number of atoms.
    V : float
        Volume in cubic metres.
    T : float
        Temperature in kelvin.
    m : float
        Atomic mass in kilograms.

    Returns
    -------
    numpy.ndarray
        The absolute entropy in J/K.
    """
    lam3 = thermal_wavelength(m, T) ** 3
    return N * k_B * (np.log(V / (N * lam3)) + 2.5)


def sackur_tetrode_molar(m, T, p) -> np.ndarray:
    r"""Standard molar entropy of a monatomic ideal gas at pressure ``p``.

    Uses :math:`V/N = k_B T / p` from the ideal-gas law.

    Parameters
    ----------
    m : float
        Atomic mass in kilograms.
    T : float
        Temperature in kelvin.
    p : float
        Pressure in pascals.

    Returns
    -------
    numpy.ndarray
        Molar entropy in J/(mol K).
    """
    lam = thermal_wavelength(m, T)
    v_per_atom = k_B * T / p
    return R * (np.log(v_per_atom / lam ** 3) + 2.5)


def helmholtz_ideal_gas(N, V, T, m) -> np.ndarray:
    r"""Helmholtz free energy with the Gibbs ``1/N!`` factor.

    .. math:: F = -N k_B T\left[\ln\!\frac{V}{N\Lambda^3} + 1\right].
    """
    lam3 = thermal_wavelength(m, T) ** 3
    return -N * k_B * T * (np.log(V / (N * lam3)) + 1.0)


def chemical_potential_ideal_gas(N, V, T, m) -> np.ndarray:
    r"""Chemical potential :math:`\mu = -k_B T\,\ln[V/(N\Lambda^3)] = k_B T\,\ln(n\Lambda^3)`."""
    lam3 = thermal_wavelength(m, T) ** 3
    return -k_B * T * np.log(V / (N * lam3))


def heat_capacity_from_fluctuation(mean_E, mean_E2, T, k_B_: float = k_B) -> float:
    r"""Heat capacity from energy fluctuations.

    .. math:: C = (\langle E^2\rangle - \langle E\rangle^2) / k_B T^2.

    Parameters
    ----------
    mean_E : float
        Mean energy :math:`\langle E\rangle`.
    mean_E2 : float
        Mean-square energy :math:`\langle E^2\rangle`.
    T : float
        Temperature.
    k_B_ : float, optional
        Boltzmann constant (defaults to the SI value; pass ``1`` for reduced
        units).

    Returns
    -------
    float
        The heat capacity.
    """
    return float((mean_E2 - mean_E ** 2) / (k_B_ * T ** 2))


__all__ = [
    "thermal_wavelength",
    "sackur_tetrode_entropy",
    "sackur_tetrode_molar",
    "helmholtz_ideal_gas",
    "chemical_potential_ideal_gas",
    "heat_capacity_from_fluctuation",
]
