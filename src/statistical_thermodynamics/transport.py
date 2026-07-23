"""Elementary transport coefficients of a dilute gas.

These are the simplest kinetic-theory estimates -- the ``1/3 n <v> lambda``
family -- for the self-diffusion coefficient, shear viscosity and thermal
conductivity of a dilute hard-sphere gas.  They are order-of-magnitude accurate
and reproduce the correct pressure and temperature scaling; the full
Chapman-Enskog treatment refines the numerical prefactors.

All results are in SI units.
"""

from __future__ import annotations

from .constants import k_B
from .kinetic_theory import mean_free_path, mean_speed, number_density


def diffusion_coefficient(n, d, m, T) -> float:
    r"""Self-diffusion coefficient :math:`D = \tfrac13\langle v\rangle\lambda` (m^2/s).

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
        The diffusion coefficient.
    """
    return (1.0 / 3.0) * mean_speed(m, T) * mean_free_path(n, d)


def viscosity(n, d, m, T) -> float:
    r"""Shear viscosity :math:`\eta = \tfrac13 n m \langle v\rangle\lambda` (Pa s).

    Because :math:`\lambda \propto 1/n`, the viscosity is independent of density
    (Maxwell's celebrated prediction) and rises as :math:`\sqrt T`.
    """
    return (1.0 / 3.0) * n * m * mean_speed(m, T) * mean_free_path(n, d)


def thermal_conductivity(n, d, m, T, cv_per_molecule=None) -> float:
    r"""Thermal conductivity :math:`\kappa = \tfrac13 n \langle v\rangle\lambda\, c_v`.

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
    cv_per_molecule : float, optional
        Heat capacity per molecule.  Defaults to the monatomic value
        :math:`\tfrac32 k_B`.

    Returns
    -------
    float
        The thermal conductivity in W/(m K).
    """
    if cv_per_molecule is None:
        cv_per_molecule = 1.5 * k_B
    return ((1.0 / 3.0) * n * mean_speed(m, T)
            * mean_free_path(n, d) * cv_per_molecule)


def mean_free_path_from_pressure(p, d, T) -> float:
    r"""Mean free path from pressure, :math:`\lambda = k_B T / (\sqrt2\, \pi d^2 p)`.

    A convenience wrapper that first converts pressure to a number density with
    the ideal-gas law.
    """
    return mean_free_path(number_density(p, T), d)


__all__ = [
    "diffusion_coefficient",
    "viscosity",
    "thermal_conductivity",
    "mean_free_path_from_pressure",
]
