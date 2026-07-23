"""Combinatorics, multiplicities and elementary probability distributions.

These are the counting tools behind Boltzmann's ``S = k_B ln Omega`` and the
canonical probabilities used from Chapter 1 onward.  All multiplicities are
computed through log-gamma arithmetic so that factorials never overflow.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def log_binomial(N, n) -> np.ndarray:
    r"""Return :math:`\ln \binom{N}{n}` using log-gamma (overflow-free).

    Parameters
    ----------
    N : array_like
        Number of elements.
    n : array_like
        Number chosen; broadcast against ``N``.

    Returns
    -------
    numpy.ndarray
        The natural logarithm of the binomial coefficient.
    """
    N = np.asarray(N, dtype=float)
    n = np.asarray(n, dtype=float)
    return gammaln(N + 1.0) - gammaln(n + 1.0) - gammaln(N - n + 1.0)


def two_state_multiplicity(N, n) -> np.ndarray:
    r"""Multiplicity :math:`\Omega(N, n) = \binom{N}{n}` of a two-state system.

    Parameters
    ----------
    N : int or array_like
        Number of two-state elements (e.g. spins).
    n : int or array_like
        Number of elements in the excited ("up") state.

    Returns
    -------
    numpy.ndarray
        The number of microstates of the macrostate.
    """
    return np.exp(log_binomial(N, n))


def gaussian_multiplicity(N, n) -> np.ndarray:
    r"""Gaussian (Stirling) approximation to :math:`\binom{N}{n}` about ``n = N/2``.

    Expanding :math:`\ln\binom{N}{n}` to second order in ``x = n - N/2`` gives a
    Gaussian of width :math:`\sqrt{N}/2`:

    .. math::

        \Omega(N, n) \approx 2^N \sqrt{\frac{2}{\pi N}}
                      \exp\!\left(-\frac{2 (n - N/2)^2}{N}\right).

    Parameters
    ----------
    N : int
        System size.
    n : array_like
        Number of up-states.

    Returns
    -------
    numpy.ndarray
        The Gaussian-approximated multiplicity.
    """
    N = float(N)
    x = np.asarray(n, dtype=float) - N / 2.0
    omega_max = 2.0 ** N * np.sqrt(2.0 / (np.pi * N))
    return omega_max * np.exp(-2.0 * x * x / N)


def einstein_log_multiplicity(N, q) -> np.ndarray:
    r"""Log multiplicity of an Einstein solid, :math:`\ln \binom{q + N - 1}{q}`.

    An Einstein solid of ``N`` oscillators holding ``q`` energy quanta has
    :math:`\Omega = \binom{q + N - 1}{q}` (a stars-and-bars count).

    Parameters
    ----------
    N : int
        Number of oscillators.
    q : array_like
        Number of energy quanta.

    Returns
    -------
    numpy.ndarray
        ``ln Omega(N, q)``.
    """
    q = np.asarray(q, dtype=float)
    return gammaln(q + N) - gammaln(q + 1.0) - gammaln(float(N))


def boltzmann_probabilities(energies, T, k_B: float = 1.0) -> np.ndarray:
    r"""Canonical (Boltzmann) probabilities for a set of energy levels.

    .. math:: p_i = e^{-E_i / k_B T} \big/ \sum_j e^{-E_j / k_B T}.

    The maximum energy is subtracted before exponentiating for numerical
    stability.

    Parameters
    ----------
    energies : array_like
        Level energies :math:`E_i`.
    T : float
        Temperature.
    k_B : float, optional
        Boltzmann constant in the caller's unit system (default ``1``).

    Returns
    -------
    numpy.ndarray
        Normalised probabilities that sum to one.
    """
    e = np.asarray(energies, dtype=float)
    w = np.exp(-(e - e.max()) / (k_B * T))
    return w / w.sum()


def shannon_entropy(p, k_B: float = 1.0) -> float:
    r"""Gibbs-Shannon entropy :math:`S = -k_B \sum_i p_i \ln p_i`.

    Zero-probability states are handled with the convention
    :math:`0 \ln 0 = 0`.

    Parameters
    ----------
    p : array_like
        A normalised probability distribution.
    k_B : float, optional
        Boltzmann constant (default ``1`` gives entropy in nats).

    Returns
    -------
    float
        The entropy.
    """
    p = np.asarray(p, dtype=float)
    nz = p > 0.0
    return float(-k_B * np.sum(p[nz] * np.log(p[nz])))


__all__ = [
    "log_binomial",
    "two_state_multiplicity",
    "gaussian_multiplicity",
    "einstein_log_multiplicity",
    "boltzmann_probabilities",
    "shannon_entropy",
]
