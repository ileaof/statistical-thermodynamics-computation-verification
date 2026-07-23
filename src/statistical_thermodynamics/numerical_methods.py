"""Monte Carlo sampling and honest error analysis.

The heart of Chapter 10: a one-dimensional Metropolis sampler, the integrated
autocorrelation time, Flyvbjerg-Petersen blocking, and bootstrap resampling.
Together they turn a correlated Markov chain into a value with a trustworthy
error bar.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def metropolis_1d(potential: Callable, n_steps: int, step: float,
                  rng: np.random.Generator, beta: float = 1.0,
                  x0: float = 0.0, burn: int = 5000) -> Tuple[np.ndarray, float]:
    r"""Sample ``p(x) ~ exp(-beta V(x))`` with the Metropolis algorithm.

    A trial move :math:`x \to x + \delta`, with :math:`\delta` uniform on
    :math:`[-\text{step}, \text{step}]`, is accepted with probability
    :math:`\min(1, e^{-\beta[V(x')-V(x)]})`.

    Parameters
    ----------
    potential : callable
        The potential energy ``V(x)`` (vectorisation not required).
    n_steps : int
        Number of samples to return (after burn-in).
    step : float
        Maximum trial-move size.
    rng : numpy.random.Generator
        Seeded random generator (see :func:`~statistical_thermodynamics.utilities.get_rng`).
    beta : float, optional
        Inverse temperature :math:`1/k_B T`.
    x0 : float, optional
        Starting position.
    burn : int, optional
        Number of burn-in steps discarded before recording.

    Returns
    -------
    trajectory : numpy.ndarray
        The sampled positions, length ``n_steps``.
    acceptance : float
        The fraction of accepted moves during the recorded phase.
    """
    x = x0
    Vx = potential(x)
    traj = np.empty(n_steps)
    n_acc = 0
    for i in range(n_steps + burn):
        xt = x + rng.uniform(-step, step)
        Vt = potential(xt)
        if Vt < Vx or rng.random() < np.exp(-beta * (Vt - Vx)):
            x, Vx = xt, Vt
            if i >= burn:
                n_acc += 1
        if i >= burn:
            traj[i - burn] = x
    return traj, n_acc / n_steps


def autocorrelation_time(series, cutoff: int = 200) -> float:
    r"""Integrated autocorrelation time :math:`\tau` of a time series.

    Computes :math:`\tau = \tfrac12 + \sum_{k\ge 1}\rho(k)`, truncating the sum
    at the first non-positive autocorrelation :math:`\rho(k)`.  The effective
    number of independent samples is :math:`N/(2\tau)`.

    Parameters
    ----------
    series : array_like
        The (scalar) observable time series.
    cutoff : int, optional
        Maximum lag considered.

    Returns
    -------
    float
        The integrated autocorrelation time in units of samples.
    """
    a = np.asarray(series, float)
    a = a - a.mean()
    n = len(a)
    var = np.dot(a, a) / n
    if var == 0.0:
        return 0.5
    tau = 0.5
    for k in range(1, min(cutoff, n)):
        c = np.dot(a[:-k], a[k:]) / (n - k) / var
        if c <= 0:
            break
        tau += c
    return float(tau)


def blocking_errors(series) -> Tuple[np.ndarray, np.ndarray]:
    r"""Flyvbjerg-Petersen blocking analysis of the standard error.

    Repeatedly averages adjacent pairs of samples; the naive standard error at
    each blocking level rises and then plateaus at the true, autocorrelation-
    corrected value once blocks exceed the correlation time.

    Parameters
    ----------
    series : array_like
        The observable time series.

    Returns
    -------
    levels : numpy.ndarray
        The blocking-transformation index at each level.
    errors : numpy.ndarray
        The estimated standard error of the mean at each level.
    """
    a = np.asarray(series, float).copy()
    levels, errs = [], []
    lvl = 0
    while len(a) >= 8:
        errs.append(a.std(ddof=1) / np.sqrt(len(a)))
        levels.append(lvl)
        if len(a) % 2:
            a = a[:-1]
        a = 0.5 * (a[0::2] + a[1::2])
        lvl += 1
    return np.array(levels), np.array(errs)


def blocking_error(series) -> float:
    """Return the plateau (autocorrelation-corrected) standard error from blocking.

    Convenience wrapper around :func:`blocking_errors` that returns the maximum
    of the last few blocking levels, a robust estimate of the plateau value.
    """
    _, errs = blocking_errors(series)
    return float(errs[-6:].max()) if len(errs) else float("nan")


def bootstrap_error(samples, n_boot: int = 2000,
                    rng: np.random.Generator = None) -> float:
    r"""Bootstrap estimate of the standard error of the mean.

    Resamples ``samples`` with replacement ``n_boot`` times and returns the
    standard deviation of the resampled means.  For a correlated chain, pass
    *decorrelated block means* rather than the raw series.

    Parameters
    ----------
    samples : array_like
        (Ideally decorrelated) samples.
    n_boot : int, optional
        Number of bootstrap resamples.
    rng : numpy.random.Generator, optional
        Random generator; a default-seeded one is created if omitted.

    Returns
    -------
    float
        The bootstrap standard error.
    """
    if rng is None:
        rng = np.random.default_rng(20260723)
    samples = np.asarray(samples, float)
    n = len(samples)
    means = np.array([np.mean(rng.choice(samples, n, replace=True))
                      for _ in range(n_boot)])
    return float(means.std())


def importance_sampling_mean(observable: Callable, target: Callable,
                             proposal_sampler: Callable, proposal_pdf: Callable,
                             n: int, rng: np.random.Generator) -> float:
    r"""Estimate :math:`\langle O\rangle` under ``target`` via importance sampling.

    .. math::

        \langle O\rangle \approx
          \frac{\sum_i O(x_i)\, w_i}{\sum_i w_i},\qquad
          w_i = \frac{\text{target}(x_i)}{\text{proposal}(x_i)},

    with :math:`x_i` drawn from the proposal distribution.

    Parameters
    ----------
    observable : callable
        The quantity ``O(x)`` to average.
    target : callable
        The (possibly unnormalised) target density, e.g.
        ``lambda x: exp(-beta V(x))``.
    proposal_sampler : callable
        ``proposal_sampler(n, rng)`` returning ``n`` draws.
    proposal_pdf : callable
        The proposal probability density ``g(x)``.
    n : int
        Number of samples.
    rng : numpy.random.Generator
        Seeded random generator.

    Returns
    -------
    float
        The importance-sampling estimate.
    """
    x = proposal_sampler(n, rng)
    w = target(x) / proposal_pdf(x)
    return float(np.sum(observable(x) * w) / np.sum(w))


__all__ = [
    "metropolis_1d",
    "autocorrelation_time",
    "blocking_errors",
    "blocking_error",
    "bootstrap_error",
    "importance_sampling_mean",
]
