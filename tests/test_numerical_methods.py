"""Tests for Monte Carlo sampling and error analysis."""

import numpy as np
from scipy.integrate import quad

from statistical_thermodynamics import numerical_methods as nm
from statistical_thermodynamics.utilities import get_rng


def _anharmonic(x):
    return 0.5 * x ** 2 + 0.1 * x ** 4


def _exact_x2():
    num, _ = quad(lambda x: x ** 2 * np.exp(-_anharmonic(x)), -np.inf, np.inf)
    den, _ = quad(lambda x: np.exp(-_anharmonic(x)), -np.inf, np.inf)
    return num / den


def test_metropolis_samples_boltzmann_mean():
    rng = get_rng(20260723)
    traj, acc = nm.metropolis_1d(_anharmonic, 200_000, step=3.0, rng=rng)
    x2 = np.mean(traj ** 2)
    assert np.isclose(x2, _exact_x2(), atol=0.02)
    assert 0.2 < acc < 0.8


def test_autocorrelation_time_at_least_half():
    rng = get_rng(1)
    traj, _ = nm.metropolis_1d(_anharmonic, 40_000, step=3.0, rng=rng)
    tau = nm.autocorrelation_time(traj ** 2)
    assert tau >= 0.5


def test_autocorrelation_of_white_noise_is_half():
    rng = get_rng(7)
    white = rng.normal(size=50_000)
    tau = nm.autocorrelation_time(white)
    assert abs(tau - 0.5) < 0.6


def test_blocking_error_exceeds_naive():
    rng = get_rng(3)
    traj, _ = nm.metropolis_1d(_anharmonic, 100_000, step=3.0, rng=rng)
    obs = traj ** 2
    naive = obs.std(ddof=1) / np.sqrt(len(obs))
    plateau = nm.blocking_error(obs)
    # Correlated data: the honest (blocked) error is larger than the naive one.
    assert plateau > naive


def test_bootstrap_matches_analytic_sem():
    rng = get_rng(11)
    samples = rng.normal(0.0, 1.0, 5000)
    analytic = samples.std(ddof=1) / np.sqrt(len(samples))
    boot = nm.bootstrap_error(samples, n_boot=1000, rng=get_rng(12))
    assert np.isclose(boot, analytic, rtol=0.15)


def test_importance_sampling_estimate():
    rng = get_rng(20260723)

    def sampler(n, r):
        return r.normal(0.0, 1.0, n)

    def gaussian_pdf(x):
        return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

    est = nm.importance_sampling_mean(
        observable=lambda x: x ** 2,
        target=lambda x: np.exp(-_anharmonic(x)),
        proposal_sampler=sampler,
        proposal_pdf=gaussian_pdf,
        n=200_000,
        rng=rng,
    )
    assert np.isclose(est, _exact_x2(), atol=0.01)
