"""Tests for combinatorics and elementary probability."""

import numpy as np
from scipy.special import comb

from statistical_thermodynamics import probability as pr


def test_log_binomial_matches_scipy():
    N, n = 40, 17
    assert np.isclose(np.exp(pr.log_binomial(N, n)), comb(N, n), rtol=1e-10)


def test_two_state_multiplicity_sums_to_2N():
    N = 20
    n = np.arange(N + 1)
    total = pr.two_state_multiplicity(N, n).sum()
    assert np.isclose(total, 2.0 ** N, rtol=1e-8)


def test_gaussian_multiplicity_peaks_at_center():
    N = 200
    n = np.arange(N + 1)
    g = pr.gaussian_multiplicity(N, n)
    assert n[np.argmax(g)] == N // 2


def test_einstein_multiplicity_vandermonde():
    # sum_q C(q+N-1,q) weighting reproduces the composite count (spot check).
    N = 30
    val = np.exp(pr.einstein_log_multiplicity(N, 10))
    assert np.isclose(val, comb(10 + N - 1, 10), rtol=1e-8)


def test_boltzmann_probabilities_normalised_and_ordered():
    energies = np.array([0.0, 1.0, 2.0, 3.0])
    p = pr.boltzmann_probabilities(energies, T=1.0)
    assert np.isclose(p.sum(), 1.0, rtol=1e-12)
    # Lower energies must be more probable.
    assert np.all(np.diff(p) < 0)


def test_shannon_entropy_uniform_is_log_n():
    p = np.full(8, 1 / 8)
    assert np.isclose(pr.shannon_entropy(p), np.log(8))


def test_shannon_entropy_handles_zeros():
    p = np.array([1.0, 0.0, 0.0])
    assert np.isclose(pr.shannon_entropy(p), 0.0)
