"""General-purpose numerical helpers shared across the package.

These small utilities capture patterns that recur in almost every example in the
book: creating a reproducible random generator, measuring the relative error
between a computed number and a reference, and fitting the empirical convergence
order of an error that is expected to fall as a power law.

The name :func:`trapezoid` is provided as a version-independent alias for the
trapezoidal integrator, which NumPy renamed from ``np.trapz`` to
``np.trapezoid`` in NumPy 2.0.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# NumPy 2.0 renamed np.trapz -> np.trapezoid; bind whichever exists.
trapezoid: Callable = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def get_rng(seed: int = 20260723) -> np.random.Generator:
    """Return a seeded NumPy random generator for reproducible Monte Carlo.

    Parameters
    ----------
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`.  The default matches the
        seed used throughout the book so that library-based reruns reproduce
        the printed results.

    Returns
    -------
    numpy.random.Generator
        A fresh, independently seeded generator.
    """
    return np.random.default_rng(seed)


def relative_error(approx, exact) -> np.ndarray:
    """Return the elementwise relative error ``|approx - exact| / |exact|``.

    Parameters
    ----------
    approx : array_like
        Computed value(s).
    exact : array_like
        Reference value(s); must be non-zero.

    Returns
    -------
    numpy.ndarray
        The relative error, broadcast over the inputs.
    """
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    return np.abs(approx - exact) / np.abs(exact)


def max_relative_error(approx, exact) -> float:
    """Return the largest relative error over all elements.

    See Also
    --------
    relative_error
    """
    return float(np.max(relative_error(approx, exact)))


def assert_close(approx, exact, rtol: float = 1e-6, atol: float = 0.0,
                 label: str = "") -> None:
    """Raise :class:`AssertionError` unless ``approx`` matches ``exact``.

    A thin wrapper around :func:`numpy.allclose` that produces an informative
    message, intended for the verification blocks of the examples and tests.

    Parameters
    ----------
    approx, exact : array_like
        Values to compare.
    rtol, atol : float, optional
        Relative and absolute tolerances passed to :func:`numpy.allclose`.
    label : str, optional
        Human-readable name of the quantity being checked.

    Raises
    ------
    AssertionError
        If the values do not agree within tolerance.
    """
    if not np.allclose(approx, exact, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label or 'value'} mismatch: approx={approx!r}, "
            f"exact={exact!r} (rtol={rtol}, atol={atol})"
        )


def convergence_order(sizes, errors) -> float:
    """Fit the power-law order of an error decay ``error ~ size**p``.

    The order ``p`` is the slope of ``log(error)`` versus ``log(size)``.  For a
    Monte Carlo estimator the expected value is ``-0.5``; for a second-order
    numerical scheme it is ``+2`` when ``size`` is a step size.

    Parameters
    ----------
    sizes : array_like
        Independent variable (sample counts, step sizes, system sizes, ...).
    errors : array_like
        Corresponding positive error magnitudes.

    Returns
    -------
    float
        The fitted exponent ``p``.
    """
    sizes = np.asarray(sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    return float(np.polyfit(np.log(sizes), np.log(errors), 1)[0])


__all__ = [
    "trapezoid",
    "get_rng",
    "relative_error",
    "max_relative_error",
    "assert_close",
    "convergence_order",
]
