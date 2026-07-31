"""Computational backends.

The :mod:`backend` package abstracts the numerical array operations used by the modes so that an
acceleration backend (Numba ``@njit``, OpenMP-parallel Numba, CUDA) can be plugged in *without
changing the public API*.

The NumPy backend is always available. The accelerated backends are imported lazily when
selected by name, so ``import statthermopy`` never pulls in numba:

>>> from statthermopy.backend import set_backend, available_backends
>>> available_backends()
['numpy', 'numba', 'openmp']
>>> set_backend("numba")          # if numba is installed

CUDA falls back to the Numba CPU backend automatically when no NVIDIA GPU is present.
"""

from __future__ import annotations

from .executor import (
    Backend,
    NumpyBackend,
    available_backends,
    get_backend,
    list_backends,
    set_backend,
)

__all__ = [
    "Backend",
    "NumpyBackend",
    "get_backend",
    "set_backend",
    "list_backends",
    "available_backends",
]