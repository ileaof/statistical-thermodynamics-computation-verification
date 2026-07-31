"""Pluggable numerical backend.

The engine performs its array work (exponentials, logs, sums over vibrational/electronic
levels) through a thin :class:`Backend` interface, and the heavier hot loops — the quantum
rotational J-sum and the temperature-batched property grid — through optional high-level
kernels. Phase 1 shipped only :class:`NumpyBackend`. Phase 3 adds functional accelerated
backends:

* :class:`NumbaBackend` (``"numba"``)  — ``@njit`` CPU kernels;
* :class:`OpenMPBackend` (``"openmp"``) — Numba ``@njit(parallel=True)`` with ``prange``;
* :class:`CudaBackend`  (``"cuda"``)   — ``numba.cuda`` GPU kernel, with automatic CPU fallback
  when no NVIDIA GPU is present.

A backend is selected by name with :func:`set_backend` (e.g. ``set_backend("numba")``) and the
active one is read with :func:`get_backend`. The accelerated backends are imported lazily, so
``import statthermopy`` never pulls in numba. The six array methods and the public API are
unchanged; the high-level kernels are concrete methods that return ``None`` by default, so the
NumPy path (and any user subclass of :class:`Backend`) keeps its existing behaviour.

The interface intentionally exposes only the handful of array functions the modes need, plus two
opt-in kernels, keeping the surface small and stable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = [
    "Backend",
    "NumpyBackend",
    "get_backend",
    "set_backend",
    "list_backends",
    "available_backends",
]

#: All backends declared by the package (architecture), irrespective of whether their
#: optional dependencies are importable in the current environment.
_DECLARED_BACKENDS: tuple[str, ...] = ("numpy", "numba", "openmp", "cuda")


class Backend(ABC):
    """Abstract array backend used by the partition-function modes.

    The six array methods are abstract (every backend must provide them). The two high-level
    kernels — :meth:`linear_quantum_moments` and :meth:`molar_property_grid` — are *concrete*
    and return ``None`` by default, signalling "use the existing Python path". Accelerated
    backends override them; :class:`NumpyBackend` and any user subclass inherit the default and
    therefore keep the original behaviour unchanged.
    """

    name: str = "abstract"

    @abstractmethod
    def exp(self, x: Any) -> Any: ...
    @abstractmethod
    def expm1(self, x: Any) -> Any: ...
    @abstractmethod
    def log(self, x: Any) -> Any: ...
    @abstractmethod
    def log1p(self, x: Any) -> Any: ...
    @abstractmethod
    def sum(self, x: Any) -> float: ...
    @abstractmethod
    def asarray(self, x: Any) -> Any: ...

    # -- opt-in high-level kernels (default: defer to the Python path) -------------------

    def linear_quantum_moments(self, theta_rot: float, T: float, cutoff: int):
        """Quantum linear-rotor J-sum, or ``None`` to defer to the mode's Python loop.

        When provided, returns ``(Q, <y>, <y^2>)`` with ``y = J(J+1) theta_rot / T`` for the
        exact quantum rigid-rotor partition function (see :class:`~statthermopy.modes.rotational`
        .Rotational`). Returning ``None`` keeps the existing pure-Python loop.
        """
        return None

    def molar_property_grid(self, mol, T_array, P: float, use_quantum: bool, cutoff: int = 150):
        """Molar properties over a temperature grid, or ``None`` to defer.

        When provided, returns a mapping with the eight core molar arrays ``"U_m"``, ``"S_m"``,
        ``"A_m"``, ``"Cv_m"`` and the log factors ``"ln_Qt"``, ``"ln_Qr"``, ``"ln_Qv"``,
        ``"ln_Qe"`` — each a 1-D array aligned with ``T_array``. :meth:`Thermodynamics.
        property_vs_T <statthermopy.thermodynamics.Thermodynamics.property_vs_T>` derives the
        remaining attributes (H, G, Cp, gamma, massic, partition Q-values) from these, identically
        to the per-T Python path. Returning ``None`` selects that Python path.

        ``cutoff`` is the maximum ``J`` for the quantum linear-rotor sum (default 150, matching
        :class:`~statthermopy.modes.rotational.Rotational`).
        """
        return None


class NumpyBackend(Backend):
    """Reference NumPy backend."""

    name = "numpy"

    def exp(self, x):
        return np.exp(x)

    def expm1(self, x):
        return np.expm1(x)

    def log(self, x):
        return np.log(x)

    def log1p(self, x):
        return np.log1p(x)

    def sum(self, x):
        return float(np.sum(x))

    def asarray(self, x):
        return np.asarray(x, dtype=float)


# --- backend registry -------------------------------------------------------

_DEFAULT = NumpyBackend()
_BACKEND_CACHE: dict[str, Backend] = {"numpy": _DEFAULT}


def _build_backend(name: str) -> Backend:
    """Construct a backend by name, importing its module lazily."""
    name = name.lower()
    if name == "numpy":
        return NumpyBackend()
    try:
        if name == "numba":
            from .numba_backend import NumbaBackend

            return NumbaBackend()
        if name == "openmp":
            from .openmp_backend import OpenMPBackend

            return OpenMPBackend()
        if name == "cuda":
            from .cuda_backend import CudaBackend

            return CudaBackend()
    except ImportError as exc:  # pragma: no cover - exercised via test for the message
        raise ImportError(
            f"Backend {name!r} requires the optional 'accel' extra "
            f"(pip install statthermopy[accel]). Original error: {exc}"
        ) from exc
    raise ValueError(
        f"Unknown backend {name!r}. Known: {', '.join(_DECLARED_BACKENDS)}."
    )


def get_backend(name: str | None = None) -> Backend:
    """Return a backend.

    With ``name=None`` returns the currently active backend. With a name (``"numpy"``,
    ``"numba"``, ``"openmp"``, ``"cuda"``) returns a cached instance of that backend, building it
    (and lazily importing its dependencies) on first use.
    """
    if name is None:
        return _DEFAULT
    if not isinstance(name, str):
        raise TypeError(f"Backend name must be a string, got {type(name).__name__}.")
    name = name.lower()
    cached = _BACKEND_CACHE.get(name)
    if cached is None:
        cached = _build_backend(name)
        _BACKEND_CACHE[name] = cached
    return cached


def set_backend(name_or_backend) -> None:
    """Select the active backend, by name or by instance.

    Examples
    --------
    >>> set_backend("numba")          # name -> lazy-built, cached backend
    >>> set_backend(NumpyBackend())   # instance
    """
    global _DEFAULT
    if isinstance(name_or_backend, Backend):
        _DEFAULT = name_or_backend
    elif isinstance(name_or_backend, str):
        _DEFAULT = get_backend(name_or_backend)
    else:
        raise TypeError(
            "set_backend expects a backend name (str) or a Backend instance, "
            f"got {type(name_or_backend).__name__}."
        )


def list_backends() -> list[str]:
    """All backends declared by the package, whether or not their dependencies are installed."""
    return list(_DECLARED_BACKENDS)


def available_backends() -> list[str]:
    """Backends whose optional dependencies are importable in this environment.

    Always includes ``"numpy"``. ``"numba"``/``"openmp"`` require numba; ``"cuda"`` additionally
    requires a detected NVIDIA GPU (``numba.cuda.is_available()``).
    """
    out = ["numpy"]
    try:
        import numba  # noqa: F401

        out.append("numba")
        out.append("openmp")
    except ImportError:  # pragma: no cover - numba not installed
        return out
    try:
        import numba.cuda

        if numba.cuda.is_available():  # pragma: no cover - requires an NVIDIA GPU
            out.append("cuda")
    except Exception:  # pragma: no cover - numba.cuda absent or broken
        pass
    return out