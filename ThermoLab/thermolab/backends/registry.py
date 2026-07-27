"""Backend registry and factory.

Lets additional backends register themselves so the public API can resolve a
backend by name without importing optional dependencies eagerly.
"""

from __future__ import annotations

from typing import Callable

from ..exceptions import BackendNotAvailableError, BackendError, ThermoLabError
from .base import BaseBackend

# name -> factory(components, reference_state=, eos=) -> BaseBackend
_REGISTRY: dict[str, Callable[..., BaseBackend]] = {}
_AVAILABLE: dict[str, bool] = {}


def register_backend(
    name: str, factory: Callable[..., BaseBackend], available: bool = True
) -> None:
    _REGISTRY[name.lower()] = factory
    _AVAILABLE[name.lower()] = available


def available_backends() -> list[str]:
    """Return the names of registered backends that are importable."""
    return sorted(n for n, ok in _AVAILABLE.items() if ok)


def is_available(name: str) -> bool:
    return _AVAILABLE.get(name.lower(), False)


def get_backend(
    name: str,
    components: list[str],
    *,
    reference_state: str = "DEFAULT",
    eos: str | None = None,
) -> BaseBackend:
    """Construct a backend by name.

    Raises :class:`BackendNotAvailableError` if the name is unknown or the
    backing package is not installed.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise BackendNotAvailableError(
            f"Unknown backend {name!r}. Registered: {available_backends() or 'none'}."
        )
    if not _AVAILABLE.get(key, False):
        raise BackendNotAvailableError(
            f"Backend {name!r} is registered but its package is not installed."
        )
    try:
        return _REGISTRY[key](components, reference_state=reference_state, eos=eos)
    except (BackendError, ThermoLabError):
        raise
    except Exception as exc:  # construction errors -> BackendError
        raise BackendError(f"Failed to create {name!r} backend: {exc}") from exc


# ---------------------------------------------------------------------------
# Register the built-in ThermoPack backend.
# ---------------------------------------------------------------------------
def _register_builtin() -> None:
    from .thermopack_backend import ThermoPackBackend

    # Detect whether thermopack is importable.
    try:
        import thermopack  # noqa: F401
        available = True
    except Exception:
        available = False

    register_backend(
        "thermopack",
        lambda components, reference_state="DEFAULT", eos=None: ThermoPackBackend(
            components, reference_state=reference_state, eos=eos
        ),
        available=available,
    )


_register_builtin()