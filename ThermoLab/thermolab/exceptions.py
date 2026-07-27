"""Exception hierarchy for ThermoLab.

All ThermoLab-specific errors derive from :class:`ThermoLabError` so callers can
catch the whole family with a single ``except`` clause.
"""

from __future__ import annotations


class ThermoLabError(Exception):
    """Base class for all ThermoLab errors."""


class UnsupportedFluidError(ThermoLabError):
    """Raised when a fluid/component is not available in the active backend."""

    def __init__(self, fluid: str, backend: str, detail: str = ""):
        self.fluid = fluid
        self.backend = backend
        msg = (
            f"Fluid/component {fluid!r} is not available in the {backend!r} "
            f"backend."
        )
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class BackendError(ThermoLabError):
    """Raised when a backend cannot be created or is misconfigured."""


class BackendNotAvailableError(BackendError):
    """Raised when a requested backend package is not installed."""


class ConvergenceError(ThermoLabError):
    """Raised when a flash / root-finding computation fails to converge."""

    def __init__(self, message: str, *, last_value: float | None = None):
        self.last_value = last_value
        super().__init__(message)


class TwoPhaseError(ThermoLabError):
    """Raised when a single-phase property is requested in a two-phase state."""


class FlashSpecificationError(ThermoLabError):
    """Raised when a state specification is invalid (missing/over-specified pair)."""


class UnsupportedPropertyError(ThermoLabError):
    """Raised when a backend cannot compute a requested property."""


class FluidAliasError(ThermoLabError):
    """Raised when a fluid alias cannot be resolved."""