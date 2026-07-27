"""Lightweight CFD interface.

CFD solvers need a small, stable set of transport and thermodynamic scalars
evaluated repeatedly. This module provides:

* :class:`CFDScalars` — a plain-attribute snapshot of the CFD-relevant
  quantities for a fixed state (zero per-access overhead after construction).
* :func:`bulk_properties` — vectorized evaluation over many states, returning a
  :class:`pandas.DataFrame` for transient post-processing.
* :func:`evaluate_grid` — evaluate over arrays of ``(T, P)`` for a fluid.

The underlying :class:`~thermolab.state.State` already caches every property, so
repeated reads of the same state are cheap; this module is a convenience layer
that picks the CFD-relevant subset and offers bulk evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CFDScalars:
    """The CFD-relevant property subset for a single state."""

    T: float
    P: float
    rho: float
    cp: float
    cv: float
    gamma: float
    mu: float
    k: float
    sound_speed: float
    Pr: float
    alpha: float
    phase: str

    @classmethod
    def from_state(cls, state) -> "CFDScalars":
        return cls(
            T=state.T,
            P=state.P,
            rho=state.rho,
            cp=state.cp,
            cv=state.cv,
            gamma=state.gamma,
            mu=state.mu,
            k=state.k,
            sound_speed=state.sound_speed,
            Pr=state.prandtl,
            alpha=state.thermal_diffusivity,
            phase=state.phase,
        )

    def to_dict(self) -> dict:
        return {
            "T": self.T, "P": self.P, "rho": self.rho, "cp": self.cp,
            "cv": self.cv, "gamma": self.gamma, "mu": self.mu, "k": self.k,
            "sound_speed": self.sound_speed, "Pr": self.Pr, "alpha": self.alpha,
            "phase": self.phase,
        }


def bulk_properties(states: Iterable) -> "pandas.DataFrame":
    """Evaluate the CFD scalar set for many states; return a DataFrame."""
    import pandas as pd

    rows = []
    for st in states:
        try:
            rows.append(CFDScalars.from_state(st).to_dict())
        except Exception:
            rows.append({k: np.nan for k in (
                "T", "P", "rho", "cp", "cv", "gamma", "mu", "k",
                "sound_speed", "Pr", "alpha", "phase")})
    return pd.DataFrame(rows)


def evaluate_grid(fluid, T: np.ndarray, P: np.ndarray, *, phase: str | None = None):
    """Evaluate CFD scalars over ``(T, P)`` arrays (same shape) for a fluid.

    Returns a DataFrame with one row per flattened (T, P) pair.
    """
    import pandas as pd

    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    if T.shape != P.shape:
        raise ValueError("T and P must have the same shape.")
    states = []
    for t, p in zip(T.ravel(), P.ravel()):
        try:
            states.append(fluid.state(T=float(t), P=float(p), phase=phase)
                          if phase else fluid.state(T=float(t), P=float(p)))
        except Exception:
            states.append(None)
    rows = []
    for st in states:
        if st is None:
            rows.append({k: np.nan for k in (
                "T", "P", "rho", "cp", "cv", "gamma", "mu", "k",
                "sound_speed", "Pr", "alpha", "phase")})
        else:
            rows.append(CFDScalars.from_state(st).to_dict())
    return pd.DataFrame(rows)