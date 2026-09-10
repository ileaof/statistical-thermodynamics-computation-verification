"""Anharmonic vibrational contribution — an explicit sum over the real level manifold.

The harmonic oscillator (:mod:`~statthermopy.modes.vibrational`) spaces every level of a mode
equally. A real potential widens with energy, so the true levels *close up*: the harmonic model
under-populates the high-lying states and therefore under-predicts ``Cv`` as temperature rises.
For water the deficit reaches ``-1.9 %`` in ``Cp`` at 2000 K (see ``docs/H2O_AUDIT.md``).

This mode removes that approximation by building the level manifold from the second-order
(Dunham) term value

    G(v_1, ..., v_n) = Σ_i ω_i (v_i + 1/2) + Σ_{i ≤ j} x_ij (v_i + 1/2)(v_j + 1/2)

and summing over it exactly, exactly as :class:`~statthermopy.modes.electronic.Electronic` sums
over electronic terms:

    Q_v = Σ_k exp(-θ_k / T),    θ_k = h c [G(v_k) - G(0)] / k_B
    U_m  = R ⟨θ⟩
    Cv_m = R (⟨θ²⟩ - ⟨θ⟩²) / T²
    S_m  = R [ ln Q_v + ⟨θ⟩ / T ]
    A_m  = -R T ln Q_v

The energy zero is the ground vibrational level ``G(0)``, matching the harmonic mode's convention,
so the zero-point energy is excluded and ``H_m`` remains the JANAF-style increment ``H(T) - H(0)``.

**This is still pure statistical mechanics.** The ω_i and x_ij are *spectroscopic* constants read
from vibrational spectra, on the same footing as the rotational constants — not empirical property
correlations. The constant set is self-validating: it must reproduce the observed fundamentals,
which :meth:`~statthermopy.core.molecule.Anharmonicity.fundamentals_cm1` checks.

**Truncation.** The second-order expansion is asymptotic: beyond some ``v`` the quadratic term
turns the spacing negative and the series stops meaning anything. Two guards apply — the manifold
is cut at ``dissociation_cm1``, and any level reached by *lowering* the energy (a turned-over mode)
is rejected.
"""

from __future__ import annotations

import functools
import math

from ..constants import R
from ..core.contribution import Contribution
from ..core.molecule import Anharmonicity
from ..core.state import ResolvedState
from ..units import CM1_TO_K
from .base import Mode

__all__ = ["AnharmonicVibrational"]

#: Safety factor on the harmonic energy used to prune the level search.
_PRUNE_FACTOR = 2.0
#: Hard cap on the enumerated manifold, to fail loudly rather than hang on a large molecule.
_MAX_LEVELS = 2_000_000


class AnharmonicVibrational(Mode):
    """Vibrational mode summed over the anharmonic level manifold.

    Parameters
    ----------
    anharmonicity : Anharmonicity
        Harmonic wavenumbers, the x_ij matrix and the truncation energy.
    """

    name = "vibrational"

    def __init__(self, anharmonicity: Anharmonicity) -> None:
        self.anharmonicity = anharmonicity
        self.theta: tuple[float, ...] = _level_temperatures(anharmonicity)

    # -- level statistics ------------------------------------------------------

    def _moments(self, T: float) -> tuple[float, float, float]:
        """Return ``(Q, <theta>, <theta^2>)`` at temperature ``T``."""
        q = 0.0
        s1 = 0.0
        s2 = 0.0
        for th in self.theta:
            x = th / T
            if x > 700.0:  # exp underflows to 0; every later level is colder still
                continue
            w = math.exp(-x)
            q += w
            s1 += w * th
            s2 += w * th * th
        if q == 0.0:  # pragma: no cover - only reachable if the ground level were dropped
            return 1.0, 0.0, 0.0
        return q, s1 / q, s2 / q

    # -- partition function ---------------------------------------------------

    def ln_q(self, state: ResolvedState) -> float:
        if not self.theta or state.T == 0.0:
            return 0.0
        return math.log(self._moments(state.T)[0])

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        """``d ln Q / dT = <theta> / T^2``."""
        if not self.theta or state.T == 0.0:
            return 0.0
        _, mean, _ = self._moments(state.T)
        return mean / (state.T * state.T)

    def cv_m(self, state: ResolvedState) -> float:
        if not self.theta or state.T == 0.0:
            return 0.0
        _, mean, mean2 = self._moments(state.T)
        return R * (mean2 - mean * mean) / (state.T * state.T)

    # -- contribution ---------------------------------------------------------

    def contribution(self, state: ResolvedState) -> Contribution:
        if not self.theta or state.T == 0.0:
            return Contribution(name=self.name, ln_q=0.0, d_ln_q_dT=0.0,
                                 U_m=0.0, S_m=0.0, A_m=0.0, Cv_m=0.0)
        T = state.T
        q, mean, mean2 = self._moments(T)
        lnq = math.log(q)
        return Contribution(
            name=self.name,
            ln_q=lnq,
            d_ln_q_dT=mean / (T * T),
            U_m=R * mean,
            S_m=R * (lnq + mean / T),
            A_m=-R * T * lnq,
            Cv_m=R * (mean2 - mean * mean) / (T * T),
        )


# --- level enumeration --------------------------------------------------------


@functools.lru_cache(maxsize=64)
def _level_temperatures(anh: Anharmonicity) -> tuple[float, ...]:
    """Characteristic temperatures of the retained manifold, cached per constant set.

    The manifold depends only on the (frozen, hashable) :class:`Anharmonicity` constants, but
    building it walks thousands of quantum-number combinations. A mode object is constructed on
    every ``Thermodynamics(...)`` call, so without this cache the enumeration dominates the cost
    of any repeated evaluation — it measured 67 % of a 30-component mixture point.
    """
    return tuple(e * CM1_TO_K for e in _enumerate_levels(anh))


def _enumerate_levels(anh: Anharmonicity) -> list[float]:
    """Build the retained level energies (cm⁻¹) measured from the vibrational ground state.

    A depth-first walk over the quantum numbers, pruned on the harmonic partial energy, then
    filtered on the exact term value and on local monotonicity.
    """
    n = anh.n_modes
    omega = anh.harmonic_wavenumbers_cm1
    limit = anh.dissociation_cm1
    zero = anh.zero_point_energy_cm1

    # Per-mode cap: the smaller of the dissociation limit and the turn-over of that mode alone.
    caps: list[int] = []
    for i in range(n):
        by_energy = int(limit / omega[i]) + 1
        x_ii = anh.x_matrix_cm1[i][i]
        by_turnover = int(omega[i] / (2.0 * abs(x_ii))) if x_ii < 0 else by_energy
        caps.append(max(1, min(by_energy, by_turnover)))

    total_box = 1
    for c in caps:
        total_box *= c + 1
        if total_box > _MAX_LEVELS:
            raise ValueError(
                f"Anharmonic manifold would exceed {_MAX_LEVELS} levels "
                f"({n} modes, caps {caps}); lower dissociation_cm1 or use the harmonic mode."
            )

    energies: list[float] = []
    v = [0] * n

    def walk(i: int, harmonic_partial: float) -> None:
        if i == n:
            e = anh.term_value_cm1(tuple(v)) - zero
            if 0.0 <= e <= limit and _is_monotonic(anh, v, zero, e):
                energies.append(e)
            return
        for vi in range(caps[i] + 1):
            partial = harmonic_partial + omega[i] * vi
            # Anharmonic terms lower the energy, so prune generously rather than exactly.
            if partial > _PRUNE_FACTOR * limit:
                break
            v[i] = vi
            walk(i + 1, partial)
        v[i] = 0

    walk(0, 0.0)
    energies.sort()
    return energies


def _is_monotonic(anh: Anharmonicity, v: list[int], zero: float, e: float) -> bool:
    """Reject a level that a mode reached by *lowering* the energy (the expansion turned over)."""
    for i, vi in enumerate(v):
        if vi == 0:
            continue
        lower = list(v)
        lower[i] = vi - 1
        if anh.term_value_cm1(tuple(lower)) - zero >= e:
            return False
    return True
