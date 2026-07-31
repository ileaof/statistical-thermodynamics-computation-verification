"""Rotational contribution to the molecular partition function.

Three geometries are supported:

* **Monoatomic** — no rotational degrees of freedom; ``Q_r = 1`` with zero contribution.
* **Linear** (diatomics and linear polyatomics) — rigid rotor. In the classical high-temperature
  limit ``Q_r = T / (σ θ_rot)`` with ``θ_rot = h^2 / (8 π^2 I k_B)``. An exact quantum sum
  ``Σ_J (2J+1) exp[-J(J+1) θ_rot / T]`` is available via ``use_quantum=True`` for low temperatures.
* **Nonlinear** — rigid asymmetric top with three principal moments of inertia
  ``I_A ≤ I_B ≤ I_C``:

      Q_r = (sqrt(π) / σ) * sqrt(T^3 / (θ_A θ_B θ_C)) .

The symmetry number ``σ`` counts the indistinguishable orientations produced by rotations.
"""

from __future__ import annotations

import math
from typing import Literal

from ..constants import R, h, k_B
from ..core.contribution import Contribution
from ..core.molecule import Geometry
from ..core.state import ResolvedState
from .base import Mode

__all__ = ["Rotational"]

GeometryKind = Literal["monoatomic", "linear", "nonlinear"]


def rotational_temperature(I: float) -> float:
    """Characteristic rotational temperature ``θ_rot = h^2 / (8 π^2 I k_B)`` (K)."""
    return (h * h) / (8.0 * math.pi * math.pi * I * k_B)


class Rotational(Mode):
    """Rotational mode built from a molecule's geometry, moments of inertia and symmetry.

    Parameters
    ----------
    geometry : Geometry
        Molecular geometry.
    symmetry_number : int
        Rotational symmetry number ``σ``.
    moments_of_inertia : tuple[float, ...]
        Principal moments in kg m^2 (empty / one / three).
    use_quantum : bool, default False
        If ``True`` and the molecule is linear, use the exact quantum rigid-rotor sum instead
        of the classical high-temperature formula. Ignored for monoatomic and nonlinear.
    quantum_cutoff : int, default 150
        Maximum ``J`` used in the quantum sum.
    """

    name = "rotational"

    def __init__(
        self,
        geometry: Geometry,
        symmetry_number: int,
        moments_of_inertia: tuple[float, ...],
        *,
        use_quantum: bool = False,
        quantum_cutoff: int = 150,
    ) -> None:
        self.geometry = geometry
        self.symmetry_number = symmetry_number
        self.use_quantum = use_quantum
        self.quantum_cutoff = quantum_cutoff
        self.theta_rot: tuple[float, ...] = tuple(
            rotational_temperature(I) for I in moments_of_inertia
        )

    # -- helpers ---------------------------------------------------------------

    @property
    def kind(self) -> GeometryKind:
        if self.geometry is Geometry.MONOATOMIC:
            return "monoatomic"
        if self.geometry is Geometry.LINEAR:
            return "linear"
        return "nonlinear"

    # -- partition function ---------------------------------------------------

    def ln_q(self, state: ResolvedState) -> float:
        if self.kind == "monoatomic":
            return 0.0
        if self.kind == "linear":
            if self.use_quantum:
                return math.log(self._q_linear_quantum(state.T))
            theta = self.theta_rot[0]
            return math.log(state.T) - math.log(self.symmetry_number * theta)
        # nonlinear
        tA, tB, tC = self.theta_rot
        return 0.5 * math.log(math.pi) - math.log(self.symmetry_number) + 0.5 * math.log(
            state.T ** 3 / (tA * tB * tC)
        )

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        if self.kind == "monoatomic":
            return 0.0
        if self.kind == "linear" and self.use_quantum:
            T = state.T
            theta = self.theta_rot[0]
            # d ln Q / dT = <y> / T  where y = J(J+1) theta / T.
            q, mean_y = self._q_linear_quantum_moments(T, theta)[:2]
            return mean_y / T
        # classical: linear -> 1/T, nonlinear -> 3/(2T)
        return 1.0 / state.T if self.kind == "linear" else 1.5 / state.T

    def cv_m(self, state: ResolvedState) -> float:
        if self.kind == "monoatomic":
            return 0.0
        if self.kind == "linear":
            if self.use_quantum:
                T = state.T
                theta = self.theta_rot[0]
                _, mean_y, mean_y2 = self._q_linear_quantum_moments(T, theta)
                return R * (mean_y2 - mean_y * mean_y)
            return R
        return 1.5 * R

    # -- quantum linear helpers -----------------------------------------------

    def _q_linear_quantum(self, T: float) -> float:
        theta = self.theta_rot[0]
        return self._q_linear_quantum_moments(T, theta)[0]

    def _q_linear_quantum_moments(self, T: float, theta: float) -> tuple[float, float, float]:
        """Return ``(Q, <y>, <y^2>)`` with ``y = J(J+1) theta / T`` for the quantum rotor.

        Delegates to the active backend's :meth:`~statthermopy.backend.Backend.
        linear_quantum_moments` kernel when one is provided (e.g. a Numba ``@njit`` kernel);
        otherwise falls back to the pure-Python loop below.
        """
        from ..backend import get_backend

        res = get_backend().linear_quantum_moments(theta, T, self.quantum_cutoff)
        if res is not None:
            return res
        q = 0.0
        s1 = 0.0  # sum of p * y
        s2 = 0.0  # sum of p * y^2
        beta = theta / T
        for J in range(self.quantum_cutoff + 1):
            y = J * (J + 1) * beta
            term = (2 * J + 1) * math.exp(-y)
            q += term
            s1 += term * y
            s2 += term * y * y
            if term < 1e-15 * q and J > 5:
                break
        if q == 0.0:
            q = 1.0  # guard
        return q, s1 / q, s2 / q

    # -- contribution (classical formula uses closed forms for S and A) -------

    def contribution(self, state: ResolvedState) -> Contribution:
        if self.kind == "monoatomic":
            return Contribution(name=self.name, ln_q=0.0, d_ln_q_dT=0.0,
                                 U_m=0.0, S_m=0.0, A_m=0.0, Cv_m=0.0)

        lnq = self.ln_q(state)
        dlnq = self.d_ln_q_dT(state)
        Cv = self.cv_m(state)
        U_m = R * state.T * state.T * dlnq
        S_m = R * (lnq + state.T * dlnq)
        A_m = -R * state.T * lnq
        return Contribution(
            name=self.name, ln_q=lnq, d_ln_q_dT=dlnq, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv
        )