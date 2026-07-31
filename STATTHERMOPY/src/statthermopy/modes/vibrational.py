"""Vibrational contribution to the molecular partition function.

Each normal mode is treated as an independent quantum harmonic oscillator with the zero of
energy at the ground state (``v = 0``), so

    Q_v,i = 1 / (1 - exp(-θ_v,i / T)) ,   θ_v,i = h c ṽ_i / k_B ,

where ``ṽ_i`` is the harmonic wavenumber in cm^-1. Degenerate modes are counted ``g_i`` times.
The total vibrational factor is the product over modes:

    Q_v = Π_i Q_v,i^(g_i) ,   ln Q_v = - Σ_i g_i ln(1 - exp(-θ_v,i / T)) .

From it:

    U_m   = Σ_i g_i R θ_v,i / (exp(θ_v,i/T) - 1)
    Cv_m  = Σ_i g_i R (θ_v,i/T)^2 exp(θ_v,i/T) / (exp(θ_v,i/T) - 1)^2
    S_m   = Σ_i g_i R [ (θ_v,i/T)/(exp(θ_v,i/T) - 1) - ln(1 - exp(-θ_v,i/T)) ]
    A_m   = Σ_i g_i R T ln(1 - exp(-θ_v,i/T))

Array work is routed through the active :mod:`~statthermopy.backend` so an accelerated backend
is actually exercised; with the default :class:`~statthermopy.backend.NumpyBackend` the results
are identical to calling NumPy directly.
"""

from __future__ import annotations

from ..backend import get_backend
from ..constants import R
from ..core.contribution import Contribution
from ..core.molecule import VibrationalMode
from ..core.state import ResolvedState
from ..units import CM1_TO_K
from .base import Mode

__all__ = ["Vibrational"]


class Vibrational(Mode):
    """Vibrational mode: a collection of (possibly degenerate) harmonic oscillators.

    Parameters
    ----------
    modes : tuple[VibrationalMode, ...]
        Vibrational modes with wavenumber (cm^-1) and degeneracy.
    """

    name = "vibrational"

    def __init__(self, modes: tuple[VibrationalMode, ...]) -> None:
        self.modes = modes
        be = get_backend()
        # Characteristic vibrational temperatures, with degeneracies.
        self.theta = be.asarray([m.wavenumber_cm1 * CM1_TO_K for m in modes])
        self.deg = be.asarray([m.degeneracy for m in modes])

    # -- partition function ---------------------------------------------------

    def ln_q(self, state: ResolvedState) -> float:
        if self.theta.size == 0:
            return 0.0
        be = get_backend()
        x = self.theta / state.T
        return float(-be.sum(self.deg * be.log1p(-be.exp(-x))))

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        if self.theta.size == 0:
            return 0.0
        be = get_backend()
        T = state.T
        x = self.theta / T
        # d ln Q / dT = Σ g (θ/T^2) / (exp(θ/T) - 1)
        return float(be.sum(self.deg * (self.theta / (T * T)) / be.expm1(x)))

    def cv_m(self, state: ResolvedState) -> float:
        if self.theta.size == 0:
            return 0.0
        be = get_backend()
        x = self.theta / state.T
        ex = be.exp(x)
        denom = be.expm1(x) ** 2
        return float(be.sum(self.deg * R * (x * x) * ex / denom))

    # -- contribution ---------------------------------------------------------

    def contribution(self, state: ResolvedState) -> Contribution:
        if self.theta.size == 0:
            return Contribution(name=self.name, ln_q=0.0, d_ln_q_dT=0.0,
                                 U_m=0.0, S_m=0.0, A_m=0.0, Cv_m=0.0)

        be = get_backend()
        T = state.T
        x = self.theta / T
        ex = be.exp(x)
        expm1 = be.expm1(x)
        lnq = float(-be.sum(self.deg * be.log1p(-be.exp(-x))))
        dlnq = float(be.sum(self.deg * (self.theta / (T * T)) / expm1))
        U_m = float(be.sum(self.deg * R * self.theta / expm1))
        Cv_m = float(be.sum(self.deg * R * (x * x) * ex / (expm1 * expm1)))
        S_m = float(be.sum(self.deg * R * (x / expm1 - be.log1p(-be.exp(-x)))))
        A_m = -R * T * lnq
        return Contribution(
            name=self.name, ln_q=lnq, d_ln_q_dT=dlnq, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
        )