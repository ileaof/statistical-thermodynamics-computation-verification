"""Vibrational contribution to the molecular partition function.

Each normal mode is treated as an independent quantum harmonic oscillator with the zero of
energy at the ground state (``v = 0``), so

    Q_v,i = 1 / (1 - exp(-θ_v,i / T)) ,   θ_v,i = h c ṽ_i / k_B ,

where ``ṽ_i`` is the harmonic wavenumber in cm^-1. Degenerate modes are counted ``g_i`` times.
The total vibrational factor is the product over modes:

    Q_v = Π_i Q_v,i^(g_i) ,   ln Q_v = - Σ_i g_i ln(1 - exp(-θ_v,i / T)) .

From it (written in the ``exp(-x)`` form, ``x = θ_v/T``, which is stable for every ``T >= 0``:
``exp(x)`` would overflow below ``T ~ θ_v / 709``, so the high-``x`` factors are kept in terms
of ``emx = exp(-x) -> 0`` and ``one = 1 - emx``):

    U_m   = Σ_i g_i R θ_v,i · emx_i / one_i
    Cv_m  = Σ_i g_i R · x_i² · emx_i / one_i²
    S_m   = Σ_i g_i R [ x_i · emx_i / one_i - ln(1 - emx_i) ]
    A_m   = Σ_i g_i R T ln(1 - emx_i)

As ``T -> 0`` (``x -> ∞``) ``emx -> 0`` and ``U_m, Cv_m, S_m -> 0`` — the harmonic oscillator
freezes out, satisfying the Third Law for this mode. At exactly ``T = 0`` the guards below
return the zero limit directly (the formal ``x = θ/0`` would otherwise produce ``inf`` and a
``0·inf`` NaN in ``Cv_m``).

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
        if self.theta.size == 0 or state.T == 0.0:
            return 0.0
        be = get_backend()
        x = self.theta / state.T
        return float(-be.sum(self.deg * be.log1p(-be.exp(-x))))

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        if self.theta.size == 0 or state.T == 0.0:
            return 0.0
        be = get_backend()
        T = state.T
        x = self.theta / T
        emx = be.exp(-x)
        one = -be.expm1(-x)  # = 1 - exp(-x), accurate for small x
        # d ln Q / dT = Σ g (θ/T^2) emx / (1 - emx)
        return float(be.sum(self.deg * (self.theta / (T * T)) * emx / one))

    def cv_m(self, state: ResolvedState) -> float:
        if self.theta.size == 0 or state.T == 0.0:
            return 0.0
        be = get_backend()
        x = self.theta / state.T
        emx = be.exp(-x)
        one = -be.expm1(-x)
        return float(be.sum(self.deg * R * (x * x) * emx / (one * one)))

    # -- contribution ---------------------------------------------------------

    def contribution(self, state: ResolvedState) -> Contribution:
        if self.theta.size == 0 or state.T == 0.0:
            return Contribution(name=self.name, ln_q=0.0, d_ln_q_dT=0.0,
                                 U_m=0.0, S_m=0.0, A_m=0.0, Cv_m=0.0)

        be = get_backend()
        T = state.T
        x = self.theta / T
        emx = be.exp(-x)
        one = -be.expm1(-x)  # = 1 - exp(-x), accurate for small x
        lnq = float(-be.sum(self.deg * be.log1p(-emx)))
        dlnq = float(be.sum(self.deg * (self.theta / (T * T)) * emx / one))
        U_m = float(be.sum(self.deg * R * self.theta * emx / one))
        Cv_m = float(be.sum(self.deg * R * (x * x) * emx / (one * one)))
        S_m = float(be.sum(self.deg * R * (x * emx / one - be.log1p(-emx))))
        A_m = -R * T * lnq
        return Contribution(
            name=self.name, ln_q=lnq, d_ln_q_dT=dlnq, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
        )