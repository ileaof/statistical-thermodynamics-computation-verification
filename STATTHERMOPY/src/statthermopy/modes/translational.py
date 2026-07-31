"""Translational contribution to the molecular partition function.

For a particle of mass ``m`` in a volume ``V`` the translational partition function is

    Q_t = (2 π m k_B T / h^2)^(3/2) * V .

It is the only mode that carries a volume dependence and the only one for which the
indistinguishability of the ``N`` identical molecules matters. The molar entropy and Helmholtz
free energy therefore receive the ``-ln N_A + 1`` correction (Stirling's approximation to
``ln N!``); the internal energy and heat capacity are unaffected because that correction is
linear in ``T`` and cancels in ``U = A + T S``.

This yields the Sackur-Tetrode entropy

    S_m,t = R [ (3/2) ln(2 π m k_B T / h^2) + ln(k_B T / P) + 5/2 ] ,

with the molar volume ``V_m = R T / P`` so that ``V_m / N_A = k_B T / P``.
"""

from __future__ import annotations

import math

from ..constants import N_A, R, h, k_B
from ..core.contribution import Contribution
from ..core.state import ResolvedState
from .base import Mode

__all__ = ["Translational"]


class Translational(Mode):
    """Translational mode for a molecule of mass ``m`` (kg)."""

    name = "translational"

    def __init__(self, molecular_mass: float) -> None:
        if molecular_mass <= 0:
            raise ValueError("molecular_mass must be > 0 kg.")
        self.molecular_mass = float(molecular_mass)

    # -- partition function ---------------------------------------------------

    def ln_q(self, state: ResolvedState) -> float:
        """``ln Q_t = (3/2) ln(2 π m k T / h^2) + ln V_m``.

        The molar volume ``V_m = V / n`` is used so that the resulting thermodynamic
        potentials are *intensive* (independent of the amount of substance). The total
        entropy ``S = n S_m`` is then properly extensive (the ``n ln n`` Gibbs-paradox term
        is cancelled by the indistinguishability correction ``-ln N`` absorbed here via the
        ``-ln N_A + 1`` term combined with ``ln V_m``).
        """
        coeff = 2.0 * math.pi * self.molecular_mass * k_B / (h * h)
        V_molar = state.V / state.n
        return 1.5 * math.log(coeff * state.T) + math.log(V_molar)

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        """``(d ln Q_t / dT)_V = 3 / (2 T)``."""
        return 1.5 / state.T

    def cv_m(self, state: ResolvedState) -> float:
        """``Cv_m,t = (3/2) R``."""
        return 1.5 * R

    # -- contribution (override to add the indistinguishability correction) -----

    def contribution(self, state: ResolvedState) -> Contribution:
        lnq = self.ln_q(state)
        dlnq = self.d_ln_q_dT(state)
        U_m = 1.5 * R * state.T
        Cv_m = 1.5 * R
        # Sackur-Tetrode: S_m = R [ ln Q_t + T (d ln Q_t/dT) - ln N_A + 1 ].
        S_m = R * (lnq + state.T * dlnq - math.log(N_A) + 1.0)
        # A_m = -R T [ ln Q_t - ln N_A + 1 ].
        A_m = -R * state.T * (lnq - math.log(N_A) + 1.0)
        return Contribution(
            name=self.name, ln_q=lnq, d_ln_q_dT=dlnq, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
        )