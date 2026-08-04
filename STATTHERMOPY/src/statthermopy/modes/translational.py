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

        At ``T = 0`` the classical translational partition function collapses (``Q_t -> 0``,
        ``ln Q_t -> -inf``); the Sackur–Tetrode entropy diverges, which is the well-known
        failure of the *classical* ideal gas at low temperature (the Third Law is not
        satisfied by the classical translational mode). ``U_m`` and ``Cv_m`` remain well
        defined (``0`` and ``3/2 R``), so the thermal field ``T_v = U_m / Cv_m -> 0`` stays
        finite and continuous.
        """
        if state.T == 0.0:
            return float("-inf")
        coeff = 2.0 * math.pi * self.molecular_mass * k_B / (h * h)
        V_molar = state.V / state.n
        return 1.5 * math.log(coeff * state.T) + math.log(V_molar)

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        """``(d ln Q_t / dT)_V = 3 / (2 T)``."""
        return 0.0 if state.T == 0.0 else 1.5 / state.T

    def cv_m(self, state: ResolvedState) -> float:
        """``Cv_m,t = (3/2) R``."""
        return 1.5 * R

    # -- contribution (override to add the indistinguishability correction) -----

    def contribution(self, state: ResolvedState) -> Contribution:
        U_m = 1.5 * R * state.T
        Cv_m = 1.5 * R
        if state.T == 0.0:
            # T = 0 limit: U = 0, Cv = 3/2 R; ln_q and S diverge to -inf (classical ideal
            # gas), A = 0 (the T factor kills the -R T ln_q term). Kept finite so the
            # thermal-field path (U, Cv, H, Cp) stays NaN-free at T = 0.
            lnq = float("-inf")
            S_m = float("-inf")
            A_m = 0.0
            return Contribution(
                name=self.name, ln_q=lnq, d_ln_q_dT=0.0, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
            )
        lnq = self.ln_q(state)
        dlnq = self.d_ln_q_dT(state)
        # Sackur-Tetrode: S_m = R [ ln Q_t + T (d ln Q_t/dT) - ln N_A + 1 ].
        S_m = R * (lnq + state.T * dlnq - math.log(N_A) + 1.0)
        # A_m = -R T [ ln Q_t - ln N_A + 1 ].
        A_m = -R * state.T * (lnq - math.log(N_A) + 1.0)
        return Contribution(
            name=self.name, ln_q=lnq, d_ln_q_dT=dlnq, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
        )