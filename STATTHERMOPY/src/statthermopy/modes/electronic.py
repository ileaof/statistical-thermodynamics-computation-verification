"""Electronic contribution to the molecular partition function.

The electronic partition function sums over the electronic terms of the molecule, with the
ground state taken as the energy zero:

    Q_e = Σ_j g_j exp(-θ_e,j / T) ,   θ_e,j = h c T̃_j / k_B ,

where ``T̃_j`` is the term energy in cm^-1 (ground state at 0) and ``g_j`` its degeneracy. The
Boltzmann population of state ``j`` is

    p_j = g_j exp(-θ_e,j / T) / Q_e .

Thermodynamic contributions:

    U_m  = R ⟨θ_e⟩
    Cv_m = R (⟨θ_e^2⟩ - ⟨θ_e⟩^2) / T^2
    S_m  = R [ ln Q_e + ⟨θ_e⟩ / T ]
    A_m  = -R T ln Q_e

Array work is routed through the active :mod:`~statthermopy.backend`; with the default
:class:`~statthermopy.backend.NumpyBackend` the results are identical to calling NumPy directly.
"""

from __future__ import annotations

from ..backend import get_backend
from ..constants import R
from ..core.contribution import Contribution
from ..core.molecule import ElectronicLevel
from ..core.state import ResolvedState
from ..units import CM1_TO_K
from .base import Mode

__all__ = ["Electronic"]


class Electronic(Mode):
    """Electronic mode from a list of electronic terms.

    Parameters
    ----------
    levels : tuple[ElectronicLevel, ...]
        Electronic terms; the first must be the ground state (energy 0).
    """

    name = "electronic"

    def __init__(self, levels: tuple[ElectronicLevel, ...]) -> None:
        if not levels:
            raise ValueError("At least one electronic level (the ground state) is required.")
        self.levels = levels
        be = get_backend()
        self.theta = be.asarray([lvl.energy_cm1 * CM1_TO_K for lvl in levels])
        self.g = be.asarray([lvl.degeneracy for lvl in levels])

    # -- populations ----------------------------------------------------------

    def populations(self, T: float):
        """Boltzmann populations ``p_j`` of each electronic state at temperature ``T``."""
        be = get_backend()
        w = self.g * be.exp(-self.theta / T)
        return w / be.sum(w)

    # -- partition function ---------------------------------------------------

    def ln_q(self, state: ResolvedState) -> float:
        be = get_backend()
        return float(be.log(be.sum(self.g * be.exp(-self.theta / state.T))))

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        be = get_backend()
        T = state.T
        w = self.g * be.exp(-self.theta / T)
        q = be.sum(w)
        mean_theta = be.sum(w * self.theta) / q
        return mean_theta / (T * T)

    def cv_m(self, state: ResolvedState) -> float:
        be = get_backend()
        T = state.T
        w = self.g * be.exp(-self.theta / T)
        q = be.sum(w)
        mean_theta = be.sum(w * self.theta) / q
        mean_theta2 = be.sum(w * self.theta * self.theta) / q
        return float(R * (mean_theta2 - mean_theta * mean_theta) / (T * T))

    # -- contribution ---------------------------------------------------------

    def contribution(self, state: ResolvedState) -> Contribution:
        be = get_backend()
        T = state.T
        w = self.g * be.exp(-self.theta / T)
        q = be.sum(w)
        lnq = float(be.log(q))
        mean_theta = float(be.sum(w * self.theta) / q)
        U_m = R * mean_theta
        S_m = R * (lnq + mean_theta / T)
        A_m = -R * T * lnq
        Cv_m = self.cv_m(state)
        dlnq = mean_theta / (T * T)
        return Contribution(
            name=self.name, ln_q=lnq, d_ln_q_dT=dlnq, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
        )