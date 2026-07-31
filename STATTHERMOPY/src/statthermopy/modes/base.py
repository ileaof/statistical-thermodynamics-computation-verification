"""Base class for partition-function modes.

Every contribution to the molecular partition function — translational, rotational,
vibrational, electronic — is implemented as a subclass of :class:`Mode`. A mode knows how to
evaluate, at a given resolved state, the log of its molecular partition-function factor and the
thermodynamic derivatives that follow from it. The aggregator
(:class:`~statthermopy.partition.PartitionFunction`,
:class:`~statthermopy.thermodynamics.Thermodynamics`) combines the four modes without needing
to know their internals.

The canonical-ensemble relations used by every mode (per mole, ``n = 1``):

    U_m  = R T^2  (d ln Q / dT)_V
    S_m  = R [ ln Q + T (d ln Q / dT)_V ]      (+ indistinguishability term, translational only)
    A_m  = -R T   ln Q                          (+ indistinguishability term, translational only)
    Cv_m = d U_m / dT

Modes other than translational carry no volume dependence, so their ``A_m`` and ``G_m``
contributions coincide; the ``P V`` and ``N!`` corrections are applied once, at the
translational mode, by the aggregator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.contribution import Contribution
from ..core.state import ResolvedState

__all__ = ["Mode"]


class Mode(ABC):
    """Abstract base for a partition-function mode."""

    #: Mode name used in reports.
    name: str = "mode"

    @abstractmethod
    def ln_q(self, state: ResolvedState) -> float:
        """Natural log of the molecular partition-function factor, ``ln Q_mode``."""

    @abstractmethod
    def d_ln_q_dT(self, state: ResolvedState) -> float:
        """Temperature derivative ``(d ln Q_mode / dT)_V`` (K^-1)."""

    def contribution(self, state: ResolvedState) -> Contribution:
        """Assemble the molar :class:`Contribution` of this mode at ``state``.

        The default implementation derives ``U``, ``S``, ``A``, ``Cv`` from ``ln_q`` and its
        temperature derivative via the canonical-ensemble relations. Subclasses may override
        (for example, the translational mode adds the indistinguishability correction).
        """
        from ..constants import R

        lnq = self.ln_q(state)
        dlnq = self.d_ln_q_dT(state)
        U_m = R * state.T * state.T * dlnq
        S_m = R * (lnq + state.T * dlnq)
        A_m = -R * state.T * lnq
        Cv_m = self.cv_m(state)
        return Contribution(
            name=self.name, ln_q=lnq, d_ln_q_dT=dlnq, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
        )

    @abstractmethod
    def cv_m(self, state: ResolvedState) -> float:
        """Molar constant-volume heat-capacity contribution (J/mol/K).

        Implemented analytically per mode for speed and accuracy; the general
        ``d/dT`` of ``U_m`` is recovered as a consistency check in tests.
        """