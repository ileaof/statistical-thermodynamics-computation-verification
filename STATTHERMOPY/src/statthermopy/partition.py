"""Assembly of the molecular partition function.

The :class:`PartitionFunction` builds the four mode objects from a :class:`Molecule`,
evaluates each factor of

    Q = Q_t Q_r Q_v Q_e

at a :class:`State`, and exposes the factors and their logarithms. It is the bridge between the
spectroscopic data and the thermodynamic aggregator.

The ``N!`` indistinguishability correction is *not* part of ``Q`` here — it is a property of the
canonical ``N``-particle partition function ``Z_N = Q^N / N!`` and is applied, per mole, inside
the translational contribution (see :mod:`statthermopy.modes.translational`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core.molecule import Molecule
from .core.state import ResolvedState, State
from .modes import Electronic, Rotational, Translational, Vibrational

__all__ = ["PartitionFunction", "PartitionValues"]


@dataclass(frozen=True)
class PartitionValues:
    """The four partition-function factors and the total at a given state.

    Attributes
    ----------
    Qt, Qr, Qv, Qe, Qtotal : float
        Translational, rotational, vibrational, electronic and total molecular partition
        functions.
    ln_Qt, ln_Qr, ln_Qv, ln_Qe, ln_Qtotal : float
        Their natural logs (more numerically convenient).
    """

    Qt: float
    Qr: float
    Qv: float
    Qe: float
    Qtotal: float
    ln_Qt: float
    ln_Qr: float
    ln_Qv: float
    ln_Qe: float
    ln_Qtotal: float


class PartitionFunction:
    """Molecular partition function ``Q = Q_t Q_r Q_v Q_e`` for a molecule.

    Parameters
    ----------
    molecule : Molecule
        The molecular species.
    use_quantum_rotation : bool, default False
        Use the exact quantum rigid-rotor sum for linear molecules.
    """

    def __init__(self, molecule: Molecule, *, use_quantum_rotation: bool = False) -> None:
        self.molecule = molecule
        self.translational = Translational(molecule.molecular_mass)
        self.rotational = Rotational(
            molecule.geometry,
            molecule.symmetry_number,
            molecule.moments_of_inertia,
            use_quantum=use_quantum_rotation,
        )
        self.vibrational = Vibrational(molecule.vibrational_modes)
        self.electronic = Electronic(molecule.electronic_levels)

    # -- evaluation ------------------------------------------------------------

    def _resolved(self, state: State) -> ResolvedState:
        return state.resolve(self.molecule.molar_mass)

    def evaluate(self, state: State) -> PartitionValues:
        """Evaluate all factors at ``state``."""
        rs = self._resolved(state)
        lnQt = self.translational.ln_q(rs)
        lnQr = self.rotational.ln_q(rs)
        lnQv = self.vibrational.ln_q(rs)
        lnQe = self.electronic.ln_q(rs)
        lnQ = lnQt + lnQr + lnQv + lnQe
        return PartitionValues(
            Qt=math.exp(lnQt), Qr=math.exp(lnQr), Qv=math.exp(lnQv),
            Qe=math.exp(lnQe), Qtotal=math.exp(lnQ),
            ln_Qt=lnQt, ln_Qr=lnQr, ln_Qv=lnQv, ln_Qe=lnQe, ln_Qtotal=lnQ,
        )

    def contributions(self, state: State):
        """Return the four :class:`Contribution` objects at ``state`` (one per mode)."""
        rs = self._resolved(state)
        return {
            "translational": self.translational.contribution(rs),
            "rotational": self.rotational.contribution(rs),
            "vibrational": self.vibrational.contribution(rs),
            "electronic": self.electronic.contribution(rs),
        }

    # -- convenience accessors -------------------------------------------------

    @property
    def modes(self) -> dict:
        """Mapping of the four mode objects."""
        return {
            "translational": self.translational,
            "rotational": self.rotational,
            "vibrational": self.vibrational,
            "electronic": self.electronic,
        }