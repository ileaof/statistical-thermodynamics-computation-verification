"""Per-mode thermodynamic contribution.

Each contribution mode (translational, rotational, vibrational, electronic) returns a
:class:`Contribution` holding the quantities that can be derived from its piece of the
molecular partition function. The :class:`~statthermopy.thermodynamics.Thermodynamics`
aggregator sums these contributions and then adds the ideal-gas ``P V`` term and the
indistinguishability (``N!``) correction — the latter is allocated to the translational mode —
to obtain the full set of thermodynamic properties.

All values are *molar* (per mole, J/mol or J/mol/K). Massic values are obtained downstream by
dividing by the molar mass.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Contribution"]


@dataclass(frozen=True)
class Contribution:
    """Molar thermodynamic contribution of a single partition-function mode.

    Attributes
    ----------
    name : str
        Mode identifier (``"translational"``, ``"rotational"``, ``"vibrational"``,
        ``"electronic"``).
    ln_q : float
        Natural log of the *molecular* partition-function factor for this mode,
        ``ln Q_mode``. For the translational mode this includes ``ln V``.
    d_ln_q_dT : float
        Temperature derivative ``(∂ ln Q_mode / ∂T)_V`` (K^-1).
    U_m : float
        Molar internal-energy contribution (J/mol).
    S_m : float
        Molar entropy contribution (J/mol/K).
    A_m : float
        Molar Helmholtz free-energy contribution (J/mol).
    Cv_m : float
        Molar constant-volume heat-capacity contribution (J/mol/K).
    """

    name: str
    ln_q: float
    d_ln_q_dT: float
    U_m: float
    S_m: float
    A_m: float
    Cv_m: float

    def __add__(self, other: "Contribution") -> "Contribution":
        """Sum two contributions mode-by-mode (used to assemble the total)."""
        return Contribution(
            name=f"{self.name}+{other.name}",
            ln_q=self.ln_q + other.ln_q,
            d_ln_q_dT=self.d_ln_q_dT + other.d_ln_q_dT,
            U_m=self.U_m + other.U_m,
            S_m=self.S_m + other.S_m,
            A_m=self.A_m + other.A_m,
            Cv_m=self.Cv_m + other.Cv_m,
        )