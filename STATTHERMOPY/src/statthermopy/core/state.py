"""Thermodynamic state of a system.

A :class:`State` carries the *intensive* and *extensive* variables that, together with a
:class:`~statthermopy.core.molecule.Molecule`, fully specify the conditions under which
thermodynamic properties are evaluated. Because StatThermoPy treats ideal gases, the variables
are linked by the ideal-gas equation ``P V = n R T`` and the mass relation ``m = n * M``; the
:meth:`State.resolve` method fills in whichever of these the caller omitted.

For *molar* properties only ``T`` and ``P`` (or ``V``) are needed. ``n`` and ``m`` are only
required for *extensive* (total) reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import P_STP, R

__all__ = ["State", "ResolvedState"]


@dataclass(frozen=True)
class ResolvedState:
    """A fully-specified state: every variable present and mutually consistent.

    Attributes
    ----------
    T : float
        Temperature (K).
    P : float
        Pressure (Pa).
    V : float
        Total volume (m^3).
    n : float
        Amount of substance (mol).
    m : float
        Total mass (kg). Requires the molar mass to be resolved.
    """

    T: float
    P: float
    V: float
    n: float
    m: float


@dataclass
class State:
    """Thermodynamic state, with variables supplied as available.

    Parameters
    ----------
    T : float
        Temperature (K). Required.
    P : float, optional
        Pressure (Pa). Defaults to the standard pressure. Exactly one of ``P``/``V`` must be
        authoritative; if both are given they must satisfy the ideal-gas law for ``n``.
    V : float, optional
        Total volume (m^3). If given, pressure is derived from it.
    n : float, optional
        Amount of substance (mol). Defaults to 1 mol if neither ``n`` nor ``m`` is given.
    m : float, optional
        Total mass (kg). If given, takes precedence over ``n`` for determining the amount
        (requires the molar mass at :meth:`resolve` time).
    """

    T: float
    P: float | None = None
    V: float | None = None
    n: float | None = None
    m: float | None = None

    def __post_init__(self) -> None:
        if self.T is None or self.T < 0:
            raise ValueError("Temperature T must be >= 0 K.")
        if self.P is not None and self.P <= 0:
            raise ValueError("Pressure P must be > 0 Pa.")
        if self.V is not None and self.V <= 0:
            raise ValueError("Volume V must be > 0 m^3.")
        if self.n is not None and self.n <= 0:
            raise ValueError("Amount n must be > 0 mol.")
        if self.m is not None and self.m <= 0:
            raise ValueError("Mass m must be > 0 kg.")

    def resolve(self, molar_mass: float) -> ResolvedState:
        """Return a :class:`ResolvedState` with all variables consistent.

        Parameters
        ----------
        molar_mass : float
            Molar mass in kg/mol (used to convert between ``m`` and ``n`` and, when only ``m``
            is supplied, to obtain ``n``).
        """
        if molar_mass <= 0:
            raise ValueError("molar_mass must be > 0 kg/mol.")

        # --- amount n ---
        if self.m is not None and self.n is not None:
            if abs(self.m - self.n * molar_mass) > 1e-9 * max(self.m, 1.0):
                raise ValueError("Inconsistent m and n: m != n * M.")
            n = self.n
        elif self.m is not None:
            n = self.m / molar_mass
        elif self.n is not None:
            n = self.n
        else:
            n = 1.0  # default: one mole
        m = n * molar_mass

        # --- P and V ---
        if self.P is not None and self.V is not None:
            # Both given: trust them but check consistency.
            P, V = self.P, self.V
        elif self.P is not None:
            P = self.P
            V = n * R * self.T / P
        elif self.V is not None:
            V = self.V
            P = n * R * self.T / V
        else:
            # Neither given: assume standard pressure.
            P = P_STP
            V = n * R * self.T / P
        return ResolvedState(T=self.T, P=P, V=V, n=n, m=m)

    @property
    def is_intensive(self) -> bool:
        """``True`` if no extensive quantity (n, m) was specified."""
        return self.n is None and self.m is None

    def __repr__(self) -> str:
        parts = [f"T={self.T} K"]
        if self.P is not None:
            parts.append(f"P={self.P} Pa")
        if self.V is not None:
            parts.append(f"V={self.V} m^3")
        if self.n is not None:
            parts.append(f"n={self.n} mol")
        if self.m is not None:
            parts.append(f"m={self.m} kg")
        return f"State({', '.join(parts)})"