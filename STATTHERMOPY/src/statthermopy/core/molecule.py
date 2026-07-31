"""Molecular-species descriptor.

A :class:`Molecule` holds the *spectroscopic* constants required to build every factor of the
molecular partition function. Nothing here is a thermodynamic property — it is the input data
from which properties are derived. The thermodynamic engine never consults empirical property
tables; it only reads these constants.

Units
-----
Internal storage is SI:

* ``molar_mass`` in kg/mol (a convenience ``molar_mass_gmol`` is kept in g/mol);
* ``moments_of_inertia`` in kg m^2 (one for linear, three for nonlinear, none for monoatomic);
* vibrational wavenumbers in cm^-1 (converted to characteristic temperatures on the fly);
* electronic term energies in cm^-1 relative to the ground state (ground at 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..units import molar_mass_gmol_to_kgmol

__all__ = ["Geometry", "VibrationalMode", "ElectronicLevel", "Molecule"]


class Geometry(str, Enum):
    """Molecular geometry, which selects the rotational partition function."""

    MONOATOMIC = "monoatomic"
    LINEAR = "linear"          # diatomics and linear polyatomics
    NONLINEAR = "nonlinear"    # general asymmetric top


@dataclass(frozen=True)
class VibrationalMode:
    """A vibrational mode (or a degenerate group of identical modes).

    Attributes
    ----------
    wavenumber_cm1 : float
        Harmonic frequency in cm^-1.
    degeneracy : int
        Number of degenerate oscillators sharing this frequency (default 1).
    """

    wavenumber_cm1: float
    degeneracy: int = 1

    def __post_init__(self) -> None:
        if self.degeneracy < 1:
            raise ValueError("Vibrational degeneracy must be >= 1.")
        if self.wavenumber_cm1 <= 0:
            raise ValueError("Vibrational wavenumber must be > 0 cm^-1.")


@dataclass(frozen=True)
class ElectronicLevel:
    """An electronic term: energy relative to the ground state and its degeneracy.

    Attributes
    ----------
    energy_cm1 : float
        Term energy in cm^-1 relative to the ground state (ground state at 0).
    degeneracy : int
        Electronic degeneracy ``g`` of this term.
    """

    energy_cm1: float
    degeneracy: int = 1

    def __post_init__(self) -> None:
        if self.degeneracy < 1:
            raise ValueError("Electronic degeneracy must be >= 1.")
        if self.energy_cm1 < 0:
            raise ValueError("Electronic energy must be >= 0 cm^-1 (ground state at 0).")


@dataclass(frozen=True)
class Molecule:
    """A molecular species and its spectroscopic constants.

    Attributes
    ----------
    name : str
        Canonical identifier used by the database, e.g. ``"N2"``.
    formula : str
        Chemical formula for display, e.g. ``"N2"`` or ``"CH4"``.
    molar_mass_gmol : float
        Molar mass in g/mol. Stored also in SI (kg/mol) as ``molar_mass``.
    geometry : Geometry
        Geometry of the molecule.
    n_atoms : int
        Number of atoms.
    symmetry_number : int
        Rotational symmetry number ``sigma`` (1 for heteronuclear, 2 for homonuclear
        diatomics, higher for symmetric polyatomics).
    moments_of_inertia : tuple[float, ...]
        Principal moments of inertia in kg m^2. Empty for monoatomic, length 1 for linear,
        length 3 for nonlinear (ordered ``I_A <= I_B <= I_C``).
    vibrational_modes : tuple[VibrationalMode, ...]
        Vibrational modes. Empty for monoatomic; ``3N-5`` for linear; ``3N-6`` for nonlinear.
    electronic_levels : tuple[ElectronicLevel, ...]
        Electronic terms with the ground state first (energy 0). Defaults to a single
        non-degenerate ground state if not specified.
    """

    name: str
    formula: str
    molar_mass_gmol: float
    geometry: Geometry
    n_atoms: int
    symmetry_number: int = 1
    moments_of_inertia: tuple[float, ...] = field(default_factory=tuple)
    vibrational_modes: tuple[VibrationalMode, ...] = field(default_factory=tuple)
    electronic_levels: tuple[ElectronicLevel, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # frozen dataclass: use object.__setattr__ to derive SI molar mass.
        object.__setattr__(self, "molar_mass", molar_mass_gmol_to_kgmol(self.molar_mass_gmol))

        if self.n_atoms < 1:
            raise ValueError("n_atoms must be >= 1.")
        if self.symmetry_number < 1:
            raise ValueError("symmetry_number must be >= 1.")

        if self.geometry is Geometry.MONOATOMIC:
            if self.n_atoms != 1:
                raise ValueError("Monoatomic molecules must have n_atoms == 1.")
            if self.moments_of_inertia or self.vibrational_modes:
                raise ValueError("Monoatomic molecules have no rotation or vibration.")
        elif self.geometry is Geometry.LINEAR:
            if self.n_atoms < 2:
                raise ValueError("Linear molecules must have n_atoms >= 2.")
            if len(self.moments_of_inertia) != 1:
                raise ValueError("Linear molecules need exactly one moment of inertia.")
            expected = 3 * self.n_atoms - 5
            n_osc = sum(m.degeneracy for m in self.vibrational_modes)
            if self.vibrational_modes and n_osc != expected:
                raise ValueError(
                    f"Linear molecule {self.name}: expected {expected} vibrational oscillators "
                    f"(3N-5), got {n_osc} (sum of degeneracies)."
                )
        elif self.geometry is Geometry.NONLINEAR:
            if self.n_atoms < 3:
                raise ValueError("Nonlinear molecules must have n_atoms >= 3.")
            if len(self.moments_of_inertia) != 3:
                raise ValueError("Nonlinear molecules need exactly three moments of inertia.")
            expected = 3 * self.n_atoms - 6
            n_osc = sum(m.degeneracy for m in self.vibrational_modes)
            if self.vibrational_modes and n_osc != expected:
                raise ValueError(
                    f"Nonlinear molecule {self.name}: expected {expected} vibrational "
                    f"oscillators (3N-6), got {n_osc} (sum of degeneracies)."
                )

        if not self.electronic_levels:
            # Default: a single, non-degenerate ground state at energy 0.
            object.__setattr__(
                self, "electronic_levels", (ElectronicLevel(0.0, 1),)
            )

    # -- convenience properties ------------------------------------------------

    @property
    def is_monoatomic(self) -> bool:
        """``True`` for monoatomic species."""
        return self.geometry is Geometry.MONOATOMIC

    @property
    def is_linear(self) -> bool:
        """``True`` for linear species (includes diatomics)."""
        return self.geometry is Geometry.LINEAR

    @property
    def is_nonlinear(self) -> bool:
        """``True`` for nonlinear species."""
        return self.geometry is Geometry.NONLINEAR

    @property
    def is_diatomic(self) -> bool:
        """``True`` for diatomic species (linear with two atoms)."""
        return self.geometry is Geometry.LINEAR and self.n_atoms == 2

    @property
    def molecular_mass(self) -> float:
        """Mass of one molecule in kg, ``molar_mass / N_A``."""
        return self.molar_mass / 6.02214076e23

    @property
    def n_vibrational_modes(self) -> int:
        """Total number of vibrational oscillators (counting degeneracies)."""
        return sum(m.degeneracy for m in self.vibrational_modes)

    @property
    def ground_state_degeneracy(self) -> int:
        """Degeneracy of the electronic ground state."""
        return self.electronic_levels[0].degeneracy

    def __repr__(self) -> str:
        return (
            f"Molecule(name={self.name!r}, formula={self.formula!r}, "
            f"M={self.molar_mass_gmol} g/mol, geometry={self.geometry.value})"
        )