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

__all__ = [
    "Geometry",
    "VibrationalMode",
    "InternalRotor",
    "ElectronicLevel",
    "LennardJones",
    "Molecule",
]


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
class InternalRotor:
    """A one-dimensional hindered internal rotor (e.g. a methyl-top torsion).

    The rotor moves in an ``n``-fold symmetric potential
    ``V(φ) = (V_n / 2) [1 - cos(n φ)]`` with kinetic term ``F P²`` where
    ``F = ħ² / (2 I_r)`` is the internal-rotation constant and ``I_r`` the reduced moment
    of inertia. The torsional eigenvalues are obtained by diagonalising this (Mathieu)
    Hamiltonian in the free-rotor basis; see
    :class:`~statthermopy.modes.hindered_rotor.HinderedRotor`.

    Both constants are stored in cm^-1 (the spectroscopic convention), exactly as
    vibrational wavenumbers are.

    Attributes
    ----------
    rotation_constant_cm1 : float
        Internal-rotation constant ``F = ħ² / (2 I_r)`` in cm^-1.
    barrier_cm1 : float
        Hindering-potential barrier height ``V_n`` in cm^-1.
    symmetry : int
        Internal symmetry number ``σ_int`` (3 for a methyl top). Divides the rotor
        partition function.
    n_minima : int
        Potential periodicity ``n`` (number of equivalent minima; 3 for a methyl top).
    degeneracy : int
        Number of identical, independent rotors sharing these constants (default 1).
    """

    rotation_constant_cm1: float
    barrier_cm1: float
    symmetry: int = 3
    n_minima: int = 3
    degeneracy: int = 1

    def __post_init__(self) -> None:
        if self.rotation_constant_cm1 <= 0:
            raise ValueError("Internal-rotation constant F must be > 0 cm^-1.")
        if self.barrier_cm1 < 0:
            raise ValueError("Internal-rotation barrier must be >= 0 cm^-1.")
        if self.symmetry < 1:
            raise ValueError("Internal symmetry number must be >= 1.")
        if self.n_minima < 1:
            raise ValueError("Potential periodicity n must be >= 1.")
        if self.degeneracy < 1:
            raise ValueError("Internal-rotor degeneracy must be >= 1.")


@dataclass(frozen=True)
class LennardJones:
    """Lennard–Jones 12-6 potential parameters for a species.

    These are *molecular* potential parameters (the collision diameter ``σ`` and well depth
    ``ε``), on the same footing as the moments of inertia: they characterise the intermolecular
    pair potential

    .. math:: u(r) = 4\\varepsilon\\bigl[(\\sigma/r)^{12} - (\\sigma/r)^{6}\\bigr]

    used by the Chapman–Enskog kinetic theory (:mod:`statthermopy.transport`) to derive the
    transport properties (viscosity, thermal conductivity, diffusion) from first principles. They
    are **not** thermodynamic property data and **not** an empirical equation of state — they are
    the inputs to the dilute-gas collision integrals, exactly as the spectroscopic constants are
    the inputs to the partition function.

    The values are the standard viscosity-derived LJ parameters (Svehla 1962 / Hirschfelder,
    Curtiss & Bird / Poling, Prausnitz & O'Connell). For polar species (H₂O, NH₃, H₂S, SO₂) the LJ
    potential is an approximation and the transport predictions carry larger uncertainty.

    Attributes
    ----------
    sigma_angstrom : float
        Collision diameter ``σ`` in Å (1 Å = 1e-10 m).
    epsilon_over_k : float
        Well depth divided by Boltzmann's constant, ``ε/k_B``, in K (the Lennard–Jones
        characteristic temperature).
    note : str
        Provenance / reliability flag (e.g. source, or "polar — LJ approximate").
    """

    sigma_angstrom: float
    epsilon_over_k: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.sigma_angstrom <= 0:
            raise ValueError("Lennard–Jones sigma must be > 0 Å.")
        if self.epsilon_over_k <= 0:
            raise ValueError("Lennard–Jones epsilon/k_B must be > 0 K.")

    @property
    def sigma_m(self) -> float:
        """Collision diameter in metres."""
        return self.sigma_angstrom * 1.0e-10

    @property
    def epsilon(self) -> float:
        """Well depth ``ε = (ε/k_B) k_B`` in joules."""
        from ..constants import k_B

        return self.epsilon_over_k * k_B


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
        Harmonic vibrational modes. Empty for monoatomic. Together with any internal
        rotors the internal-motion count is ``3N-5`` for linear and ``3N-6`` for nonlinear.
    internal_rotors : tuple[InternalRotor, ...]
        Hindered internal rotors (e.g. methyl torsions). Each rotor replaces one internal
        degree of freedom that would otherwise be counted as a harmonic vibration, so the
        oscillator count plus the rotor count must still equal ``3N-5`` / ``3N-6``.
    electronic_levels : tuple[ElectronicLevel, ...]
        Electronic terms with the ground state first (energy 0). Defaults to a single
        non-degenerate ground state if not specified.
    lennard_jones : LennardJones | None
        Lennard–Jones potential parameters (σ, ε/k_B) used by the Chapman–Enskog transport
        module (:mod:`statthermopy.transport`) to derive viscosity, thermal conductivity and
        diffusion. ``None`` for species without LJ data; the thermodynamic properties remain
        available either way.
    """

    name: str
    formula: str
    molar_mass_gmol: float
    geometry: Geometry
    n_atoms: int
    symmetry_number: int = 1
    moments_of_inertia: tuple[float, ...] = field(default_factory=tuple)
    vibrational_modes: tuple[VibrationalMode, ...] = field(default_factory=tuple)
    internal_rotors: tuple[InternalRotor, ...] = field(default_factory=tuple)
    electronic_levels: tuple[ElectronicLevel, ...] = field(default_factory=tuple)
    lennard_jones: LennardJones | None = None

    def __post_init__(self) -> None:
        # frozen dataclass: use object.__setattr__ to derive SI molar mass.
        object.__setattr__(self, "molar_mass", molar_mass_gmol_to_kgmol(self.molar_mass_gmol))

        if self.n_atoms < 1:
            raise ValueError("n_atoms must be >= 1.")
        if self.symmetry_number < 1:
            raise ValueError("symmetry_number must be >= 1.")

        n_rot = sum(r.degeneracy for r in self.internal_rotors)

        if self.geometry is Geometry.MONOATOMIC:
            if self.n_atoms != 1:
                raise ValueError("Monoatomic molecules must have n_atoms == 1.")
            if self.moments_of_inertia or self.vibrational_modes or self.internal_rotors:
                raise ValueError(
                    "Monoatomic molecules have no rotation, vibration or internal rotation."
                )
        elif self.geometry is Geometry.LINEAR:
            if self.n_atoms < 2:
                raise ValueError("Linear molecules must have n_atoms >= 2.")
            if len(self.moments_of_inertia) != 1:
                raise ValueError("Linear molecules need exactly one moment of inertia.")
            expected = 3 * self.n_atoms - 5
            n_osc = sum(m.degeneracy for m in self.vibrational_modes)
            if (self.vibrational_modes or self.internal_rotors) and n_osc + n_rot != expected:
                raise ValueError(
                    f"Linear molecule {self.name}: expected {expected} internal modes (3N-5), "
                    f"got {n_osc} oscillators + {n_rot} internal rotors = {n_osc + n_rot}."
                )
        elif self.geometry is Geometry.NONLINEAR:
            if self.n_atoms < 3:
                raise ValueError("Nonlinear molecules must have n_atoms >= 3.")
            if len(self.moments_of_inertia) != 3:
                raise ValueError("Nonlinear molecules need exactly three moments of inertia.")
            expected = 3 * self.n_atoms - 6
            n_osc = sum(m.degeneracy for m in self.vibrational_modes)
            if (self.vibrational_modes or self.internal_rotors) and n_osc + n_rot != expected:
                raise ValueError(
                    f"Nonlinear molecule {self.name}: expected {expected} internal modes (3N-6), "
                    f"got {n_osc} oscillators + {n_rot} internal rotors = {n_osc + n_rot}."
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
    def n_internal_rotors(self) -> int:
        """Total number of hindered internal rotors (counting degeneracies)."""
        return sum(r.degeneracy for r in self.internal_rotors)

    @property
    def ground_state_degeneracy(self) -> int:
        """Degeneracy of the electronic ground state."""
        return self.electronic_levels[0].degeneracy

    @property
    def has_lennard_jones(self) -> bool:
        """``True`` if Lennard–Jones parameters are available (transport properties)."""
        return self.lennard_jones is not None

    def __repr__(self) -> str:
        return (
            f"Molecule(name={self.name!r}, formula={self.formula!r}, "
            f"M={self.molar_mass_gmol} g/mol, geometry={self.geometry.value})"
        )
