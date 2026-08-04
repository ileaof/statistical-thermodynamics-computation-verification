"""Lennard–Jones combining rules for unlike pair interactions.

The binary diffusion coefficient ``D_ij`` requires the unlike-pair LJ parameters
``σ_ij`` and ``ε_ij`` and the reduced mass ``m_ij``. For the LJ potential the standard
**Lorentz–Berthelot** combining rules are used (Hirschfelder, Curtiss & Bird 1964):

    σ_ij = (σ_i + σ_j) / 2            (arithmetic mean of the diameters),
    ε_ij = sqrt(ε_i ε_j)             (geometric mean of the well depths).

These are the conventional, well-accepted rules; the architecture is open to a composition-dependent
``k_ij`` correction factor (``ε_ij = (1 - k_ij) sqrt(ε_i ε_j)``) for future high-accuracy mixture
work. The reduced mass is exact: ``m_ij = m_i m_j / (m_i + m_j)``.
"""

from __future__ import annotations

from ..core.molecule import LennardJones, Molecule

__all__ = [
    "combine_sigma",
    "combine_epsilon_over_k",
    "reduced_mass",
    "pair_sigma_m",
    "pair_epsilon_over_k",
]


def combine_sigma(lj_i: LennardJones, lj_j: LennardJones) -> float:
    """Lorentz rule: ``σ_ij = (σ_i + σ_j)/2`` in Å."""
    return 0.5 * (lj_i.sigma_angstrom + lj_j.sigma_angstrom)


def combine_epsilon_over_k(lj_i: LennardJones, lj_j: LennardJones) -> float:
    """Berthelot rule: ``ε_ij/k_B = sqrt((ε_i/k_B)(ε_j/k_B))`` in K."""
    return (lj_i.epsilon_over_k * lj_j.epsilon_over_k) ** 0.5


def reduced_mass(mol_i: Molecule, mol_j: Molecule) -> float:
    """Reduced molecular mass ``m_ij = m_i m_j / (m_i + m_j)`` (kg)."""
    mi = mol_i.molecular_mass
    mj = mol_j.molecular_mass
    return mi * mj / (mi + mj)


def pair_sigma_m(lj_i: LennardJones, lj_j: LennardJones) -> float:
    """Lorentz combined diameter ``σ_ij`` in metres."""
    return combine_sigma(lj_i, lj_j) * 1.0e-10


def pair_epsilon_over_k(lj_i: LennardJones, lj_j: LennardJones) -> float:
    """Berthelot combined well depth ``ε_ij/k_B`` in K (alias of :func:`combine_epsilon_over_k`)."""
    return combine_epsilon_over_k(lj_i, lj_j)
