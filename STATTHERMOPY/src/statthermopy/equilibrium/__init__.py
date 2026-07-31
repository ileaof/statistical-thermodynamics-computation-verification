"""Chemical-equilibrium architecture (placeholder for a future phase).

Phase 1 only *prepares the architecture* for chemical equilibrium so that future work on
Gibbs-energy minimisation, reaction equilibria and equilibrium constants does not require
changes to the public API. Nothing here is functional yet; the documented interfaces describe
the intended design.

Planned components
------------------
* :class:`ChemicalSystem` — a set of species and their elemental composition.
* :func:`gibbs_minimisation` — minimise ``G = Σ n_i μ_i(T, P)`` subject to element balance, using
  the chemical potentials ``μ_i`` computed by :mod:`statthermopy.thermodynamics` (Lagrange
  multipliers / RAND algorithm).
* :class:`Reaction` — a stoichiometric reaction with ``ΔG°`` and ``K(T) = exp(-ΔG°/RT)`` built
  from the species' Gibbs free energies.
* :class:`EquilibriumConstant` — temperature-dependent ``K(T)`` from first-principles ``G``.

Because the chemical potential of each species is already available from the partition function,
all of the above can be derived without empirical correlations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class EquilibriumNotImplemented(NotImplementedError):
    """Raised when an equilibrium feature that belongs to a later phase is invoked."""


@dataclass
class Reaction:
    """Stoichiometric reaction (placeholder).

    Attributes
    ----------
    reactants : dict[str, float]
        Mapping species name -> stoichiometric coefficient (positive).
    products : dict[str, float]
        Mapping species name -> stoichiometric coefficient (positive).
    """

    reactants: dict[str, float] = field(default_factory=dict)
    products: dict[str, float] = field(default_factory=dict)

    def delta_G(self, T: float, P: float = 101325.0) -> float:
        """Reaction Gibbs free energy ``ΔG = Σ ν G`` (to be implemented)."""
        raise EquilibriumNotImplemented(
            "Reaction.delta_G is part of the equilibrium phase (not yet implemented)."
        )

    def equilibrium_constant(self, T: float, P: float = 101325.0) -> float:
        """``K(T) = exp(-ΔG°/RT)`` (to be implemented)."""
        raise EquilibriumNotImplemented(
            "Reaction.equilibrium_constant is part of the equilibrium phase."
        )


def gibbs_minimisation(species: list[str], elements: dict, T: float, P: float):
    """Minimise total Gibbs free energy subject to element conservation (placeholder)."""
    raise EquilibriumNotImplemented(
        "Gibbs-energy minimisation is part of the equilibrium phase (not yet implemented)."
    )


__all__ = ["Reaction", "EquilibriumNotImplemented", "gibbs_minimisation"]