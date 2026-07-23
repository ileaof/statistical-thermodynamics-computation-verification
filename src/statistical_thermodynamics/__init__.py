"""Statistical Thermodynamics -- reusable computational library.

Companion package to the textbook

    *Statistical Thermodynamics: Theory, Computation, and Molecular Applications*
    *-- A Computational Approach with Python* by I. L. Ferreira.

The package collects the physics and numerical tools that recur throughout the
book's worked examples into a small, well-documented and unit-tested library.
The per-chapter example programs remain fully self-contained; this library
exists so that readers can *reuse* the same verified building blocks in their own
work.

Subpackages / modules
---------------------
constants
    CODATA/SI physical constants.
partition_functions
    Two-level, harmonic, rotational, vibrational and translational partition
    functions.
probability
    Multiplicities, combinatorics and canonical probabilities.
thermodynamics
    Ideal-gas thermodynamics (Sackur-Tetrode, chemical potential, fluctuations).
kinetic_theory
    Maxwell-Boltzmann speeds, collisions and the mean free path.
transport
    Elementary transport coefficients (diffusion, viscosity, conductivity).
potentials
    Lennard-Jones and hard-sphere pair potentials and the Mayer function.
quantum_statistics
    Occupation numbers, Bose functions and blackbody radiation.
solids
    Einstein and Debye heat-capacity models.
equilibrium
    Virial coefficients, Carnahan-Starling EOS and chemical equilibrium.
numerical_methods
    Metropolis sampling, autocorrelation, blocking and bootstrap error analysis.
plotting
    Shared publication-quality Matplotlib styling.
utilities
    Reproducible RNGs, error metrics and convergence-order fitting.

Examples
--------
>>> import statistical_thermodynamics as st
>>> round(st.constants.N_A * st.constants.k_B, 6) == round(st.constants.R, 6)
True
>>> st.kinetic_theory.characteristic_speeds(28.0134 * st.constants.u, 300.0)[0] > 0
True
"""

from __future__ import annotations

from . import (  # noqa: F401
    constants,
    equilibrium,
    kinetic_theory,
    numerical_methods,
    partition_functions,
    plotting,
    potentials,
    probability,
    quantum_statistics,
    solids,
    thermodynamics,
    transport,
    utilities,
)

__version__ = "1.0.0"
__author__ = "I. L. Ferreira"
__license__ = "MIT"

__all__ = [
    "constants",
    "partition_functions",
    "probability",
    "thermodynamics",
    "kinetic_theory",
    "transport",
    "potentials",
    "quantum_statistics",
    "solids",
    "equilibrium",
    "numerical_methods",
    "plotting",
    "utilities",
    "__version__",
]
