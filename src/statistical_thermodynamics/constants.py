"""Physical constants in SI units.

All values are the CODATA / SI-2019 exact defining constants where applicable,
matching the numbers hard-coded throughout the book's example programs.  Import
them from here so that every calculation in the package uses one authoritative
set of values.

Examples
--------
>>> from statistical_thermodynamics.constants import k_B, N_A, R
>>> round(N_A * k_B, 6) == round(R, 6)
True
"""

from __future__ import annotations

import numpy as np

# --- Exact defining constants (SI, effective 2019 redefinition) -------------
k_B = 1.380649e-23           #: Boltzmann constant, J/K (exact)
h = 6.62607015e-34           #: Planck constant, J s (exact)
c = 2.99792458e8             #: speed of light in vacuum, m/s (exact)
N_A = 6.02214076e23          #: Avogadro constant, 1/mol (exact)
e_charge = 1.602176634e-19   #: elementary charge, C (exact)

# --- Derived constants ------------------------------------------------------
hbar = h / (2.0 * np.pi)     #: reduced Planck constant, J s
R = N_A * k_B                #: molar gas constant, J/(mol K)
eV = e_charge               #: one electronvolt, J

#: unified atomic mass unit (atomic mass constant), kg
u = 1.66053906660e-27

#: Stefan-Boltzmann constant sigma = pi^2 k_B^4 / (60 hbar^3 c^2), W m^-2 K^-4
sigma_SB = (np.pi ** 2 * k_B ** 4) / (60.0 * hbar ** 3 * c ** 2)

#: radiation constant a = 4 sigma / c, J m^-3 K^-4
a_rad = 4.0 * sigma_SB / c

# --- Selected Riemann zeta values used by the Bose-gas examples -------------
ZETA_3_2 = 2.6123753486854883   #: zeta(3/2)
ZETA_5_2 = 1.3414872572509171   #: zeta(5/2)

# CODATA reference values, useful as verification targets.
SIGMA_SB_CODATA = 5.670374419e-8      #: CODATA Stefan-Boltzmann constant
WIEN_B_CODATA = 2.897771955e-3        #: CODATA Wien displacement constant, m K

__all__ = [
    "k_B", "h", "hbar", "c", "N_A", "R", "e_charge", "eV", "u",
    "sigma_SB", "a_rad", "ZETA_3_2", "ZETA_5_2",
    "SIGMA_SB_CODATA", "WIEN_B_CODATA",
]
