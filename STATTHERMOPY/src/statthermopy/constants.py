"""Physical constants used throughout StatThermoPy.

All values are CODATA 2018 / recommended SI values. They are the *only* external numbers
used in the calculations: together with the spectroscopic constants stored in the molecular
database they are the inputs from which every thermodynamic property is derived via the
molecular partition function. No empirical property correlations are used.
"""

from __future__ import annotations

import numpy as np

# --- Fundamental constants (CODATA 2018) -------------------------------------

#: Avogadro number (mol^-1).
N_A: float = 6.02214076e23
#: Boltzmann constant (J K^-1).
k_B: float = 1.380649e-23
#: Planck constant (J s).
h: float = 6.62607015e-34
#: Speed of light in vacuum (m s^-1).
c: float = 2.99792458e8
#: Molar gas constant (J mol^-1 K^-1). R = N_A * k_B (exact by definition).
R: float = N_A * k_B
#: Atomic mass unit (kg).
amu: float = 1.66053906660e-27
#: Planck constant times speed of light (J m), i.e. h*c, handy for wavenumber conversions.
hc: float = h * c

# --- Reference states --------------------------------------------------------

#: Standard pressure (Pa).
P_STP: float = 101325.0
#: Standard (reference) temperature (K).
T_STP: float = 298.15

# Numpy scalar forms for vectorised code.
_NA = np.float64(N_A)
_kB = np.float64(k_B)
_h = np.float64(h)
_c = np.float64(c)
_R = np.float64(R)
_amu = np.float64(amu)
_hc = np.float64(hc)


__all__ = [
    "N_A", "k_B", "h", "c", "R", "amu", "hc",
    "P_STP", "T_STP",
]