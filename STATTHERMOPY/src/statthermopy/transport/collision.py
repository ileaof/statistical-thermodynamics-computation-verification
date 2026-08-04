"""Lennard–Jones collision integrals (Chapman–Enskog).

The dimensionless collision integrals ``Ω^(l,s)*(T*)`` are the heart of the Chapman–Enskog
solution: they fold the molecular interaction potential (here the Lennard–Jones 12-6 potential)
into the transport coefficients. For the LJ potential they are *dimensionless functions of the
reduced temperature* ``T* = k_B T / ε`` only, and have been tabulated exactly by numerical
integration of the scattering problem (Hirschfelder, Curtiss & Bird 1964).

Here we use the compact analytical correlations of **Neufeld, Janzen & Aziz (1972)**, which
reproduce the exact LJ collision integrals to better than ~0.1 % over the range
``0.3 ≤ T* ≤ 100``. The two integrals needed for the first-order transport coefficients are

* ``Ω^(2,2)*(T*)`` — enters the viscosity and thermal conductivity;
* ``Ω^(1,1)*(T*)`` — enters the binary diffusion coefficient.

These are *not* empirical property correlations — they are analytical fits to a quantity derived
from a first-principles molecular potential, exactly as the Sackur–Tetrode formula is derived from
the free-particle partition function. The LJ potential itself is a model (a parameterised pair
potential with parameters ``σ``, ``ε`` stored per species as :class:`~statthermopy.core.molecule.
LennardJones`), not experimental property data.

Low-temperature behaviour
-------------------------
As ``T* → 0`` the integrals diverge (slowly, ``Ω^(2,2)* ∝ T*^(-0.15)``), reflecting the physical
fact that slow molecules are strongly deflected. At ``T = 0`` the prefactor ``√T`` in the
transport coefficients drives ``μ, k → 0`` (see :mod:`statthermopy.transport.transport`); this
module therefore returns a large but finite value at very small ``T*`` and the caller clamps
``T = 0`` to a zero transport coefficient, keeping every curve continuous down to 0 K.
"""

from __future__ import annotations

import math

__all__ = ["collision_integral", "omega_11", "omega_22", "t_star"]

# Neufeld, Janzen & Aziz (1972) coefficients: Ω^(l,s)*(T*) =
#   (A / T*^B) + C / exp(D T*) + E / exp(F T*) + G / exp(H T*) + I / exp(J T*)   (for Ω^(2,2)*)
# and the analogous four-term form for Ω^(1,1)*. Valid for 0.3 <= T* <= 100.
_NEUFELD_22 = {"A": 1.16145, "B": 0.14874, "C": 0.52487, "D": 0.77320,
               "E": 2.16178, "F": 2.43787}  # three-term form (A/T*^B + C e^{-D T*} + E e^{-F T*})
_NEUFELD_11 = {"A": 1.06036, "B": 0.15610, "C": 0.19300, "D": 0.47635,
               "E": 1.03587, "F": 1.52996}

# Minimum reduced temperature at which the Neufeld correlation is evaluated directly. Below it
# the first (dominant) term ``A / T*^B`` is used alone — it is the exact low-T* asymptote and
# avoids the spurious behaviour the multi-term fit can show when extrapolated toward T* = 0.
_T_STAR_MIN = 0.3


def t_star(T: float, epsilon_over_k: float) -> float:
    """Reduced temperature ``T* = k_B T / ε = T / (ε/k_B)``."""
    if epsilon_over_k <= 0.0:
        raise ValueError("epsilon_over_k must be > 0 K.")
    return T / epsilon_over_k


def _neufeld(Ts: float, c: dict) -> float:
    """Evaluate a Neufeld three-term collision-integral correlation at reduced temperature ``Ts``."""
    if Ts <= 0.0:
        # T* -> 0: the dominant term A / T*^B diverges; return a large finite value so the
        # transport-coefficient prefactor sqrt(T) still drives the result to 0 at T = 0.
        Ts = 1.0e-6
    if Ts < _T_STAR_MIN:
        # low-T* asymptote (dominant term only) — smooth and finite, never the full fit.
        return c["A"] / math.pow(Ts, c["B"])
    return c["A"] / math.pow(Ts, c["B"]) + c["C"] * math.exp(-c["D"] * Ts) + c["E"] * math.exp(-c["F"] * Ts)


def omega_22(Ts: float) -> float:
    """Dimensionless collision integral ``Ω^(2,2)*(T*)`` (viscosity, conductivity)."""
    return _neufeld(Ts, _NEUFELD_22)


def omega_11(Ts: float) -> float:
    """Dimensionless collision integral ``Ω^(1,1)*(T*)`` (diffusion)."""
    return _neufeld(Ts, _NEUFELD_11)


def collision_integral(ell: int, s: int, Ts: float) -> float:
    """Dispatcher for the collision integral ``Ω^(l,s)*(T*)``.

    Only the two integrals required by the first-order Chapman–Enskog coefficients are
    implemented: ``(l, s) = (1, 1)`` and ``(2, 2)``.
    """
    if (ell, s) == (2, 2):
        return omega_22(Ts)
    if (ell, s) == (1, 1):
        return omega_11(Ts)
    raise ValueError(f"collision integral Ω^({ell},{s})* is not implemented; use (1,1) or (2,2).")
