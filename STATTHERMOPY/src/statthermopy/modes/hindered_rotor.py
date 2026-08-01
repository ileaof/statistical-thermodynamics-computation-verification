"""Hindered internal-rotation contribution to the molecular partition function.

A one-dimensional internal rotor (e.g. a methyl-top torsion about a single bond) moves in an
``n``-fold symmetric hindering potential

    V(φ) = (V_n / 2) [1 - cos(n φ)] ,

with the torsional Hamiltonian

    Ĥ = -F d²/dφ² + (V_n / 2)[1 - cos(n φ)] ,   F = ħ² / (2 I_r) ,

where ``I_r`` is the reduced moment of inertia and ``F`` the internal-rotation constant. This is
the Mathieu equation; its eigenvalues are obtained **exactly** (to basis truncation) by
diagonalising ``Ĥ`` in the free-rotor basis ``|m⟩ = e^{i m φ} / √(2π)``, in which

    ⟨m|Ĥ|m⟩   = F m² + V_n / 2 ,
    ⟨m|Ĥ|m±n⟩ = -V_n / 4 .

From the torsional levels ``ε_i`` (measured from the ground level, so the zero of energy matches
the harmonic-oscillator convention ``v = 0``) the rotor partition function is

    q = (1/σ_int) Σ_i exp(-ε_i / k_B T) ,

with ``σ_int`` the internal symmetry number. This reproduces both limits automatically: as
``V_n → 0`` it becomes the free internal rotor ``q = (8π³ I_r k_B T)^{1/2} / (σ_int h)`` (heat
capacity → R/2), and as ``V_n → ∞`` it becomes a harmonic oscillator of the small-oscillation
frequency ``ṽ = n √(F V_n)`` (heat capacity → R). The thermodynamic functions follow from the
usual moments of the level distribution:

    U_m  = R T ⟨x⟩ ,
    Cv_m = R (⟨x²⟩ - ⟨x⟩²) ,        x_i = ε_i / (k_B T) ,
    S_m  = R (ln q + ⟨x⟩) ,
    A_m  = -R T ln q ,

summed over every rotor (a rotor of degeneracy ``d`` contributes ``d`` times).

Unlike the harmonic-oscillator physics, the torsional eigenvalues have no closed form, so this
mode is evaluated with NumPy and is **not** part of the accelerated ``molar_property_grid``
kernel — the accelerated backends fall back to the per-temperature Python path for molecules
that carry internal rotors (see :mod:`statthermopy.backend.numba_backend`). The engine remains
pure statistical mechanics: only the two spectroscopic constants ``F`` and ``V_n`` enter, no
empirical property correlation.
"""

from __future__ import annotations

import numpy as np

from ..constants import R
from ..core.contribution import Contribution
from ..core.molecule import InternalRotor
from ..core.state import ResolvedState
from ..units import CM1_TO_K
from .base import Mode

__all__ = ["HinderedRotor", "torsional_levels_kelvin"]

#: Half-width of the free-rotor basis (m = -M..M, i.e. 2M+1 functions). Comfortably converges the
#: low-lying torsional levels for methyl-scale rotors up to several thousand K.
_BASIS_HALF_WIDTH: int = 100


def torsional_levels_kelvin(
    rotation_constant_cm1: float,
    barrier_cm1: float,
    n_minima: int,
    *,
    basis_half_width: int = _BASIS_HALF_WIDTH,
) -> np.ndarray:
    """Torsional eigenvalues (K), relative to the ground level, of one hindered rotor.

    Solves the Mathieu Hamiltonian in the free-rotor basis and returns the ascending eigenvalue
    ladder shifted so the lowest level sits at 0 K (matching the harmonic ``v = 0`` reference).
    """
    F = rotation_constant_cm1 * CM1_TO_K       # internal-rotation constant in K
    V = barrier_cm1 * CM1_TO_K                 # barrier height in K
    m = np.arange(-basis_half_width, basis_half_width + 1, dtype=float)
    H = np.diag(F * m * m + V / 2.0)
    # Off-diagonal coupling ⟨m|V|m±n⟩ = -V/4 links basis functions n apart.
    idx = np.arange(m.size - n_minima)
    H[idx, idx + n_minima] = -V / 4.0
    H[idx + n_minima, idx] = -V / 4.0
    eig = np.linalg.eigvalsh(H)
    return eig - eig[0]


class HinderedRotor(Mode):
    """Sum of one-dimensional hindered internal rotors.

    Parameters
    ----------
    rotors : tuple[InternalRotor, ...]
        The internal rotors. An empty tuple makes this a null mode (``ln_q = 0``, no
        contribution) so molecules without internal rotation are unaffected.
    """

    name = "internal_rotation"

    def __init__(self, rotors: tuple[InternalRotor, ...]) -> None:
        self.rotors = tuple(rotors)
        # Precompute each rotor's torsional ladder (K) once; only T varies afterwards.
        self._levels: list[tuple[np.ndarray, int, int]] = [
            (
                torsional_levels_kelvin(r.rotation_constant_cm1, r.barrier_cm1, r.n_minima),
                int(r.symmetry),
                int(r.degeneracy),
            )
            for r in self.rotors
        ]

    # -- per-rotor moments -----------------------------------------------------

    @staticmethod
    def _moments(levels: np.ndarray, symmetry: int, T: float):
        """Return ``(ln_q, <x>, var_x)`` for one rotor at temperature ``T``.

        ``x_i = ε_i / (k_B T)`` are the dimensionless level energies; ``ln_q`` already carries the
        internal symmetry number ``-ln σ_int``.
        """
        x = levels / T
        w = np.exp(-x)
        Z = float(w.sum())
        mean = float((x * w).sum() / Z)
        mean_sq = float((x * x * w).sum() / Z)
        ln_q = float(np.log(Z)) - float(np.log(symmetry))
        return ln_q, mean, mean_sq - mean * mean

    # -- Mode interface --------------------------------------------------------

    def ln_q(self, state: ResolvedState) -> float:
        total = 0.0
        for levels, sym, deg in self._levels:
            ln_q, _, _ = self._moments(levels, sym, state.T)
            total += deg * ln_q
        return total

    def d_ln_q_dT(self, state: ResolvedState) -> float:
        # d ln q / dT = <x> / T  (since U_m = R T <x> = R T² d ln q/dT).
        T = state.T
        total = 0.0
        for levels, sym, deg in self._levels:
            _, mean, _ = self._moments(levels, sym, T)
            total += deg * mean / T
        return total

    def cv_m(self, state: ResolvedState) -> float:
        T = state.T
        total = 0.0
        for levels, sym, deg in self._levels:
            _, _, var = self._moments(levels, sym, T)
            total += deg * R * var
        return total

    def contribution(self, state: ResolvedState) -> Contribution:
        if not self._levels:
            return Contribution(name=self.name, ln_q=0.0, d_ln_q_dT=0.0,
                                U_m=0.0, S_m=0.0, A_m=0.0, Cv_m=0.0)
        T = state.T
        ln_q = d_ln_q = U_m = S_m = A_m = Cv_m = 0.0
        for levels, sym, deg in self._levels:
            lq, mean, var = self._moments(levels, sym, T)
            ln_q += deg * lq
            d_ln_q += deg * mean / T
            U_m += deg * R * T * mean
            Cv_m += deg * R * var
            S_m += deg * R * (lq + mean)
            A_m += deg * (-R * T * lq)
        return Contribution(
            name=self.name, ln_q=ln_q, d_ln_q_dT=d_ln_q, U_m=U_m, S_m=S_m, A_m=A_m, Cv_m=Cv_m
        )
