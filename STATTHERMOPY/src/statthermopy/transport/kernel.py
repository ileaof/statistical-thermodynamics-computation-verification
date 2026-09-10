"""Vectorised transport kernel — the array path for CFD-style evaluation.

:class:`~statthermopy.transport.air.MixtureTransportCalculator` is the reference implementation:
object-oriented, validated, one state at a time. It is the right tool interactively and the
authority on the physics, but it costs ~13 ms for a 30-component point — a million CFD cells
would take hours.

This module keeps *exactly the same physics* and reorganises the data. Everything that does not
depend on the local state is hoisted into the constructor:

* per-species arrays — ``M``, ``σ``, ``ε/k``, ``δ`` (the Stockmayer reduced dipole, ``0`` for a
  non-polar species, which reduces the collision integrals to plain Lennard-Jones);
* per-pair arrays — ``σ_ij²``, ``ε_ij``, reduced masses, and the mass-only halves of the Wilke
  factor, which never change;
* a ``C_v,i(T)`` table built once from the partition-function engine, so the quantum mode sums
  never run inside the loop.

The call then evaluates whole arrays of cells at once with NumPy broadcasting: ``T`` and ``P``
of shape ``(...)``, mole fractions ``X`` of shape ``(..., n_species)``. No YAML is read, no
:class:`~statthermopy.core.molecule.Molecule` is built, no dictionary is keyed by a string and
no per-cell Python object is created.

The physics is unchanged, so results match the reference path to the interpolation tolerance of
the ``C_v`` table (a few parts in 10⁻⁶ with the default grid); ``tests/test_transport_kernel.py``
asserts that directly.

Layout is deliberately array-of-structs-free and free of Python control flow over cells, so the
same code runs under CuPy or a Numba kernel by swapping the array module.

Example
-------
::

    kernel = TransportKernel(["N2", "O2", "Ar", "CO2", "H2O"])
    T = np.full((256, 256), 300.0)
    P = np.full((256, 256), 101325.0)
    X = np.broadcast_to([0.76, 0.20, 0.009, 0.0004, 0.03], (256, 256, 5))
    out = kernel(T, P, X)
    out["mu"].shape        # (256, 256)
    out["D_im"].shape      # (256, 256, 5)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..constants import R, k_B
from ..core.state import State
from ..database import get
from ..thermodynamics import Thermodynamics
from .collision import _NEUFELD_11, _NEUFELD_22, _T_STAR_MIN

__all__ = ["TransportKernel"]


def _omega(Ts, coeff: dict, polar_coeff: float, delta):
    """Vectorised Neufeld collision integral with Brokaw's dipole correction.

    Mirrors :func:`~statthermopy.transport.collision.omega_22` /
    :func:`~statthermopy.transport.collision.omega_11` element-wise, including the low-``T*``
    branch where only the dominant ``A/T*^B`` term is kept.
    """
    Ts = np.maximum(Ts, 1.0e-6)
    dominant = coeff["A"] / np.power(Ts, coeff["B"])
    full = (
        dominant
        + coeff["C"] * np.exp(-coeff["D"] * Ts)
        + coeff["E"] * np.exp(-coeff["F"] * Ts)
    )
    base = np.where(Ts < _T_STAR_MIN, dominant, full)
    return base + polar_coeff * delta * delta / Ts


class TransportKernel:
    """Pre-loaded, vectorised transport evaluator for a fixed set of species.

    Parameters
    ----------
    species : sequence of str
        Species names, in the order the mole-fraction axis will use. Any name the molecular
        database knows is accepted; the order is fixed for the life of the kernel.
    T_min, T_max : float
        Range of the tabulated ``C_v,i(T)``. Evaluating outside it is refused rather than
        silently extrapolated.
    n_T : int
        Number of tabulation points. The default resolves ``C_v`` to a few parts in 10⁻⁶.

    Attributes
    ----------
    species : tuple[str, ...]
        The species names, in axis order.
    M : numpy.ndarray
        Molar masses (kg/mol), shape ``(n,)``.
    """

    def __init__(
        self,
        species: Sequence[str],
        *,
        T_min: float = 200.0,
        T_max: float = 3000.0,
        n_T: int = 1024,
    ) -> None:
        names = [str(s) for s in species]
        if not names:
            raise ValueError("At least one species is required.")
        if T_max <= T_min:
            raise ValueError("T_max must exceed T_min.")
        mols = [get(n) for n in names]
        self.species: tuple[str, ...] = tuple(m.name for m in mols)
        self._mols = mols
        self.T_min = float(T_min)
        self.T_max = float(T_max)

        n = len(mols)
        self.n_species = n

        # --- per-species constants -------------------------------------------
        self.M = np.array([m.molar_mass for m in mols])                  # kg/mol
        self._mass = np.array([m.molecular_mass for m in mols])           # kg
        sigma = np.empty(n)
        eps = np.empty(n)
        delta = np.zeros(n)
        for i, m in enumerate(mols):
            sm = m.stockmayer
            if sm is not None:
                sigma[i], eps[i], delta[i] = sm.sigma_m, sm.epsilon_over_k, sm.reduced_dipole
            else:
                lj = m.lennard_jones
                if lj is None:
                    raise ValueError(f"{m.name} has no Lennard-Jones parameters.")
                sigma[i], eps[i] = lj.sigma_m, lj.epsilon_over_k
        self._sigma = sigma
        self._eps = eps
        self._delta = delta

        # Mason-Monchick inputs. Species without a measured Z_rot keep the Eucken correlation,
        # so `_mm_mask` selects between the two without any per-cell branch.
        from ..core.molecule import Geometry, _parker

        self._mm_mask = np.array(
            [m.rotational_relaxation is not None for m in mols], dtype=bool
        )
        self._z298 = np.array(
            [m.rotational_relaxation.z_rot_298 if m.rotational_relaxation else 1.0 for m in mols]
        )
        self._parker_298 = np.array([_parker(e, 298.15) for e in eps])
        cv_rot = np.zeros(n)
        for i, m in enumerate(mols):
            if m.geometry is Geometry.LINEAR:
                cv_rot[i] = R
            elif m.geometry is Geometry.NONLINEAR:
                cv_rot[i] = 1.5 * R
        self._cv_rot = cv_rot
        self._is_mono = np.array(
            [m.geometry is Geometry.MONOATOMIC for m in mols], dtype=bool
        )

        # --- per-pair constants (never change) --------------------------------
        # Diffusion always uses the Lennard-Jones set, matching binary_diffusion(); the polar
        # refinement is applied to the pure-species coefficients only.
        lj_sigma = np.array([m.lennard_jones.sigma_m for m in mols])
        lj_eps = np.array([m.lennard_jones.epsilon_over_k for m in mols])
        s_ij = 0.5 * (lj_sigma[:, None] + lj_sigma[None, :])
        self._sigma_ij_sq = s_ij * s_ij
        self._eps_ij = np.sqrt(lj_eps[:, None] * lj_eps[None, :])
        mi, mj = self._mass[:, None], self._mass[None, :]
        self._m_ij = mi * mj / (mi + mj)
        self._diff_pref = (3.0 / 16.0) / self._sigma_ij_sq * np.sqrt(
            2.0 * k_B / (np.pi * self._m_ij)
        )

        # Wilke/Mason-Saxena: the mass-only halves, hoisted out of the loop.
        Mi, Mj = self.M[:, None], self.M[None, :]
        self._wilke_mass = np.power(Mj / Mi, 0.25)
        self._wilke_denom = np.sqrt(8.0 * (1.0 + Mi / Mj))

        self._visc_pref = (5.0 / 16.0) * np.sqrt(self._mass * k_B / np.pi) / (sigma * sigma)

        # --- tabulated heat capacities ---------------------------------------
        self._T_grid = np.linspace(self.T_min, self.T_max, int(n_T))
        cv = np.empty((int(n_T), n))
        for i, m in enumerate(mols):
            for j, T in enumerate(self._T_grid):
                cv[j, i] = Thermodynamics(m, State(T=float(T), P=101325.0)).compute().Cv_m
        self._cv_table = cv                                    # J/mol/K

    # -- helpers ---------------------------------------------------------------

    def _cv_of(self, T):
        """Interpolate ``C_v,i(T)`` (J/mol/K) for every species; shape ``(..., n)``."""
        T = np.asarray(T, dtype=float)
        if np.any(self.T_min > T) or np.any(self.T_max < T):
            raise ValueError(
                f"Temperature outside the tabulated range [{self.T_min}, {self.T_max}] K. "
                "Rebuild the kernel with a wider range rather than extrapolating."
            )
        flat = T.reshape(-1)
        out = np.empty((flat.size, self.n_species))
        for i in range(self.n_species):
            out[:, i] = np.interp(flat, self._T_grid, self._cv_table[:, i])
        return out.reshape(T.shape + (self.n_species,))

    # -- evaluation ------------------------------------------------------------

    def __call__(self, T, P, X) -> dict:
        """Evaluate mixture transport for whole arrays of cells.

        Parameters
        ----------
        T, P : array_like
            Temperature (K) and pressure (Pa), any broadcastable shape ``(...)``.
        X : array_like
            Mole fractions, shape ``(..., n_species)``, in the kernel's species order. Rows are
            normalised internally; a zero entry simply drops out of every sum.

        Returns
        -------
        dict
            ``mu``, ``k``, ``rho``, ``nu``, ``alpha``, ``Pr``, ``M_mix``, ``R_specific``,
            ``cp_s``, ``cv_s``, ``gamma``, ``a`` with shape ``(...)``, plus ``mu_i``, ``k_i``,
            ``D_im``, ``Sc_i``, ``Le_i`` with shape ``(..., n_species)``.
        """
        T = np.asarray(T, dtype=float)
        P = np.asarray(P, dtype=float)
        X = np.asarray(X, dtype=float)
        if X.shape[-1] != self.n_species:
            raise ValueError(
                f"X last axis is {X.shape[-1]}, expected {self.n_species} "
                f"({', '.join(self.species)})."
            )
        if np.any(X < 0.0):
            raise ValueError("Mole fractions must be >= 0.")
        total = X.sum(axis=-1, keepdims=True)
        if np.any(total <= 0.0):
            raise ValueError("Every cell needs at least one species with a positive fraction.")
        x = X / total

        T_c = T[..., None]                                     # (..., 1)

        # --- pure-species coefficients ---------------------------------------
        Ts = T_c / self._eps                                   # (..., n)
        o22 = _omega(Ts, _NEUFELD_22, 0.2, self._delta)
        mu_i = self._visc_pref * np.sqrt(T_c) / o22            # Pa·s

        cv_m = self._cv_of(T)                                  # J/mol/K, (..., n)

        # --- binary diffusion matrix (the diagonal feeds Mason-Monchick) ------
        Ts_ij = T[..., None, None] / self._eps_ij              # (..., n, n)
        o11_ij = _omega(Ts_ij, _NEUFELD_11, 0.19, 0.0)
        D_ij = (
            self._diff_pref
            * (k_B * T[..., None, None] / P[..., None, None])
            * np.power(T[..., None, None], 0.5)
            / o11_ij
        )
        D_ii = np.diagonal(D_ij, axis1=-2, axis2=-1)           # (..., n)

        k_i = self._conductivity(T, P, mu_i, cv_m, D_ii)

        # --- Wilke / Mason-Saxena --------------------------------------------
        mu_mix = self._wilke(x, mu_i)
        k_mix = self._wilke(x, k_i)
        D_im = self._blanc(x, D_ij)

        # --- mixture bulk ------------------------------------------------------
        M_mix = (x * self.M).sum(axis=-1)
        cv_mix_m = (x * cv_m).sum(axis=-1)
        cp_mix_m = cv_mix_m + R
        gamma = cp_mix_m / cv_mix_m
        cp_s = cp_mix_m / M_mix
        cv_s = cv_mix_m / M_mix
        R_specific = R / M_mix
        rho = P * M_mix / (R * T)
        nu = mu_mix / rho
        alpha = k_mix / (rho * cp_s)
        Pr = mu_mix * cp_s / k_mix
        a = np.sqrt(gamma * R_specific * T)

        with np.errstate(divide="ignore", invalid="ignore"):
            Sc_i = np.where(D_im > 0.0, nu[..., None] / D_im, 0.0)
            Le_i = np.where(D_im > 0.0, alpha[..., None] / D_im, 0.0)

        return {
            "mu": mu_mix, "k": k_mix, "rho": rho, "nu": nu, "alpha": alpha, "Pr": Pr,
            "M_mix": M_mix, "R_specific": R_specific, "cp_s": cp_s, "cv_s": cv_s,
            "gamma": gamma, "a": a,
            "mu_i": mu_i, "k_i": k_i, "D_im": D_im, "Sc_i": Sc_i, "Le_i": Le_i,
        }

    def _conductivity(self, T, P, mu_i, cv_m, D_ii):
        """Per-species ``k_i``: Eucken, or Mason-Monchick where a measured ``Z_rot`` exists.

        Both branches are evaluated on the whole array and selected by a precomputed mask, so
        there is no per-cell Python branch. Mirrors
        :meth:`~statthermopy.transport.transport.TransportCalculator.conductivity` exactly.
        """
        cp_m = cv_m + R
        gamma_i = cp_m / cv_m
        eucken = mu_i * (cv_m / self.M) * (9.0 * gamma_i - 5.0) / 4.0

        if not self._mm_mask.any():
            return eucken

        Cv_tr = 1.5 * R
        cv_rot = self._cv_rot
        cv_vib = np.maximum(cv_m - Cv_tr - cv_rot, 0.0)
        rho_i = P[..., None] * self.M / (R * T[..., None])
        with np.errstate(divide="ignore", invalid="ignore"):
            rhoD_mu = np.where(mu_i > 0.0, rho_i * D_ii / mu_i, 0.0)

        # Parker scaling of Z_rot, elementwise in T.
        x_par = self._eps / T[..., None]
        parker_T = (
            1.0
            + (np.pi ** 1.5 / 2.0) * np.sqrt(x_par)
            + (np.pi ** 2 / 4.0 + 2.0) * x_par
            + (np.pi ** 1.5) * np.power(x_par, 1.5)
        )
        Z = self._z298 * self._parker_298 / parker_T

        A = 2.5 - rhoD_mu
        B = Z + (2.0 / np.pi) * ((5.0 / 3.0) * (cv_rot / R) + rhoD_mu)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(B != 0.0, A / B, 0.0)
        f_tr = 2.5 * (1.0 - (2.0 / np.pi) * np.where(Cv_tr > 0, cv_rot / Cv_tr, 0.0) * ratio)
        f_rot = rhoD_mu * (1.0 + (2.0 / np.pi) * ratio)
        mm = (mu_i / self.M) * (f_tr * Cv_tr + f_rot * cv_rot + rhoD_mu * cv_vib)
        # A monatomic species has no internal modes; the expression collapses to the exact
        # Chapman-Enskog result, which is what Eucken already gives at gamma = 5/3.
        mm = np.where(self._is_mono, (mu_i / self.M) * 2.5 * Cv_tr, mm)
        return np.where(self._mm_mask, mm, eucken)

    # -- mixing rules (vectorised) ---------------------------------------------

    def _wilke(self, x, value_i):
        """Wilke / Mason-Saxena sum ``Σ_i x_i v_i / Σ_j x_j φ_ij``, over the last axis.

        ``φ_ij = [1 + (v_i/v_j)^½ (M_j/M_i)^¼]² / √(8(1 + M_i/M_j))`` — identical algebra to
        :func:`~statthermopy.transport.air.mixture_transport.wilke_viscosity`, with the
        mass-only factors precomputed.
        """
        ratio = np.sqrt(value_i[..., :, None] / value_i[..., None, :])   # (..., n, n)
        phi = (1.0 + ratio * self._wilke_mass) ** 2 / self._wilke_denom
        denom = (x[..., None, :] * phi).sum(axis=-1)                     # (..., n)
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = np.where(denom > 0.0, x * value_i / denom, 0.0)
        return contrib.sum(axis=-1)

    @staticmethod
    def _blanc(x, D_ij):
        """Blanc's law ``D_i,m = (1 − x_i) / Σ_{j≠i} (x_j / D_ij)``, over the last axis."""
        n = x.shape[-1]
        eye = np.eye(n, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(D_ij > 0.0, 1.0 / D_ij, 0.0)
        inv = np.where(eye, 0.0, inv)                       # drop the i = j term
        s = (x[..., None, :] * inv).sum(axis=-1)            # (..., n)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0.0, (1.0 - x) / s, 0.0)
