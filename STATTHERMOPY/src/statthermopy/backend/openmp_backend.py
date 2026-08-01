"""OpenMP-style CPU backend (multi-threaded Numba).

:class:`OpenMPBackend` is :class:`~statthermopy.backend.NumbaBackend` with the temperature-batched
molar-property kernel parallelised over the temperature grid using Numba's ``prange``
(``@njit(parallel=True)``). Pure Python has no portable OpenMP pragma; the honest equivalent is
Numba's parallel CPU compilation, which lowers ``prange`` to OpenMP-style threaded loops.

The per-temperature physics is shared with the Numba backend (a single ``device=True`` function),
so there is no physics duplication. The quantum rotational ``J`` sum inherits the Numba ``@njit``
implementation (it is already fast; parallelising its reduction adds little).
"""

from __future__ import annotations

import math

import numpy as np

from .numba_backend import (
    NumbaBackend,
    _extract_spec,
    _has_internal_rotors,
    _import_numba,
    _kernels,
)

__all__ = ["OpenMPBackend"]


_PARALLEL_KERNEL = None


def _parallel_kernel():
    global _PARALLEL_KERNEL
    if _PARALLEL_KERNEL is None:
        numba = _import_numba()
        njit = numba.njit
        prange = numba.prange
        _, props_at_T, _ = _kernels()

        @njit(cache=True, parallel=True)
        def _molar_props_jit_parallel(geometry, symmetry, mass, theta_rot, use_quantum, cutoff,
                                       theta_v, deg_v, theta_e, g_e, T_arr, P, R, N_A, kB, h):  # pragma: no cover (compiled by Numba)
            nT = T_arr.shape[0]
            U_m = np.empty(nT, dtype=np.float64)
            S_m = np.empty(nT, dtype=np.float64)
            A_m = np.empty(nT, dtype=np.float64)
            Cv_m = np.empty(nT, dtype=np.float64)
            lnQt = np.empty(nT, dtype=np.float64)
            lnQr = np.empty(nT, dtype=np.float64)
            lnQv = np.empty(nT, dtype=np.float64)
            lnQe = np.empty(nT, dtype=np.float64)

            coeff_t = 2.0 * math.pi * mass * kB / (h * h)
            lnNA = math.log(N_A)
            nv = theta_v.shape[0]
            ne = theta_e.shape[0]

            for i in prange(nT):
                T = T_arr[i]
                U, S, A, Cv, lQt, lQr, lQv, lQe = props_at_T(
                    geometry, symmetry, mass, theta_rot, use_quantum, cutoff,
                    theta_v, deg_v, theta_e, g_e, T, P, R, N_A, kB, h,
                    coeff_t, lnNA, nv, ne,
                )
                U_m[i] = U
                S_m[i] = S
                A_m[i] = A
                Cv_m[i] = Cv
                lnQt[i] = lQt
                lnQr[i] = lQr
                lnQv[i] = lQv
                lnQe[i] = lQe

            return U_m, S_m, A_m, Cv_m, lnQt, lnQr, lnQv, lnQe

        _PARALLEL_KERNEL = _molar_props_jit_parallel
    return _PARALLEL_KERNEL


class OpenMPBackend(NumbaBackend):
    """Multi-threaded Numba CPU backend (``name = "openmp"``).

    Identical results to :class:`NumbaBackend`; the temperature grid is processed in parallel with
    ``prange`` (Numba's OpenMP-style parallel loops).
    """

    name = "openmp"

    def molar_property_grid(self, mol, T_array, P, use_quantum, cutoff=150):
        from ..constants import N_A, R, h, k_B

        if _has_internal_rotors(mol):
            return None  # internal rotors aren't in the kernel; use the per-T Python path
        kernel = _parallel_kernel()
        spec = _extract_spec(mol)
        T_arr = np.asarray(T_array, dtype=np.float64)
        U_m, S_m, A_m, Cv_m, lnQt, lnQr, lnQv, lnQe = kernel(
            spec["geometry"], spec["symmetry"], spec["mass"], spec["theta_rot"],
            int(bool(use_quantum)), int(cutoff),
            spec["theta_v"], spec["deg_v"], spec["theta_e"], spec["g_e"],
            T_arr, float(P), float(R), float(N_A), float(k_B), float(h),
        )
        return {
            "U_m": U_m, "S_m": S_m, "A_m": A_m, "Cv_m": Cv_m,
            "ln_Qt": lnQt, "ln_Qr": lnQr, "ln_Qv": lnQv, "ln_Qe": lnQe,
        }