"""CUDA GPU backend (numba.cuda) with automatic CPU fallback.

:class:`CudaBackend` runs the temperature-batched molar-property kernel on an NVIDIA GPU via
``numba.cuda`` when one is available. When ``numba.cuda.is_available()`` is false it emits a
warning and transparently delegates to :class:`~statthermopy.backend.NumbaBackend`, so
``set_backend("cuda")`` never raises on a GPU-less machine — it just runs on the CPU.

The mode array operations delegate to NumPy (CPU) regardless: the mode arrays are tiny and the
GPU is only worthwhile for the large temperature grid, which is what the kernel accelerates. The
per-temperature physics mirrors :mod:`statthermopy.modes` exactly (the GPU kernel is the CUDA
analogue of the Numba ``@njit`` kernel in :mod:`statthermopy.backend.numba_backend`).

numba is imported lazily; ``import statthermopy`` never pulls in numba.cuda.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .executor import Backend
from .numba_backend import NumbaBackend, _extract_spec, _has_internal_rotors

__all__ = ["CudaBackend"]


def _cuda_available() -> bool:
    try:
        import numba.cuda

        return bool(numba.cuda.is_available())
    except Exception:  # pragma: no cover - numba.cuda absent or broken
        return False


def _build_cuda_kernel():  # pragma: no cover - requires an NVIDIA GPU to execute
    import numba
    from numba import cuda

    @cuda.jit(device=True)
    def _q_rot_cuda(theta_rot, T, cutoff):
        beta = theta_rot / T
        q = 0.0
        s1 = 0.0
        s2 = 0.0
        for J in range(cutoff + 1):
            y = J * (J + 1) * beta
            term = (2 * J + 1) * math.exp(-y)
            q += term
            s1 += term * y
            s2 += term * y * y
            if term < 1e-15 * q and J > 5:
                break
        if q == 0.0:
            q = 1.0
        return q, s1 / q, s2 / q

    @cuda.jit(device=True)
    def _props_at_T_cuda(geometry, symmetry, mass, theta_rot, use_quantum, cutoff,
                         theta_v, deg_v, theta_e, g_e, T, P, R, N_A, kB, h,
                         coeff_t, lnNA, nv, ne):
        # T = 0 limit (see numba_backend._props_at_T for the rationale): quantum modes
        # freeze (Cv -> 0), classical translation/rotation keep equipartition, U_m = 0 so
        # the thermal field T_v -> 0 stays finite, classical S_m -> -inf.
        if T == 0.0:
            if geometry == 0:
                cv0 = 1.5 * R
                lnQr0 = 0.0
            elif geometry == 1:
                if use_quantum:
                    cv0 = 1.5 * R
                    lnQr0 = 0.0
                else:
                    cv0 = 2.5 * R
                    lnQr0 = -float('inf')
            else:
                cv0 = 3.0 * R
                lnQr0 = -float('inf')
            qe0 = 0.0
            for k in range(ne):
                if theta_e[k] == 0.0:
                    qe0 += g_e[k]
            lnQe0 = math.log(qe0) if qe0 > 0.0 else 0.0
            return (0.0, -float('inf'), 0.0, cv0, -float('inf'), lnQr0, 0.0, lnQe0)

        lnQt_i = 1.5 * math.log(coeff_t * T) + math.log(R * T / P)
        U_t = 1.5 * R * T
        Cv_t = 1.5 * R
        S_t = R * (lnQt_i + 1.5 - lnNA + 1.0)
        A_t = -R * T * (lnQt_i - lnNA + 1.0)

        if geometry == 0:
            lnQr_i = 0.0
            U_r = 0.0
            Cv_r = 0.0
            S_r = 0.0
            A_r = 0.0
        elif geometry == 1:
            theta = theta_rot[0]
            if use_quantum:
                q, mean_y, mean_y2 = _q_rot_cuda(theta, T, cutoff)
                lnQr_i = math.log(q)
                dlnq = mean_y / T
                U_r = R * T * T * dlnq
                Cv_r = R * (mean_y2 - mean_y * mean_y)
                S_r = R * (lnQr_i + T * dlnq)
                A_r = -R * T * lnQr_i
            else:
                lnQr_i = math.log(T) - math.log(symmetry * theta)
                U_r = R * T
                Cv_r = R
                S_r = R * (lnQr_i + 1.0)
                A_r = -R * T * lnQr_i
        else:
            tA = theta_rot[0]
            tB = theta_rot[1]
            tC = theta_rot[2]
            lnQr_i = (0.5 * math.log(math.pi) - math.log(symmetry)
                      + 0.5 * math.log((T * T * T) / (tA * tB * tC)))
            U_r = 1.5 * R * T
            Cv_r = 1.5 * R
            S_r = R * (lnQr_i + 1.5)
            A_r = -R * T * lnQr_i

        lnQv_i = 0.0
        U_v = 0.0
        Cv_v = 0.0
        S_v = 0.0
        for k in range(nv):
            # Stable exp(-x) form (avoids exp(x) overflow for T < theta/709).
            x = theta_v[k] / T
            g = deg_v[k]
            emx = math.exp(-x)
            one = -math.expm1(-x)  # = 1 - exp(-x)
            l1p = math.log1p(-emx)
            lnQv_i += -g * l1p
            U_v += g * R * theta_v[k] * emx / one
            Cv_v += g * R * (x * x) * emx / (one * one)
            S_v += g * R * (x * emx / one - l1p)
        A_v = -R * T * lnQv_i

        qe = 0.0
        w_theta = 0.0
        w_theta2 = 0.0
        for k in range(ne):
            w = g_e[k] * math.exp(-theta_e[k] / T)
            qe += w
            w_theta += w * theta_e[k]
            w_theta2 += w * theta_e[k] * theta_e[k]
        lnQe_i = math.log(qe)
        mean_theta = w_theta / qe
        mean_theta2 = w_theta2 / qe
        U_e = R * mean_theta
        Cv_e = R * (mean_theta2 - mean_theta * mean_theta) / (T * T)
        S_e = R * (lnQe_i + mean_theta / T)
        A_e = -R * T * lnQe_i

        return (U_t + U_r + U_v + U_e,
                S_t + S_r + S_v + S_e,
                A_t + A_r + A_v + A_e,
                Cv_t + Cv_r + Cv_v + Cv_e,
                lnQt_i, lnQr_i, lnQv_i, lnQe_i)

    @cuda.jit
    def _molar_props_cuda(geometry, symmetry, mass, theta_rot, use_quantum, cutoff,
                          theta_v, deg_v, theta_e, g_e, T_arr, P, R, N_A, kB, h,
                          U_m, S_m, A_m, Cv_m, lnQt, lnQr, lnQv, lnQe):
        i = cuda.grid(1)
        nT = T_arr.shape[0]
        if i < nT:
            coeff_t = 2.0 * math.pi * mass * kB / (h * h)
            lnNA = math.log(N_A)
            nv = theta_v.shape[0]
            ne = theta_e.shape[0]
            T = T_arr[i]
            U, S, A, Cv, lQt, lQr, lQv, lQe = _props_at_T_cuda(
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

    return _molar_props_cuda


class CudaBackend(Backend):
    """CUDA GPU backend (``name = "cuda"``) with automatic CPU fallback.

    On a GPU-less machine, construction warns and delegates the kernels to a
    :class:`NumbaBackend`; results are identical to the Numba/CPU path.
    """

    name = "cuda"

    def __init__(self) -> None:
        # ``self._cpu`` is the fallback used for *every* kernel when no GPU is present; it stays
        # ``None`` on a GPU machine so :meth:`molar_property_grid` takes the GPU path. The quantum
        # J-sum is always run on CPU (it is a tiny scalar loop); a separate delegate is used so
        # that delegating it never flips the GPU flag.
        self._qsum: NumbaBackend | None = None
        if _cuda_available():  # pragma: no cover - GPU path (no NVIDIA GPU in CI)
            self._cpu: NumbaBackend | None = None
            self._kernel = _build_cuda_kernel()
        else:
            warnings.warn(
                "CudaBackend: no NVIDIA GPU available (numba.cuda.is_available() is False); "
                "falling back to the Numba CPU backend. Results are identical.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._cpu = NumbaBackend()
            self._kernel = None

    # -- array operations (always on CPU; mode arrays are tiny) ----------------

    def exp(self, x):
        return np.exp(x)

    def expm1(self, x):
        return np.expm1(x)

    def log(self, x):
        return np.log(x)

    def log1p(self, x):
        return np.log1p(x)

    def sum(self, x):
        return float(np.sum(x))

    def asarray(self, x):
        return np.asarray(x, dtype=float)

    # -- kernels --------------------------------------------------------------

    def linear_quantum_moments(self, theta_rot, T, cutoff):
        # The J-sum is a scalar loop; always run it on CPU (Numba), even with a GPU present.
        if self._qsum is None:
            self._qsum = NumbaBackend()
        return self._qsum.linear_quantum_moments(theta_rot, T, cutoff)

    def molar_property_grid(self, mol, T_array, P, use_quantum, cutoff=150):
        if _has_internal_rotors(mol):
            return None  # internal rotors aren't in the kernel; use the per-T Python path
        if self._cpu is not None:
            return self._cpu.molar_property_grid(mol, T_array, P, use_quantum, cutoff)
        return self._molar_property_grid_gpu(mol, T_array, P, use_quantum, cutoff)  # pragma: no cover - GPU

    def _molar_property_grid_gpu(self, mol, T_array, P, use_quantum, cutoff=150):  # pragma: no cover - GPU
        from numba import cuda
        from ..constants import N_A, R, h, k_B

        spec = _extract_spec(mol)
        T_arr = np.asarray(T_array, dtype=np.float64)
        nT = T_arr.shape[0]
        device_args = [
            spec["geometry"], spec["symmetry"], spec["mass"], spec["theta_rot"],
            int(bool(use_quantum)), int(cutoff),
            spec["theta_v"], spec["deg_v"], spec["theta_e"], spec["g_e"],
            cuda.to_device(T_arr), float(P), float(R), float(N_A), float(k_B), float(h),
        ]
        out = [cuda.to_device(np.empty(nT, dtype=np.float64)) for _ in range(8)]
        threads = 256
        blocks = (nT + threads - 1) // threads
        self._kernel[blocks, threads](*device_args, *out)
        U_m, S_m, A_m, Cv_m, lnQt, lnQr, lnQv, lnQe = [o.copy_to_host() for o in out]
        return {
            "U_m": U_m, "S_m": S_m, "A_m": A_m, "Cv_m": Cv_m,
            "ln_Qt": lnQt, "ln_Qr": lnQr, "ln_Qv": lnQv, "ln_Qe": lnQe,
        }