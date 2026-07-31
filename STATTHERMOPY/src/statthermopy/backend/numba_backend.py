"""Numba-accelerated CPU backend.

:class:`NumbaBackend` provides the same array operations as :class:`~statthermopy.backend.
NumpyBackend` (delegating to NumPy — the mode arrays are small and a vectorised ufunc would only
add overhead) and overrides the two high-level kernels with ``@njit`` compiled code:

* :meth:`linear_quantum_moments` — the quantum linear-rotor ``J`` sum, a scalar loop that is the
  natural ``@njit`` target;
* :meth:`molar_property_grid` — the whole molar property set over a temperature grid, encoded in
  one ``@njit`` pass.

The kernel physics mirrors :mod:`statthermopy.modes` exactly (translation Sackur–Tetrode, rigid
rotor classical/quantum, quantum harmonic oscillator, electronic Boltzmann sum); correctness is
guarded by ``tests/test_backend.py`` which checks agreement with the NumPy path to ~1e-6.

numba is imported lazily — selecting this backend by name triggers the import. ``import
statthermopy`` never pulls in numba.
"""

from __future__ import annotations

import math

import numpy as np

from .executor import Backend

__all__ = ["NumbaBackend"]


# ---------------------------------------------------------------------------
# Spectroscopic spec extraction (shared with OpenMP/CUDA backends)
# ---------------------------------------------------------------------------

def _extract_spec(mol) -> dict:
    """Flatten a :class:`~statthermopy.core.molecule.Molecule` into Numba-friendly arrays/scalars.

    The constants are folded into the characteristic temperatures here (NumPy, on the host) so the
    kernels receive plain float64 arrays. The formulas mirror
    :mod:`statthermopy.modes.rotational`/`vibrational`/`electronic`.
    """
    from ..constants import h, k_B  # noqa: F401  (h, k_B used below)
    from ..units import CM1_TO_K

    if mol.is_monoatomic:
        geometry = 0
    elif mol.is_linear:
        geometry = 1
    else:
        geometry = 2

    # Rotational characteristic temperatures: theta = h^2 / (8 pi^2 I k_B).
    theta_rot = np.array(
        [(h * h) / (8.0 * math.pi * math.pi * I * k_B) for I in mol.moments_of_inertia],
        dtype=np.float64,
    )
    theta_v = np.array(
        [m.wavenumber_cm1 * CM1_TO_K for m in mol.vibrational_modes], dtype=np.float64
    )
    deg_v = np.array([m.degeneracy for m in mol.vibrational_modes], dtype=np.float64)
    theta_e = np.array(
        [lvl.energy_cm1 * CM1_TO_K for lvl in mol.electronic_levels], dtype=np.float64
    )
    g_e = np.array([lvl.degeneracy for lvl in mol.electronic_levels], dtype=np.float64)
    return {
        "geometry": geometry,
        "symmetry": int(mol.symmetry_number),
        "mass": float(mol.molecular_mass),
        "theta_rot": theta_rot,
        "theta_v": theta_v,
        "deg_v": deg_v,
        "theta_e": theta_e,
        "g_e": g_e,
    }


# ---------------------------------------------------------------------------
# @njit kernels (module-level so Numba can compile and cache them)
# ---------------------------------------------------------------------------

def _import_numba():
    try:
        import numba
    except ImportError as exc:  # pragma: no cover - message path
        raise ImportError(
            "The 'numba' backend requires the optional 'accel' extra "
            "(pip install statthermopy[accel])."
        ) from exc
    return numba


def _make_kernels():
    """Build (and compile-on-first-call) the @njit kernels.

    Done lazily so that merely importing :mod:`statthermopy.backend.numba_backend` does not
    import numba — only instantiating/selecting :class:`NumbaBackend` does.

    The per-temperature physics lives in a single ``device=True`` function (``_props_at_T``) so
    that the serial and the OpenMP-parallel wrappers share one implementation — no physics
    duplication across backends.
    """
    numba = _import_numba()
    njit = numba.njit

    @njit(cache=True)
    def _q_rot_jit(theta_rot, T, cutoff):  # pragma: no cover (compiled by Numba)
        # Exact quantum linear-rotor J sum: (Q, <y>, <y^2>), y = J(J+1) theta/T.
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

    @njit(cache=True)
    def _props_at_T(geometry, symmetry, mass, theta_rot, use_quantum, cutoff,
                    theta_v, deg_v, theta_e, g_e, T, P, R, N_A, kB, h, coeff_t, lnNA, nv, ne):  # pragma: no cover (compiled by Numba)
        # Returns (U_m, S_m, A_m, Cv_m, lnQt, lnQr, lnQv, lnQe) for one temperature.
        lnQt_i = 1.5 * math.log(coeff_t * T) + math.log(R * T / P)
        U_t = 1.5 * R * T
        Cv_t = 1.5 * R
        S_t = R * (lnQt_i + 1.5 - lnNA + 1.0)
        A_t = -R * T * (lnQt_i - lnNA + 1.0)

        if geometry == 0:  # monoatomic
            lnQr_i = 0.0
            U_r = 0.0
            Cv_r = 0.0
            S_r = 0.0
            A_r = 0.0
        elif geometry == 1:  # linear
            theta = theta_rot[0]
            if use_quantum:
                q, mean_y, mean_y2 = _q_rot_jit(theta, T, cutoff)
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
        else:  # nonlinear asymmetric top
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
            x = theta_v[k] / T
            ex = math.exp(x)
            e1 = math.expm1(x)
            g = deg_v[k]
            emx = math.exp(-x)
            l1p = math.log1p(-emx)
            lnQv_i += -g * l1p
            U_v += g * R * theta_v[k] / e1
            Cv_v += g * R * (x * x) * ex / (e1 * e1)
            S_v += g * R * (x / e1 - l1p)
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

    @njit(cache=True)
    def _molar_props_jit(geometry, symmetry, mass, theta_rot, use_quantum, cutoff,
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

        for i in range(nT):
            T = T_arr[i]
            U, S, A, Cv, lQt, lQr, lQv, lQe = _props_at_T(
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

    return _q_rot_jit, _props_at_T, _molar_props_jit


_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _make_kernels()
    return _KERNELS


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class NumbaBackend(Backend):
    """Numba ``@njit`` CPU backend (``name = "numba"``).

    Array operations delegate to NumPy; the two hot loops — the quantum rotational ``J`` sum and
    the temperature-batched molar property grid — are compiled with Numba.
    """

    name = "numba"

    # -- array operations (delegated to NumPy) ---------------------------------

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

    # -- high-level kernels ---------------------------------------------------

    def linear_quantum_moments(self, theta_rot, T, cutoff):
        q_rot_jit, _props_at_T, _molar_props_jit = _kernels()
        return q_rot_jit(float(theta_rot), float(T), int(cutoff))

    def molar_property_grid(self, mol, T_array, P, use_quantum, cutoff=150):
        from ..constants import N_A, R, h, k_B

        _q_rot_jit, _props_at_T, molar_props_jit = _kernels()
        spec = _extract_spec(mol)
        T_arr = np.asarray(T_array, dtype=np.float64)
        U_m, S_m, A_m, Cv_m, lnQt, lnQr, lnQv, lnQe = molar_props_jit(
            spec["geometry"], spec["symmetry"], spec["mass"], spec["theta_rot"],
            int(bool(use_quantum)), int(cutoff),
            spec["theta_v"], spec["deg_v"], spec["theta_e"], spec["g_e"],
            T_arr, float(P), float(R), float(N_A), float(k_B), float(h),
        )
        return {
            "U_m": U_m, "S_m": S_m, "A_m": A_m, "Cv_m": Cv_m,
            "ln_Qt": lnQt, "ln_Qr": lnQr, "ln_Qv": lnQv, "ln_Qe": lnQe,
        }