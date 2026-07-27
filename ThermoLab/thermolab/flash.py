"""Variable-pair flash solver.

Resolves an arbitrary pair of independent state variables (from ``T, P, rho, v,
h, s, u``) into a fully determined thermodynamic state ``(T, P, phase)`` using
the backend's property functions. The solver is **backend-agnostic**: it only
calls ``BaseBackend`` methods, so every backend gets flash support for free.

Strategy:

* ``(T, P)`` -> direct.
* exactly one of ``{T, P}`` known  -> 1-D root find (``brentq``) for the other.
* neither ``T`` nor ``P`` known (e.g. ``rho, h`` or ``h, s``) -> 2-D ``fsolve``.

For **pure fluids**, the solver is *saturation-aware*: when an energy/entropy
spec lies between the saturated liquid and vapor values it returns a two-phase
state directly (with quality), and otherwise restricts the single-phase search
to the physically correct side of the saturation curve. This avoids asking the
backend to evaluate single-phase roots inside the dome, where its density
solver may fail to converge.

All public quantities are **mass-based** SI; conversions to/from the backend's
molar interface use the mixture molar mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, fsolve

from .backends.base import BaseBackend, Phase
from .exceptions import ConvergenceError, FlashSpecificationError
from .units import mixture_molar_mass

# Recognized specification variables.
_SPEC_VARS = ("T", "P", "rho", "v", "h", "s", "u")
_ENERGY_VARS = ("h", "s", "u")

# Default search bounds (K, Pa), narrowed per-fluid inside the solver.
_T_BOUNDS = (20.0, 6000.0)
_P_BOUNDS = (1.0, 1e9)


@dataclass
class FlashResult:
    """Resolved state from a flash."""

    T: float
    P: float
    phase: Phase
    two_phase: bool = False
    quality: float | None = None
    sat: object | None = None


def _normalize_specs(specs: dict) -> dict[str, float]:
    given = {k: float(v) for k, v in specs.items() if v is not None and k in _SPEC_VARS}
    if len(given) != 2:
        raise FlashSpecificationError(
            f"Exactly two independent variables from {_SPEC_VARS} are required; "
            f"got {sorted(given)}."
        )
    return given


def _bracket(f, x_lo: float, x_hi: float):
    """Find a sign-change interval for ``f`` within [x_lo, x_hi] then brentq.

    Only safe when both bounds are known to evaluate without aborting.
    """
    flo, fhi = f(x_lo), f(x_hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        raise ConvergenceError("Flash: non-finite residual at search bound.")
    if flo == 0.0:
        return x_lo
    if fhi == 0.0:
        return x_hi
    if np.sign(flo) != np.sign(fhi):
        return brentq(f, x_lo, x_hi, xtol=1e-10, rtol=1e-12, maxiter=200)
    raise ConvergenceError("Flash: bounds do not bracket a root.")


def _solve1d(f, guesses, *, lo=None, hi=None):
    """Solve ``f(x)=0`` via fsolve from a sequence of initial guesses.

    fsolve only evaluates ``f`` near its starting point, so it never probes the
    extreme temperatures/pressures that can make the backend's density solver
    abort the process. A small list of guesses gives robustness against
    non-convergence from a poor start. Returns the root with the smallest
    residual.
    """
    best_x, best_r = None, np.inf
    for g in guesses:
        if not np.isfinite(g):
            continue
        try:
            sol = fsolve(f, g, full_output=False, xtol=1e-12, maxfev=200)
            x = float(sol[0])
        except Exception:
            continue
        if not np.isfinite(x):
            continue
        if lo is not None:
            x = max(x, lo)
        if hi is not None:
            x = min(x, hi)
        try:
            r = abs(float(f(x)))
        except Exception:
            continue
        if not np.isfinite(r):
            continue
        if r < best_r:
            best_r, best_x = r, x
        if best_r < 1e-6:
            break
    if best_x is None or best_r > 1e-3:
        raise ConvergenceError(
            f"Flash: 1-D solver did not converge (best residual={best_r:.3e})."
        )
    return best_x


# ---------------------------------------------------------------------------
# Property accessors (mass-based) with safe fallback for robust root-finding
# ---------------------------------------------------------------------------
# Safe trial box: the flash never asks the backend to evaluate outside this
# range. Beyond it ThermoPack's density solver may fail to converge and call
# Fortran STOP (which aborts the whole process, uncatchable from Python).
_SAFE_T = (1.0, 6000.0)
_SAFE_P = (1.0, 5e7)


def _safe(T, P) -> bool:
    return (np.isfinite(T) and np.isfinite(P)
            and _SAFE_T[0] <= T <= _SAFE_T[1]
            and _SAFE_P[0] <= P <= _SAFE_P[1])


def _prop_getter(backend, z, M, var, phase_hint=None):
    def get(T, P):
        if not _safe(T, P):
            return np.nan
        try:
            phase = phase_hint or backend.guess_phase(T, P, z)
            if var in ("rho", "v"):
                v_mol = backend.specific_volume(T, P, z, phase)
                rho = M / v_mol
                return rho if var == "rho" else 1.0 / rho
            mp = backend.molar_properties(T, P, z, phase)
            return {"h": mp.h, "s": mp.s, "u": mp.u}[var] / M
        except Exception:
            return np.nan
    return get


def _has_tv(backend) -> bool:
    try:
        backend.pressure_at_volume  # noqa: B018
        return True
    except AttributeError:
        return False


def _P_from_v(backend, z, M, T, rho_mass):
    """Pressure at temperature T and mass density rho_mass [kg/m^3]."""
    if not (np.isfinite(T) and T > 0.0 and np.isfinite(rho_mass) and rho_mass > 0.0):
        return np.nan
    v_mol = M / rho_mass
    if _has_tv(backend):
        try:
            return backend.pressure_at_volume(T, v_mol, z)
        except Exception:
            pass
    # Fallback: 1-D solve P such that rho(T, P) = rho_mass.
    f = lambda P: M / backend.specific_volume(T, P, z, backend.guess_phase(T, P, z)) - rho_mass
    return _solve1d(f, [1e5, 1e6, 1e4], lo=_P_BOUNDS[0], hi=_P_BOUNDS[1])


# ---------------------------------------------------------------------------
# Pure-fluid saturation-aware two-phase resolution
# ---------------------------------------------------------------------------
def _sat_mass_values(sat, M, P):
    """Saturated liquid/vapor mass values for h, s, u."""
    hf, hg = sat.h_f / M, sat.h_g / M
    sf, sg = sat.s_f / M, sat.s_g / M
    vf, vg = 1.0 / (sat.rho_f * M), 1.0 / (sat.rho_g * M)
    uf, ug = hf - P * vf, hg - P * vg
    return {"h": (hf, hg), "s": (sf, sg), "u": (uf, ug)}


def _try_pure_two_phase(backend, z, given, M):
    """If the spec lies in the two-phase dome for a pure fluid, return a result."""
    if backend.nc != 1:
        return None
    energy = next((k for k in _ENERGY_VARS if k in given), None)
    if energy is None:
        return None

    try:
        if "P" in given:
            P = given["P"]
            Pc = backend.critical_pressure(z)
            if P >= Pc:
                return None
            Tsat = backend.saturation_temperature(P)
            sat = backend.saturation_state(Tsat)
            f, g = _sat_mass_values(sat, M, P)[energy]
            target = given[energy]
            if f <= target <= g or g <= target <= f:
                x = (target - f) / (g - f) if g != f else 0.0
                return FlashResult(Tsat, P, Phase.TWOPHASE, True, float(np.clip(x, 0, 1)), sat)
            # store sat + side for single-phase bracketing
            return ("single", Tsat, P, sat, "superheated" if target > max(f, g) else "subcooled")
        if "T" in given:
            T = given["T"]
            Tc = backend.critical_temperature(z)
            if T >= Tc:
                return None
            Psat = backend.saturation_pressure(T)
            sat = backend.saturation_state(T)
            f, g = _sat_mass_values(sat, M, Psat)[energy]
            target = given[energy]
            if f <= target <= g or g <= target <= f:
                x = (target - f) / (g - f) if g != f else 0.0
                return FlashResult(T, Psat, Phase.TWOPHASE, True, float(np.clip(x, 0, 1)), sat)
            return ("single", T, Psat, sat, "superheated" if target > max(f, g) else "subcooled")
    except NotImplementedError:
        return None
    except Exception:
        return None
    return None


def flash(
    backend: BaseBackend,
    z: np.ndarray,
    specs: dict,
    *,
    phase: str | Phase | None = None,
) -> FlashResult:
    """Solve a two-variable specification into a :class:`FlashResult`."""
    given = _normalize_specs(specs)
    M = mixture_molar_mass(z, backend.molar_masses())
    phase_hint: Phase | None = None
    if phase is not None:
        phase_hint = Phase(phase) if isinstance(phase, str) else phase
    keys = set(given.keys())

    # ---- Pure-fluid saturation awareness --------------------------------
    sat_info = None
    if phase_hint is None:
        res = _try_pure_two_phase(backend, z, given, M)
        if isinstance(res, FlashResult):
            return res
        if isinstance(res, tuple) and res[0] == "single":
            sat_info = res  # ("single", Tsat_or_T, P_or_Psat, sat, side)

    # ---- (T, P) both given ----------------------------------------------
    if {"T", "P"} <= keys:
        T, P = given["T"], given["P"]

    # ---- (T, rho) or (T, v): P from (T, v) directly --------------------
    elif "T" in keys and ("rho" in keys or "v" in keys):
        T = given["T"]
        rho = given["rho"] if "rho" in keys else 1.0 / given["v"]
        try:
            P = _P_from_v(backend, z, M, T, rho)
        except ConvergenceError:
            get = _prop_getter(backend, z, M, "rho" if "rho" in keys else "v", phase_hint)
            P = _solve1d(lambda P: get(T, P) - rho, [1e5, 1e6, 1e4])

    # ---- (P, rho) or (P, v): solve T from (T, v) = P -------------------
    elif "P" in keys and ("rho" in keys or "v" in keys):
        P = given["P"]
        rho = given["rho"] if "rho" in keys else 1.0 / given["v"]
        R_s = 8.31446261815324 / M
        T_ig = P / (rho * R_s)  # ideal-gas T estimate
        if _has_tv(backend):
            v_mol = M / rho
            def f(T):
                if not (np.isfinite(T) and T > 0.0):
                    return np.nan
                try:
                    return backend.pressure_at_volume(T, v_mol, z) - P
                except Exception:
                    return np.nan
            T = _solve1d(f, [T_ig, 300.0, 500.0, T_ig * 0.5, T_ig * 2.0],
                         lo=100.0, hi=None)
        else:
            get = _prop_getter(backend, z, M, "rho", phase_hint)
            T = _solve1d(lambda T: get(T, P) - rho,
                         [T_ig, 300.0, 500.0], lo=100.0, hi=None)

    # ---- (T, h/s/u): solve P -------------------------------------------
    elif "T" in keys and "P" not in keys:
        T = given["T"]
        other = (keys - {"T"}).pop()
        get = _prop_getter(backend, z, M, other, phase_hint)
        target = given[other]
        f = lambda P: get(T, P) - target
        guesses = [1e5, 5e4, 2e5, 1e6, 1e4, 5e5]
        if sat_info and other in _ENERGY_VARS:
            Psat = sat_info[2]; side = sat_info[4]
            if side == "superheated":
                guesses = [Psat * 0.5, Psat * 0.1, 1e4]; lo, hi = _SAFE_P[0], Psat
            else:
                guesses = [Psat * 2.0, Psat * 5.0, 1e6]; lo, hi = Psat, _SAFE_P[1]
        else:
            lo, hi = _SAFE_P
        P = _solve1d(f, guesses, lo=lo, hi=hi)

    # ---- (P, h/s/u): solve T (saturation-aware) ------------------------
    elif "P" in keys and "T" not in keys:
        P = given["P"]
        other = (keys - {"P"}).pop()
        get = _prop_getter(backend, z, M, other, phase_hint)
        target = given[other]
        f = lambda T: get(T, P) - target
        guesses = [300.0, 500.0, 800.0, 150.0]
        if sat_info and other in _ENERGY_VARS:
            Tsat = sat_info[1]; side = sat_info[4]
            if side == "superheated":
                guesses = [Tsat * 1.1, Tsat * 1.5, Tsat * 2.0]; lo, hi = Tsat, _SAFE_T[1]
            else:
                floor = max(100.0, 0.3 * Tsat)
                guesses = [Tsat * 0.9, Tsat * 0.7, floor + 1.0]; lo, hi = floor, Tsat
        else:
            lo, hi = _SAFE_T
        T = _solve1d(f, guesses, lo=lo, hi=hi)

    # ---- (rho/v, h/s/u): solve T via (T, v) interface ------------------
    elif ("rho" in keys or "v" in keys) and any(k in keys for k in _ENERGY_VARS):
        rho = given["rho"] if "rho" in keys else 1.0 / given["v"]
        energy = next(k for k in _ENERGY_VARS if k in keys)
        target = given[energy]
        R_s = 8.31446261815324 / M
        get = _prop_getter(backend, z, M, energy, phase_hint)

        def residual(T):
            try:
                P = _P_from_v(backend, z, M, T, rho)
            except Exception:
                return np.nan
            return get(T, P) - target

        # initial guesses for T (ideal-gas + a few)
        guesses = [300.0, 600.0, 1000.0, 150.0]
        T = _solve1d(residual, guesses, lo=100.0, hi=None)
        P = _P_from_v(backend, z, M, T, rho)

    # ---- (h, s), (h, u), (s, u): 2-D -----------------------------------
    else:
        v1, v2 = tuple(keys)
        t1, t2 = given[v1], given[v2]
        g1 = _prop_getter(backend, z, M, v1, phase_hint)
        g2 = _prop_getter(backend, z, M, v2, phase_hint)

        def residuals(x):
            T, P = x
            r1 = g1(T, P) - t1
            r2 = g2(T, P) - t2
            return [r1 if np.isfinite(r1) else 1e20, r2 if np.isfinite(r2) else 1e20]

        sol = fsolve(residuals, [300.0, 1e5], full_output=False)
        T, P = float(sol[0]), float(sol[1])
        if not (np.isfinite(T) and np.isfinite(P)):
            raise ConvergenceError("Flash: 2-D solver did not converge.")

    if not (np.isfinite(T) and np.isfinite(P)):
        raise ConvergenceError(f"Flash produced non-finite state T={T}, P={P}.")

    # ---- Phase determination + mixture two-phase detection --------------
    resolved_phase = phase_hint or backend.guess_phase(T, P, z)
    two_phase = False
    quality: float | None = None
    sat = None
    try:
        if backend.is_two_phase(T, P, z):
            two_phase = True
            resolved_phase = Phase.TWOPHASE
            try:
                sat = backend.saturation_state(T, z)
            except NotImplementedError:
                sat = None
            if sat is not None and any(k in keys for k in _ENERGY_VARS):
                ev = next(k for k in _ENERGY_VARS if k in keys)
                f, g = _sat_mass_values(sat, M, sat.P)[ev]
                quality = float(np.clip((given[ev] - f) / (g - f), 0.0, 1.0)) if g != f else 0.0
    except NotImplementedError:
        pass

    return FlashResult(T=T, P=P, phase=resolved_phase, two_phase=two_phase,
                       quality=quality, sat=sat)