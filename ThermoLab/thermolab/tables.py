"""Thermodynamic tables and property interpolation.

Builds tabulated property data over a temperature/pressure grid and provides
fast multidimensional interpolation suitable for CFD lookup tables.
"""

from __future__ import annotations

import numpy as np

from .state import State


class PropertyTable:
    """A gridded property table over ``(T, P)`` for a fluid.

    Parameters
    ----------
    fluid:
        A :class:`~thermolab.Gas` or :class:`~thermolab.Mixture`.
    T_range:
        Sequence of temperatures [K] (or ``(min, max, n)`` triple).
    P_range:
        Sequence of pressures [Pa] (or ``(min, max, n)`` triple).

    Attributes
    ----------
    df : pandas.DataFrame
        Long-format table with columns ``T, P`` plus every property.
    """

    PROPERTIES = (
        "rho", "h", "s", "u", "cp", "cv", "gamma", "Z", "sound_speed",
        "mu", "k", "thermal_diffusivity", "prandtl",
    )

    def __init__(self, fluid, T_range, P_range):
        self.fluid = fluid
        self.T = self._as_array(T_range)
        self.P = self._as_array(P_range)
        self._states: list[list[State | None]] = []
        for T in self.T:
            row = []
            for P in self.P:
                try:
                    row.append(fluid.state(T=float(T), P=float(P)))
                except Exception:
                    row.append(None)
            self._states.append(row)
        self.df = self._to_dataframe()

    @staticmethod
    def _as_array(rng):
        if len(rng) == 3 and all(np.isscalar(x) for x in rng):
            lo, hi, n = rng
            return np.linspace(lo, hi, int(n))
        return np.asarray(rng, dtype=float)

    def _to_dataframe(self):
        import pandas as pd

        records = []
        for i, T in enumerate(self.T):
            for j, P in enumerate(self.P):
                st = self._states[i][j]
                rec = {"T": float(T), "P": float(P)}
                if st is not None:
                    for prop in self.PROPERTIES:
                        try:
                            rec[prop] = float(getattr(st, prop))
                        except Exception:
                            rec[prop] = np.nan
                else:
                    for prop in self.PROPERTIES:
                        rec[prop] = np.nan
                records.append(rec)
        return pd.DataFrame.from_records(records)

    # ------------------------------------------------------------------
    def interpolate(self):
        """Return a callable ``f(T, P)`` for fast interpolated lookups.

        The returned function accepts scalars or arrays and returns a dict of
        interpolated property arrays (one per property).
        """
        from scipy.interpolate import RegularGridInterpolator

        grid = {}
        for prop in self.PROPERTIES:
            vals = np.full((len(self.T), len(self.P)), np.nan)
            for i in range(len(self.T)):
                for j in range(len(self.P)):
                    st = self._states[i][j]
                    if st is not None:
                        try:
                            vals[i, j] = float(getattr(st, prop))
                        except Exception:
                            vals[i, j] = np.nan
            vals = self._fill_nans(vals)
            grid[prop] = RegularGridInterpolator(
                (self.T, self.P), vals, bounds_error=False, fill_value=np.nan
            )

        def f(T, P):
            T = np.asarray(T, dtype=float)
            P = np.asarray(P, dtype=float)
            shape = np.broadcast_shapes(T.shape, P.shape)
            pts = np.stack([
                np.broadcast_to(T, shape).ravel(),
                np.broadcast_to(P, shape).ravel(),
            ], axis=-1)
            out = {prop: float(interp(pts)[0]) if pts.size == 2 else interp(pts).reshape(shape)
                   for prop, interp in grid.items()}
            return out

        return f

    @staticmethod
    def _fill_nans(arr: np.ndarray) -> np.ndarray:
        arr = arr.copy()
        nan = np.isnan(arr)
        if nan.all():
            return np.zeros_like(arr)
        from scipy.ndimage import distance_transform_edt
        if nan.any():
            idxs = distance_transform_edt(nan, return_distances=False, return_indices=True)
            arr = arr[tuple(idxs)]
        return arr


class SaturationTable:
    """Saturated liquid/vapor properties along the saturation curve of a pure fluid."""

    PROPERTIES = ("rho_f", "rho_g", "h_f", "h_g", "s_f", "s_g")

    def __init__(self, fluid, T_range, *, n: int | None = None):
        self.fluid = fluid
        if len(T_range) == 3 and all(np.isscalar(x) for x in T_range):
            lo, hi, nT = T_range
            self.T = np.linspace(lo, hi, int(nT))
        else:
            self.T = np.asarray(T_range, dtype=float)
        records = []
        for T in self.T:
            rec = {"T": float(T)}
            try:
                sat = fluid.backend.saturation_state(float(T), fluid.fractions)
                M = fluid.molar_mass
                rec["P"] = sat.P
                rec["rho_f"] = sat.rho_f * M
                rec["rho_g"] = sat.rho_g * M
                rec["h_f"] = sat.h_f / M
                rec["h_g"] = sat.h_g / M
                rec["s_f"] = sat.s_f / M
                rec["s_g"] = sat.s_g / M
            except Exception:
                for k in ("P",) + self.PROPERTIES:
                    rec[k] = np.nan
            records.append(rec)
        import pandas as pd
        self.df = pd.DataFrame.from_records(records)