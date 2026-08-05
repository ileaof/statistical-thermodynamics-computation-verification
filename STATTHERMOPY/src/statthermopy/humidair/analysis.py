"""Comparative psychrometric and thermodynamic analysis for humid air.

This is the *data layer* behind the graphical comparisons: it turns a
:class:`~statthermopy.humidair.humidair.HumidAir` model into tabular results that can be both
plotted (see :mod:`statthermopy.humidair.plots`) and exported (CSV/Excel). It is fully integrated
with the humid-air engine — any future change to the partition-function or psychrometric
calculations is reflected here automatically, because everything is computed through the same
:class:`HumidAir` / :class:`~statthermopy.mixture.IdealGasMixture` objects.

Two analyses are provided:

* :meth:`PsychrometricAnalysis.water_vapor_content` — the actual and the maximum (saturation)
  humidity ratio versus temperature, with the dew point (onset of condensation).
* :meth:`PsychrometricAnalysis.property_comparison` — a thermodynamic property of **dry air** and
  **humid air** versus temperature under both the **isobaric** (constant P) and **isochoric**
  (constant V) constraints, i.e. up to four curves.

Physics note (ideal gas). ``U, H, Cp, Cv, T_v, T_p`` are functions of temperature only, so their
isobaric and isochoric curves coincide; only the pressure-dependent properties ``S, G, A`` differ
between the two constraints. Under the isochoric constraint the molar volume is held at its
reference value ``v_ref = R T_ref / P``, so the pressure tracks ``P(T) = P · T / T_ref``.
Condensation is *not* modelled in the property comparison (the composition is held fixed); the
saturation limit and condensation onset are shown by the water-vapour-content analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.state import State
from .humidair import HumidAir

__all__ = ["PsychrometricAnalysis", "ComparisonTable", "COMPARISON_PROPERTIES"]

#: Comparable thermodynamic properties: display name -> (MixtureProperties field, unit).
COMPARISON_PROPERTIES: dict[str, tuple[str, str]] = {
    "Enthalpy H": ("H_m", "J/mol"),
    "Internal energy U": ("U_m", "J/mol"),
    "Entropy S": ("S_m", "J/mol/K"),
    "Gibbs energy G": ("G_m", "J/mol"),
    "Helmholtz energy A": ("A_m", "J/mol"),
    "Cp": ("Cp_m", "J/mol/K"),
    "Cv": ("Cv_m", "J/mol/K"),
    "T_v": ("T_v", "K"),
    "T_p": ("T_p", "K"),
}

#: Properties whose isobaric and isochoric curves coincide (temperature-only for an ideal gas).
_PRESSURE_INDEPENDENT = {"U_m", "H_m", "Cv_m", "Cp_m", "T_v", "T_p"}


@dataclass
class ComparisonTable:
    """A named set of temperature-indexed columns, ready to plot or export.

    ``x`` is the temperature axis (in ``x_unit``); ``columns`` maps a series label to its values;
    ``x_K`` is always the temperature in kelvin. ``meta`` carries annotations (units, the dew
    point, the fixed water content, …) used by the plots and the exporters.
    """

    title: str
    x_label: str
    y_label: str
    x_unit: str
    x: list
    x_K: list
    columns: dict
    meta: dict = field(default_factory=dict)

    def as_rows(self) -> tuple[list[str], list[list]]:
        """Return ``(header, rows)`` for tabular export (temperature first, then each column)."""
        header = [f"T [{self.x_unit}]", *self.columns.keys()]
        rows = []
        n = len(self.x)
        keys = list(self.columns.keys())
        for i in range(n):
            rows.append([self.x[i], *[self.columns[k][i] for k in keys]])
        return header, rows

    def to_dataframe(self):
        """Return a :class:`pandas.DataFrame` (temperature index + one column per series)."""
        import pandas as pd
        data = {f"T [{self.x_unit}]": self.x}
        data.update(self.columns)
        return pd.DataFrame(data)

    def to_csv(self, path) -> str:
        """Write the table to a CSV file and return the path."""
        self.to_dataframe().to_csv(path, index=False)
        return str(path)

    def to_excel(self, path) -> str:
        """Write the table to an Excel (.xlsx) file and return the path."""
        self.to_dataframe().to_excel(path, index=False, sheet_name="humid_air")
        return str(path)

    def to_json(self, path) -> str:
        """Write the table (header + columns + meta) to a JSON file and return the path."""
        import json

        data = {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "meta": self.meta,
            "columns": self.columns,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=float)
        return str(path)

    def to_pdf(self, path) -> str:
        """Render the table to a PDF (matplotlib table figure) and return the path.

        No external PDF engine is required — matplotlib's built-in PDF backend is used.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        header = [self.x_label, *self.columns.keys()]
        keys = list(self.columns.keys())
        n = len(self.x)
        cell_text = [
            [f"{self.x[i]:.4g}", *[f"{self.columns[k][i]:.6g}" for k in keys]]
            for i in range(n)
        ]
        fig, ax = plt.subplots(figsize=(8, 0.4 * n + 1.2))
        ax.axis("off")
        ax.set_title(self.title, fontsize=10)
        tbl = ax.table(cellText=cell_text, colLabels=header, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.1)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return str(path)


def _to_unit(T_K: float, unit: str) -> float:
    return T_K - 273.15 if unit.upper().startswith("C") else T_K


class PsychrometricAnalysis:
    """Comparative analyses built on a :class:`HumidAir` model.

    Parameters
    ----------
    model : HumidAir, optional
        The humid-air model (dry-air background + saturation solver). Defaults to standard air.
    """

    def __init__(self, model: HumidAir | None = None) -> None:
        self.model = model if model is not None else HumidAir()

    # -- helpers --------------------------------------------------------------

    def _fixed_water_content(
        self, T_ref: float, P: float,
        relative_humidity, humidity_ratio, mole_fraction,
    ) -> float:
        """Resolve the fixed humidity ratio ``w`` (kg/kg dry air) from a humidity spec at T_ref."""
        eps = self.model.epsilon
        if humidity_ratio is not None:
            return float(humidity_ratio)
        if mole_fraction is not None:
            x = float(mole_fraction)
            return eps * x / (1.0 - x)
        if relative_humidity is not None:
            p_v = float(relative_humidity) * self.model.saturation_pressure(T_ref)
        else:  # saturated at the reference temperature
            p_v = self.model.saturation_pressure(T_ref)
        p_v = min(p_v, 0.999999 * P)
        return eps * p_v / (P - p_v)

    def _humid_mixture_for(
        self, T_ref: float, P: float,
        relative_humidity, humidity_ratio, mole_fraction,
    ):
        """Build the fixed-composition humid-air mixture from a humidity spec at T_ref."""
        w = self._fixed_water_content(T_ref, P, relative_humidity, humidity_ratio, mole_fraction)
        r = w / self.model.epsilon
        x_w = r / (1.0 + r)
        return self.model._humid_mixture(x_w), w, x_w

    # -- 1) water-vapour content ---------------------------------------------

    def water_vapor_content(
        self, T_range, P: float = 101325.0, *,
        temperature_unit: str = "K",
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
        T_ref: float | None = None,
    ) -> ComparisonTable:
        """Actual and saturation humidity ratio (g H₂O / kg dry air) versus temperature.

        The *actual* content is the fixed water mass set by the humidity spec (evaluated at
        ``T_ref``, default: the midpoint of the range); it is capped by the saturation value once
        the air cools past its dew point (condensation). The saturation curve is the maximum the
        air can hold. Both are returned in g/kg together with the relative humidity and the dew
        point (onset of condensation).
        """
        m = self.model
        eps = m.epsilon
        Ts = [float(t) for t in T_range]
        if T_ref is None:
            T_ref = 0.5 * (Ts[0] + Ts[-1])
        w_fixed = self._fixed_water_content(
            T_ref, P, relative_humidity, humidity_ratio, mole_fraction
        )
        p_v_fixed = w_fixed * P / (eps + w_fixed)

        w_actual, w_sat, rh = [], [], []
        for t in Ts:
            ps = m.saturation_pressure(t)
            wsat = eps * ps / (P - ps) if ps < P else float("nan")
            w_sat.append(wsat * 1e3 if wsat == wsat else float("nan"))
            # condensation: actual content cannot exceed saturation
            if wsat != wsat or w_fixed <= wsat:
                w_actual.append(w_fixed * 1e3)
            else:
                w_actual.append(wsat * 1e3)
            rh.append(min(p_v_fixed / ps, 1.0) if ps > 0 else float("nan"))

        dew_K = m.saturation.dew_point(p_v_fixed) if p_v_fixed > 0 else float("nan")
        x = [_to_unit(t, temperature_unit) for t in Ts]
        return ComparisonTable(
            title="Water-vapour content vs temperature",
            x_label=f"Temperature ({temperature_unit})",
            y_label="Water vapour content [g/kg dry air]",
            x_unit=temperature_unit,
            x=x, x_K=Ts,
            columns={
                "actual w [g/kg]": w_actual,
                "saturation w_sat [g/kg]": list(w_sat),
                "relative humidity [-]": rh,
            },
            meta={
                "P": P, "w_fixed_g_per_kg": w_fixed * 1e3,
                "dew_point_K": dew_K, "dew_point": _to_unit(dew_K, temperature_unit),
                "condensation_below_K": dew_K,
            },
        )

    # -- 2) dry vs humid property comparison ---------------------------------

    def property_comparison(
        self, prop_field: str, T_range, P: float = 101325.0, *,
        temperature_unit: str = "K",
        isobaric: bool = True,
        isochoric: bool = True,
        T_ref: float | None = None,
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
    ) -> ComparisonTable:
        """A thermodynamic property of dry vs humid air vs temperature, isobaric and isochoric.

        ``prop_field`` is a :class:`~statthermopy.mixture.MixtureProperties` attribute (e.g.
        ``"H_m"``, ``"S_m"``, ``"T_v"``). Up to four curves are returned: dry/humid × isobaric/
        isochoric. For the pressure-independent properties the isobaric and isochoric curves are
        identical (and are still emitted so the four-curve legend is consistent).
        """
        m = self.model
        Ts = [float(t) for t in T_range]
        if T_ref is None:
            T_ref = 0.5 * (Ts[0] + Ts[-1])
        dry = m.dry_air
        humid, w_fixed, x_w = self._humid_mixture_for(
            T_ref, P, relative_humidity, humidity_ratio, mole_fraction
        )

        def series(mixture, constant_volume: bool):
            out = []
            for t in Ts:
                p = P * t / T_ref if constant_volume else P
                out.append(getattr(mixture.compute(State(T=t, P=p)), prop_field))
            return out

        columns: dict[str, list] = {}
        if isobaric:
            columns["Dry air — const P"] = series(dry, False)
            columns["Humid air — const P"] = series(humid, False)
        if isochoric:
            columns["Dry air — const V"] = series(dry, True)
            columns["Humid air — const V"] = series(humid, True)

        name = next((k for k, v in COMPARISON_PROPERTIES.items() if v[0] == prop_field), prop_field)
        unit = next((v[1] for v in COMPARISON_PROPERTIES.values() if v[0] == prop_field), "")
        x = [_to_unit(t, temperature_unit) for t in Ts]
        return ComparisonTable(
            title=f"Dry vs humid air — {name} vs temperature",
            x_label=f"Temperature ({temperature_unit})",
            y_label=f"{name} [{unit}]",
            x_unit=temperature_unit,
            x=x, x_K=Ts,
            columns=columns,
            meta={
                "P": P, "T_ref": T_ref, "property": prop_field, "unit": unit,
                "humid_x_H2O": x_w, "humid_w_g_per_kg": w_fixed * 1e3,
                "pressure_independent": prop_field in _PRESSURE_INDEPENDENT,
            },
        )

    # -- 3) thermal fields comparison ----------------------------------------

    def thermal_fields_comparison(
        self, T_range, P: float = 101325.0, *,
        temperature_unit: str = "K",
        T_ref: float | None = None,
        relative_humidity: float | None = None,
        humidity_ratio: float | None = None,
        mole_fraction: float | None = None,
    ) -> ComparisonTable:
        """Thermal fields of dry vs humid air: constant-volume ``T_v`` and constant-pressure
        ``T_p`` (K), each computed **independently** from the respective mixture's own properties.

        Four curves, all in kelvin:

        * Dry air, constant volume:   ``T_v = U_m^dry   / Cv_m^dry``
        * Humid air, constant volume: ``T_v = U_m^humid / Cv_m^humid``
        * Dry air, constant pressure: ``T_p = H_m^dry   / Cp_m^dry``
        * Humid air, constant pressure: ``T_p = H_m^humid / Cp_m^humid``

        Under no circumstances is a property of one mixture reused for the other: ``U_m``, ``H_m``,
        ``Cv_m`` and ``Cp_m`` come from each mixture's own :meth:`compute`, so the dry and humid
        fields differ because the underlying thermodynamic properties differ.
        """
        m = self.model
        Ts = [float(t) for t in T_range]
        if T_ref is None:
            T_ref = 0.5 * (Ts[0] + Ts[-1])
        dry = m.dry_air
        humid, w_fixed, x_w = self._humid_mixture_for(
            T_ref, P, relative_humidity, humidity_ratio, mole_fraction
        )

        def fields(mixture):
            tv, tp = [], []
            for t in Ts:
                pr = mixture.compute(State(T=t, P=P))
                # T_v = U_m/Cv_m and T_p = H_m/Cp_m of *this* mixture (no cross-use)
                tv.append(pr.U_m / pr.Cv_m)
                tp.append(pr.H_m / pr.Cp_m)
            return tv, tp

        dry_tv, dry_tp = fields(dry)
        humid_tv, humid_tp = fields(humid)
        x = [_to_unit(t, temperature_unit) for t in Ts]
        return ComparisonTable(
            title="Thermal fields (T_v, T_p): dry vs humid air",
            x_label=f"Temperature ({temperature_unit})",
            y_label="Thermal field [K]",
            x_unit=temperature_unit,
            x=x, x_K=Ts,
            columns={
                "Dry air T_v (const V)": dry_tv,
                "Humid air T_v (const V)": humid_tv,
                "Dry air T_p (const P)": dry_tp,
                "Humid air T_p (const P)": humid_tp,
            },
            meta={
                "P": P, "humid_x_H2O": x_w, "humid_w_g_per_kg": w_fixed * 1e3,
                "note": "each field uses its own mixture's U_m/H_m/Cv_m/Cp_m",
                "max_diff_Tv": max(abs(h - d) for h, d in zip(humid_tv, dry_tv, strict=False)),
                "max_diff_Tp": max(abs(h - d) for h, d in zip(humid_tp, dry_tp, strict=False)),
            },
        )
