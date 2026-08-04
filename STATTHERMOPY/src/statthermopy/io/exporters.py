"""Export thermodynamic results to common formats.

The :class:`Exporter` takes a :class:`~statthermopy.thermodynamics.ThermoProperties` (or a
:class:`~statthermopy.mixture.MixtureProperties`) plus, optionally, a property-vs-temperature
table, and writes it to CSV, JSON, YAML, Excel or LaTeX. PDF is deferred to a later phase.

The numeric results are produced entirely by the statistical-mechanics engine; the exporters only
serialise them.
"""

from __future__ import annotations

import json
import math
import csv as csvlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..thermodynamics import ThermoProperties
from ..mixture import MixtureProperties

__all__ = ["Exporter"]


def _native(value: Any) -> Any:
    """Convert numpy scalars/arrays to native Python types for safe serialisation.

    Non-finite floats (``±inf`` / ``NaN``) are mapped to ``None`` so that the result can be
    written to strict formats such as YAML. This only arises at the singular ``T = 0`` point
    of the classical ideal gas (e.g. ``S_m -> -inf``); every other state is finite.
    """
    if isinstance(value, np.generic):
        v = value.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(value, np.ndarray):
        return [_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_dict(obj: Any) -> dict:
    if hasattr(obj, "as_dict"):
        return _native(obj.as_dict())
    if hasattr(obj, "__dict__"):
        return _native(dict(obj.__dict__))
    raise TypeError(f"Cannot serialise object of type {type(obj)!r}.")


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dicts for tabular formats."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class Exporter:
    """Serialise a thermodynamic result to one of several file formats.

    Parameters
    ----------
    result : ThermoProperties | MixtureProperties
        The result to export.
    table : dict, optional
        A property-vs-temperature mapping (e.g. ``{"T": [...], "Cp_m": [...]}``) to append as a
        table section. Useful for plotting exports.
    """

    def __init__(self, result: ThermoProperties | MixtureProperties, table: dict | None = None) -> None:
        self.result = result
        self.table = table

    # -- text formats ---------------------------------------------------------

    def to_json(self, path: str | Path) -> Path:
        """Write JSON (with the optional table)."""
        data = {"properties": _as_dict(self.result)}
        if self.table:
            data["table"] = self.table
        path = Path(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=float)
        return path

    def to_yaml(self, path: str | Path) -> Path:
        """Write YAML."""
        data = {"properties": _as_dict(self.result)}
        if self.table:
            data["table"] = self.table
        path = Path(path)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        return path

    def to_csv(self, path: str | Path) -> Path:
        """Write a flat CSV (one key/value row per property; table appended if present)."""
        path = Path(path)
        flat = _flatten(_as_dict(self.result))
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csvlib.writer(fh)
            w.writerow(["property", "value"])
            for k, v in flat.items():
                w.writerow([k, v])
            if self.table:
                w.writerow([])
                # table header + rows
                cols = list(self.table.keys())
                w.writerow(cols)
                nrows = len(self.table[cols[0]]) if cols else 0
                for i in range(nrows):
                    w.writerow([self.table[c][i] for c in cols])
        return path

    def to_excel(self, path: str | Path) -> Path:
        """Write an Excel workbook (requires ``openpyxl`` via the ``excel`` extra)."""
        import pandas as pd  # local import; pandas is a core dep

        path = Path(path)
        props = _flatten(_as_dict(self.result))
        df_props = pd.DataFrame(
            [{"property": k, "value": v} for k, v in props.items()]
        )
        sheets = {"properties": df_props}
        if self.table:
            sheets["table"] = pd.DataFrame(self.table)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        return path

    def to_latex(self, path: str | Path) -> Path:
        """Write a LaTeX ``tabular`` of the molar and massic properties."""
        path = Path(path)
        d = _as_dict(self.result)
        rows = [
            ("Internal energy U", d.get("U_m"), d.get("U_s")),
            ("Enthalpy H", d.get("H_m"), d.get("H_s")),
            ("Entropy S", d.get("S_m"), d.get("S_s")),
            ("Helmholtz A", d.get("A_m"), d.get("A_s")),
            ("Gibbs G", d.get("G_m"), d.get("G_s")),
            ("Cv", d.get("Cv_m"), d.get("Cv_s")),
            ("Cp", d.get("Cp_m"), d.get("Cp_s")),
            ("gamma", d.get("gamma"), d.get("gamma")),
        ]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(r"\begin{tabular}{lrr}" + "\n")
            fh.write(r"\hline" + "\n")
            fh.write(r"Property & Molar (per mol) & Massic (per kg) \\\\" + "\n")
            fh.write(r"\hline" + "\n")
            for name, m, s in rows:
                fh.write(f"{name} & {_fmt(m)} & {_fmt(s)} \\\\\n")
            fh.write(r"\hline" + "\n")
            fh.write(r"\end{tabular}" + "\n")
        return path


def _fmt(v: Any) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)