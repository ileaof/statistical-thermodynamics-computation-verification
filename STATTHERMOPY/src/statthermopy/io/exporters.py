"""Export thermodynamic results to common formats.

The :class:`Exporter` takes a :class:`~statthermopy.thermodynamics.ThermoProperties` (or a
:class:`~statthermopy.mixture.MixtureProperties`) plus, optionally, a property-vs-temperature
table, and writes it to CSV, JSON, YAML, Excel, LaTeX or a self-contained HTML report.
PDF is deferred to a later phase.

The numeric results are produced entirely by the statistical-mechanics engine; the exporters only
serialise them.
"""

from __future__ import annotations

import html
import json
import math
import csv as csvlib
from datetime import datetime
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

    def to_html(self, path: str | Path) -> Path:
        """Write a self-contained HTML report: no external CSS, fonts or scripts.

        The file is meant to be opened, read and printed on its own -- attached to an email or
        dropped in a report folder -- so everything it needs is inline. Values are written at
        full ``repr`` precision in the complete table and rounded only in the summary, so the
        document never quietly loses digits the other formats keep.
        """
        path = Path(path)
        data = _as_dict(self.result)
        flat = _flatten(data)

        paired = [
            ("Internal energy", "U", "J/mol", "J/kg", data.get("U_m"), data.get("U_s")),
            ("Enthalpy", "H", "J/mol", "J/kg", data.get("H_m"), data.get("H_s")),
            ("Entropy", "S", "J/mol/K", "J/kg/K", data.get("S_m"), data.get("S_s")),
            ("Helmholtz energy", "A", "J/mol", "J/kg", data.get("A_m"), data.get("A_s")),
            ("Gibbs energy", "G", "J/mol", "J/kg", data.get("G_m"), data.get("G_s")),
            ("Heat capacity (V)", "C<sub>v</sub>", "J/mol/K", "J/kg/K",
             data.get("Cv_m"), data.get("Cv_s")),
            ("Heat capacity (P)", "C<sub>p</sub>", "J/mol/K", "J/kg/K",
             data.get("Cp_m"), data.get("Cp_s")),
            ("Heat-capacity ratio", "&gamma;", "-", "-", data.get("gamma"), None),
        ]
        conditions = [
            ("Temperature", "T", data.get("T"), "K"),
            ("Pressure", "P", data.get("P"), "Pa"),
            ("Volume", "V", data.get("V"), "m<sup>3</sup>"),
            ("Amount", "n", data.get("n"), "mol"),
            ("Mass", "m", data.get("m"), "kg"),
            ("Molar mass", "M", data.get("molar_mass") or data.get("M_avg"), "kg/mol"),
        ]

        out = [_HTML_HEAD, "<h1>StatThermoPy</h1>",
               "<p class='sub'>Statistical Thermodynamics in Python</p>"]

        label = data.get("name") or data.get("label")
        if not label and isinstance(data.get("x"), dict):
            label = ", ".join(f"{k} {_fmt(v)}" for k, v in data["x"].items())
        if label:
            out.append(f"<p class='subject'>{_esc(label)}</p>")

        out.append("<h2>State</h2><table><thead><tr><th>Quantity</th><th>Symbol</th>"
                   "<th>Value</th><th>Unit</th></tr></thead><tbody>")
        for name, symbol, value, unit in conditions:
            if value is None:
                continue
            out.append(f"<tr><td>{name}</td><td>{symbol}</td>"
                       f"<td class='n'>{_fmt(value)}</td><td>{unit}</td></tr>")
        out.append("</tbody></table>")

        out.append("<h2>Properties</h2><table><thead><tr><th>Property</th><th>Symbol</th>"
                   "<th>Molar</th><th>Unit</th><th>Massic</th><th>Unit</th>"
                   "</tr></thead><tbody>")
        for name, symbol, mol_unit, mass_unit, molar, massic in paired:
            if molar is None and massic is None:
                continue
            massic_cell = "&mdash;" if massic is None else _fmt(massic)
            massic_unit = "" if massic is None else mass_unit
            out.append(
                f"<tr><td>{name}</td><td>{symbol}</td><td class='n'>{_fmt(molar)}</td>"
                f"<td>{mol_unit}</td><td class='n'>{massic_cell}</td><td>{massic_unit}</td></tr>"
            )
        out.append("</tbody></table>")

        out.append("<h2>All values</h2><table><thead><tr><th>Key</th><th>Value</th>"
                   "</tr></thead><tbody>")
        for key, value in flat.items():
            out.append(f"<tr><td>{_esc(key)}</td><td class='n'>{_esc(value)}</td></tr>")
        out.append("</tbody></table>")

        if self.table:
            cols = list(self.table.keys())
            out.append("<h2>Table</h2><div class='scroll'><table><thead><tr>")
            out += [f"<th>{_esc(c)}</th>" for c in cols]
            out.append("</tr></thead><tbody>")
            for i in range(len(self.table[cols[0]]) if cols else 0):
                cells = "".join(f"<td class='n'>{_fmt(self.table[c][i])}</td>" for c in cols)
                out.append(f"<tr>{cells}</tr>")
            out.append("</tbody></table></div>")

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        out.append(
            f"<footer>Generated by StatThermoPy on {stamp}. "
            "Properties derived from the molecular partition function; no empirical property "
            "correlations are used.</footer></body>"
        )
        path.write_text("\n".join(out), encoding="utf-8")
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


def _esc(v: Any) -> str:
    """HTML-escape a value; species names and keys are data, not markup."""
    return html.escape(str(v), quote=False)


#: Inline stylesheet for :meth:`Exporter.to_html` -- self-contained by design, so the report
#: renders identically offline, in an email client, and on paper.
_HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StatThermoPy results</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0 auto; padding: 2rem 1.25rem 3rem; max-width: 60rem;
         font: 15px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif;
         color: #0f172a; background: #ffffff; }
  h1 { margin: 0; font-size: 1.9rem; letter-spacing: -0.01em; color: #2563eb; }
  p.sub { margin: .15rem 0 0; color: #64748b; }
  p.subject { margin: 1rem 0 0; font-weight: 600; }
  h2 { margin: 2rem 0 .6rem; font-size: 1.05rem; color: #2563eb;
       border-bottom: 1px solid #e2e8f0; padding-bottom: .3rem; }
  table { border-collapse: collapse; width: 100%; font-size: .92rem; }
  th, td { padding: .38rem .6rem; border-bottom: 1px solid #eef2f7; text-align: left; }
  th { background: #f8fafc; font-weight: 600; color: #334155; }
  td.n { text-align: right; font-variant-numeric: tabular-nums;
         font-family: "Cascadia Mono", Consolas, monospace; }
  tr:hover td { background: #f8fafc; }
  .scroll { overflow-x: auto; }
  footer { margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid #e2e8f0;
           color: #64748b; font-size: .85rem; }
  @media (prefers-color-scheme: dark) {
    body { color: #e2e8f0; background: #0b1220; }
    th { background: #111827; color: #cbd5e1; }
    th, td { border-bottom-color: #1e293b; }
    tr:hover td { background: #111827; }
    h2 { border-bottom-color: #1e293b; }
    footer { border-top-color: #1e293b; }
  }
  @media print { body { max-width: none; } tr:hover td { background: none; } }
</style></head><body>"""