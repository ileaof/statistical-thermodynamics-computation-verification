"""Export helpers for air-transport results: CSV, Excel, JSON and PDF.

The numeric results are produced entirely by the air-transport engine (:mod:`.mixture_transport`,
:mod:`.air_transport`); these helpers only serialise them. The serialisation utilities
(``_native``, ``_as_dict``, ``_flatten``) are reused from :mod:`statthermopy.io.exporters` so the
output format stays consistent with the rest of the package. Tabular (vs-T) data is exported via
:class:`~statthermopy.humidair.analysis.ComparisonTable` (CSV / Excel / JSON / PDF).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...io.exporters import _as_dict, _flatten, _native

__all__ = ["AirTransportExporter"]


class AirTransportExporter:
    """Serialise a :class:`.MixtureTransportProperties` to CSV / Excel / JSON / PDF.

    Parameters
    ----------
    result : MixtureTransportProperties
        The point-evaluation result to export.
    table : ComparisonTable, optional
        A property-vs-temperature table (e.g. from :func:`.plot_air_transport`) to append.
    """

    def __init__(self, result, table=None) -> None:
        self.result = result
        self.table = table

    # -- helpers --------------------------------------------------------------

    def _properties(self) -> dict[str, Any]:
        """Flat property dict (conditions + mixture transport), excluding the nested breakdown."""
        d = _native(self.result.as_dict())
        d.pop("components", None)
        return _flatten(d)

    def _components(self) -> list[dict[str, Any]]:
        out = []
        for name, c in (self.result.components or {}).items():
            row = {"name": name}
            row.update(_flatten(_native(c.as_dict() if hasattr(c, "as_dict") else c.__dict__)))
            out.append(row)
        return out

    # -- text formats ---------------------------------------------------------

    def to_json(self, path) -> Path:
        """Write JSON (properties + per-species components + optional table)."""
        import json

        data = {
            "properties": self._properties(),
            "components": {c["name"]: c for c in self._components()},
        }
        if self.table is not None:
            data["table"] = {
                "title": self.table.title,
                "x_label": self.table.x_label,
                "y_label": self.table.y_label,
                "meta": _native(self.table.meta),
                "columns": _native(self.table.columns),
            }
        path = Path(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=float)
        return path

    def to_csv(self, path) -> Path:
        """Write a flat CSV (one key/value row per property; components + table appended)."""
        import csv as csvlib

        path = Path(path)
        props = self._properties()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csvlib.writer(fh)
            w.writerow(["property", "value"])
            for k, v in props.items():
                w.writerow([k, v])
            # per-species components
            w.writerow([])
            w.writerow(["component"] + list(self._components()[0].keys()) if self._components() else [])
            for c in self._components():
                w.writerow(list(c.values()))
            if self.table is not None:
                w.writerow([])
                cols = list(self.table.columns.keys())
                w.writerow([self.table.x_label, *cols])
                n = len(self.table.x)
                for i in range(n):
                    w.writerow([self.table.x[i], *[self.table.columns[k][i] for k in cols]])
        return path

    def to_excel(self, path) -> Path:
        """Write an Excel workbook (sheets: properties, components, table)."""
        import pandas as pd

        path = Path(path)
        sheets = {
            "properties": pd.DataFrame(
                [{"property": k, "value": v} for k, v in self._properties().items()]
            ),
            "components": pd.DataFrame(self._components()),
        }
        if self.table is not None:
            df = pd.DataFrame({self.table.x_label: self.table.x})
            df = pd.concat([df, pd.DataFrame(self.table.columns)], axis=1)
            sheets["table"] = df
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        return path

    def to_pdf(self, path) -> Path:
        """Render the property table to a PDF (matplotlib table figure). No external PDF engine."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        props = self._properties()
        rows = [[k, f"{float(v):.6g}" if isinstance(v, (int, float)) else str(v)]
                for k, v in props.items()]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1.2))
        ax.axis("off")
        ax.set_title(f"Air transport — {getattr(self.result, 'label', '')} @ "
                     f"T={self.result.T:.2f} K, P={self.result.P:.4g} Pa", fontsize=10)
        tbl = ax.table(cellText=rows, colLabels=["property", "value"], loc="center",
                       cellLoc="left", colWidths=[0.5, 0.45])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.15)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return Path(path)