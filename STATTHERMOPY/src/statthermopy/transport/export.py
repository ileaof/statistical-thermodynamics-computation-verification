"""Export helpers for transport results: tabular (CSV/Excel) and Tecplot ASCII grids.

The numeric results are produced entirely by the transport engine; these helpers only
serialise them. The Tecplot writer targets the two-dimensional T×P maps (an ``(i, j)`` ordered
point zone with ``VARIABLES = T P <prop>``), the format used for CFD/heat-transfer post-processing.
"""

from __future__ import annotations

import csv as csvlib
from collections.abc import Iterable
from pathlib import Path

from .transport import TRANSPORT_PROPS, TRANSPORT_UNITS

__all__ = ["write_tecplot_grid", "transport_to_rows"]


def write_tecplot_grid(path, x, y, z, title: str = "transport map") -> Path:
    """Write a 2-D ``(x, y, z)`` grid as a Tecplot ASCII point zone.

    Parameters
    ----------
    path : str | Path
        Output ``.dat`` file.
    x, y : 1-D array-like
        The grid coordinates (T, P) defining the columns/rows.
    z : 2-D array-like, shape ``(len(y), len(x))``
        The property values on the grid (row-major over ``y``).
    title : str
        Zone title.
    """
    import numpy as np

    path = Path(path)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    nx, ny = len(x), len(y)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f'TITLE = "{title}"\n')
        fh.write('VARIABLES = "T" "P" "value"\n')
        fh.write(f'ZONE T="{title}", I={nx}, J={ny}, DATAPACKING=POINT\n')
        for j in range(ny):
            for i in range(nx):
                fh.write(f"{x[i]:.10g} {y[j]:.10g} {z[j, i]:.10g}\n")
    return path


def transport_to_rows(props: Iterable[str], table: dict) -> list[list]:
    """Build CSV rows (header + data) from a property-vs-axis ``table`` mapping.

    ``table`` is a mapping ``{"T": [...], "P": [...], <prop>: [...]}`` (as produced by the
    GUI/CLI transport plotting path). The first axis column present (``T`` then ``P``) leads.
    """
    cols = list(table.keys())
    rows = [cols]
    n = len(table[cols[0]]) if cols else 0
    for i in range(n):
        rows.append([table[c][i] for c in cols])
    return rows


def write_transport_csv(path, props: Iterable[str], table: dict) -> Path:
    """Write a transport property table to CSV (axis columns + one column per property)."""
    path = Path(path)
    rows = transport_to_rows(props, table)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csvlib.writer(fh)
        for r in rows:
            w.writerow(r)
    return path


# silence unused-import lint: TRANSPORT_PROPS/TRANSPORT_UNITS document the export contract
_ = (TRANSPORT_PROPS, TRANSPORT_UNITS)
