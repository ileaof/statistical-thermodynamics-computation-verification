"""Consistent, publication-quality plotting helpers.

Every figure in the book uses the same restrained colour palette and a small set
of layout conventions.  Collecting them here keeps the look uniform and lets the
examples and tools produce identical styling with a single import.

The palette is deliberately colour-blind friendly and prints legibly in
grayscale.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt

#: Named colours used consistently throughout the figures.
COLORS = {
    "navy": "#2c3e50",        # primary data / reference curves
    "red": "#c0392b",         # highlighted result / exact value
    "blue": "#2980b9",        # secondary series
    "green": "#27ae60",       # limits / annotations
    "orange": "#e67e22",      # tertiary series
    "grey": "#95a5a6",        # de-emphasised guides
    "light_blue": "#9ecae1",  # filled histograms / bands
}

#: Default colour cycle for multi-series plots.
CYCLE = [COLORS["navy"], COLORS["red"], COLORS["blue"],
         COLORS["green"], COLORS["orange"], COLORS["grey"]]


def apply_style() -> None:
    """Apply the book's global Matplotlib style.

    Sets font sizes, line widths, legend frames and figure DPI to the values
    used for the published figures.  Safe to call more than once.
    """
    matplotlib.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.constrained_layout.use": True,
    })


def new_figure(nrows: int = 1, ncols: int = 1, figsize=None):
    """Create a styled figure and axes.

    Parameters
    ----------
    nrows, ncols : int, optional
        Subplot grid shape.
    figsize : tuple of float, optional
        Figure size in inches.  Defaults to a width that scales with the number
        of columns.

    Returns
    -------
    tuple
        ``(figure, axes)`` exactly as returned by
        :func:`matplotlib.pyplot.subplots`.
    """
    apply_style()
    if figsize is None:
        figsize = (5.5 * ncols, 4.2 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)


def save_figure(fig, name: str, directory: Optional[str] = None,
                dpi: int = 200) -> str:
    """Save a figure as a PNG, creating the target directory if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    name : str
        File name; a ``.png`` suffix is added if absent.
    directory : str, optional
        Destination directory.  Defaults to the current working directory.
    dpi : int, optional
        Output resolution.

    Returns
    -------
    str
        The full path of the written file.
    """
    if not name.lower().endswith(".png"):
        name += ".png"
    if directory:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
    else:
        path = name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


__all__ = ["COLORS", "CYCLE", "apply_style", "new_figure", "save_figure"]
