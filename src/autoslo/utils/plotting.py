"""Reusable paper-quality plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass
class ScatterPoint:
    """A single labelled point on a cost-vs-compliance scatter plot."""

    label: str
    x: float
    y: float
    color: str
    marker: str = "o"


def cost_vs_compliance_scatter(
    points: Sequence[ScatterPoint],
    *,
    xlabel: str = "Violation Rate",
    ylabel: str = "Cost ($)",
    title: str | None = None,
    x_pad: float = 0.05,
    y_bottom: float = 0,
    xscale: str = "linear",
    figsize: tuple[float, float] = (6, 5),
    legend: bool = True,
    ax: Axes | None = None,
    x_threshold: float | None = None,
    x_threshold_color: str = "#d4edda",
    x_threshold_label: str | None = None,
) -> tuple[Figure, Axes]:
    """Create a cost-vs-compliance scatter plot.

    Parameters
    ----------
    points :
        Labelled data points to plot.
    xlabel / ylabel / title :
        Axis labels and optional title.
    x_pad :
        Relative padding added to the x-axis limits around the data.
    y_bottom :
        Lower bound for the y-axis.
    xscale :
        Scale for the x-axis (``"linear"`` or ``"log"``).
    figsize :
        Figure size when creating a new figure (ignored if *ax* given).
    legend :
        Whether to draw a legend.
    ax :
        Optional pre-existing axes to draw on.  When *None* a new figure
        is created.
    x_threshold :
        If set, a vertical band is shaded from the y-axis (``x = 0``)
        to ``x = x_threshold``.
    x_threshold_color :
        Fill colour for the threshold band (default: light green).
    x_threshold_label :
        Optional legend label for the shaded region.

    Returns
    -------
    (fig, ax)
        The matplotlib figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    for pt in points:
        ax.scatter(
            pt.x, pt.y, label=pt.label, color=pt.color, marker=pt.marker, s=60
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    y_top = max(pt.y for pt in points) * 1.1 if points else 1.0
    ax.set_ylim(bottom=y_bottom, top=y_top)
    ax.set_xscale(xscale)
    if title:
        ax.set_title(title)

    # Relative padding around x data.
    xvals = [pt.x for pt in points]
    xrange = (
        max(xvals) - min(xvals)
        if len(xvals) > 1
        else max(abs(v) for v in xvals) or 1.0
    )
    if xscale == "linear" and len(points) > 0:
        ax.set_xlim(0, max(xvals) + x_pad * xrange)
    elif xscale == "log":
        ax.set_xlim(min(xvals) - x_pad * xrange, max(xvals) + x_pad * xrange)

    # Shaded threshold region.
    if x_threshold is not None:
        ax.axvspan(
            0,
            x_threshold,
            color=x_threshold_color,
            alpha=0.3,
            zorder=0,
            label=x_threshold_label,
        )

    if legend:
        ax.legend()

    fig.tight_layout()
    return fig, ax
