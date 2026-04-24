"""Reusable paper-quality plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.utils.colors import Palette


@dataclass
class ScatterPoint:
    """A single point on a cost-vs-compliance scatter plot."""

    formatting_id: str
    x: float
    y: float


FORMATTING = {
    "initial": ("Initial", Palette.gray, "x"),
    "ground_truth": ("Opt. on Target Day", Palette.dark_purple, "*"),
    "prev_day": ("Opt. on Past 1 day", Palette.light_blue, "o"),
    "prev_week": ("Opt. on Past 1 week", Palette.dark_blue, "o"),
    "prev_month": ("Opt. on Past 1 month", Palette.dark_green, "o"),
    "16 RPU": ("16 RPU", Palette.light_orange, "s"),
    "16+16 RPU": ("16+16 RPU", Palette.light_orange, "D"),
    "32 RPU": ("32 RPU", Palette.dark_orange, "s"),
    "32+32 RPU": ("32+32 RPU", Palette.dark_orange, "D"),
    "64 RPU": ("64 RPU", Palette.dark_red, "s"),
}


def cost_vs_compliance_scatter(
    points: Sequence[ScatterPoint],
    *,
    x_metric: str | SloMetric,
    x_scale: Optional[str] = None,
    title: str | None = None,
    x_pad: float = 0.05,
    y_bottom: float = 0,
    figsize: tuple[float, float] = (6, 5),
    legend: bool = True,
    ax: Axes | None = None,
    x_threshold_color: str = Palette.light_green,
    x_threshold_objective: SloObjective | None = None,
) -> tuple[Figure, Axes]:
    """Create a cost-vs-compliance scatter plot.

    Parameters
    ----------
    points :
        Labelled data points to plot.
    x_metric :
        The SLO metric or column name for the x-axis.
    x_scale :
        Scale for the x-axis (``"linear"`` or ``"log"``). If None, the scale is
        determined from the SLO metric.
    title :
        Optional title for the plot.
    x_pad :
        Relative padding added to the x-axis limits around the data.
    y_bottom :
        Lower bound for the y-axis.

    figsize :
        Figure size when creating a new figure (ignored if *ax* given).
    legend :
        Whether to draw a legend.
    ax :
        Optional pre-existing axes to draw on.  When *None* a new figure
        is created.
    x_threshold_color :
        Fill colour for the threshold band (default: light green).
    x_threshold_objective :
        If set, a vertical band is shaded according to the SLO threshold defined
        in the given SLO objective, only if the x-axis metric matches the SLO
        metric of the objective.

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

    x_metric_asobj = (
        SloMetric(x_metric) if isinstance(x_metric, str) else x_metric
    )

    # Shaded threshold region.
    if (
        x_threshold_objective is not None
    ) and x_threshold_objective.slo_metric == x_metric_asobj:
        x_threshold = x_threshold_objective.slo_threshold
        ax.axvspan(
            0,
            x_threshold,
            color=x_threshold_color,
            alpha=0.3,
            zorder=0,
        )

        # Add a textbox at the top of the span with the threshold value.
        ax.text(
            x_threshold,
            0.95,
            f"SLO Objective",
            color=Palette.dark_green,
            rotation=90,
            ha="right",
            va="top",
            transform=ax.get_xaxis_transform(),
        )

    for pt in points:
        label, color, marker = FORMATTING.get(
            pt.formatting_id, (pt.formatting_id, Palette.gray, "o")
        )
        ax.scatter(pt.x, pt.y, label=label, color=color, marker=marker, s=60)

    ax.set_xlabel(x_metric_asobj.to_plot_axis_label())
    ax.set_ylabel("Cost ($)")

    y_top = max(pt.y for pt in points) * 1.1 if points else 1.0
    ax.set_ylim(bottom=y_bottom, top=y_top)

    if x_scale is None:
        x_scale = x_metric_asobj.to_plot_axis_scale()
    ax.set_xscale(x_scale)
    if title:
        ax.set_title(title)

    # Relative padding around x data.
    xvals = [pt.x for pt in points]
    if x_scale == "linear" and len(points) > 0:
        additional = (max(xvals) - min(xvals)) * x_pad
        ax.set_xlim(0, max(xvals) + additional)
    elif x_scale == "log":
        factor = (max(xvals) / min(xvals)) ** x_pad
        ax.set_xlim(min(xvals) / factor, max(xvals) * factor)

    if legend:
        # Order the legend according to FORMATTING.
        handles, labels = ax.get_legend_handles_labels()
        label_sort_order = [fmt[0] for fmt in FORMATTING.values()]
        sorted_handles_labels = sorted(
            zip(handles, labels),
            key=lambda hl: (
                label_sort_order.index(hl[1])
                if hl[1] in label_sort_order
                else len(label_sort_order)
            ),
        )
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        ax.legend(sorted_handles, sorted_labels, ncols=2)

    fig.tight_layout()
    return fig, ax
