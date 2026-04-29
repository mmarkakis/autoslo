"""Reusable paper-quality plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import plotext

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.tuner.tuner_utils import AggregatedSimulationResults
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

CLI_SCATTER_MARKERS = ["●", "■", "▲", "◆", "★", "✦", "◉", "▶"]


def plot_legend_to(path: str | Path):
    """
    Plot just the legend as a small image.
    """
    fig, ax = plt.subplots(figsize=(3, 1))
    for fmt in FORMATTING.values():
        ax.scatter([], [], label=fmt[0], color=fmt[1], marker=fmt[2], s=60)
    ax.legend(ncol=1, loc="center")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def cost_vs_compliance_scatter(
    points: Sequence[ScatterPoint],
    *,
    x_metric: str | SloMetric,
    x_scale: Optional[str] = None,
    existing_xlims: Optional[tuple[float, float]] = None,
    existing_ylims: Optional[tuple[float, float]] = None,
    title: str | None = None,
    x_pad: float = 0.05,
    y_bottom: float = 0,
    figsize: tuple[float, float] = (6, 5),
    ax: Axes | None = None,
    x_threshold_color: str = Palette.light_green,
    x_threshold_objective: SloObjective | None = None,
    report_improvement: bool = False,
) -> tuple[Figure, Axes, tuple[float, float], tuple[float, float]]:
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
    existing_xlims :
        If given, existing x-axis limits from other panels. Make sure we only
        expand these limits, never shrink them, to ensure consistent x-axis
        limits across panels.
    existing_ylims :
        If given, existing y-axis limits from other panels. Make sure we only
        expand these limits, never shrink them, to ensure consistent y-axis
        limits across panels.
    title :
        Optional title for the plot.
    x_pad :
        Relative padding added to the x-axis limits around the data.
    y_bottom :
        Lower bound for the y-axis.
    figsize :
        Figure size when creating a new figure (ignored if *ax* given).
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
    xlims
        The x-axis limits after plotting, which may be useful for ensuring
        consistent limits across panels.
    ylims
        The y-axis limits after plotting, which may be useful for ensuring
        consistent limits across panels.
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

    # Report improvement, if requested and possible.
    if report_improvement:
        # Find the point formatted with "prev_month", the end of the arrow.
        ending_point = [
            pt for pt in points if pt.formatting_id.startswith("prev")
        ]
        ending_point.sort(key=lambda pt: (pt.x, pt.y))
        # Find the point formatted with "32+32 RPU", the start of the arrow.
        starting_point = [pt for pt in points if "RPU" in pt.formatting_id]
        starting_point.sort(key=lambda pt: (pt.x, pt.y))

        start = starting_point[0]
        end = ending_point[0]
        ax.annotate(
            "",
            xy=(end.x, end.y),
            xytext=(start.x, start.y),
            arrowprops=dict(arrowstyle="->", color=Palette.gray, lw=1),
            zorder=-10,
        )
        violation_ratio = end.x / start.x if start.x > 0 else float("inf")
        cost_ratio = end.y / start.y if start.y > 0 else float("inf")
        ax.text(
            min(start.x, end.x) + abs(end.x - start.x) * 0.6,
            (start.y + end.y) * 0.5,
            f"Violation ↓ {1 -violation_ratio:.1%}\nCost ↓ {1 -cost_ratio:.1%}",
            color=Palette.gray,
            ha="left",
            va="center",
        )

    # Plot points.
    for pt in points:
        label, color, marker = FORMATTING.get(
            pt.formatting_id, (pt.formatting_id, Palette.gray, "o")
        )
        ax.scatter(pt.x, pt.y, label=label, color=color, marker=marker, s=60)

    ax.set_xlabel(x_metric_asobj.to_plot_axis_label())
    ax.set_ylabel("Cost ($)")

    yvals = [pt.y for pt in points]
    bottom, top = existing_ylims if existing_ylims else (0, max(yvals))
    top = max(top, max(yvals) * 1.1)
    ax.set_ylim(bottom=bottom, top=top)

    if x_scale is None:
        x_scale = x_metric_asobj.to_plot_axis_scale()
    ax.set_xscale(x_scale)
    if title:
        ax.set_title(title)

    # Relative padding around x data.
    xvals = [pt.x for pt in points]
    left, right = existing_xlims if existing_xlims else (min(xvals), max(xvals))
    if x_scale == "linear" and len(points) > 0:
        additional = (max(xvals) - min(xvals)) * x_pad
        left = 0
        right = max(right, max(xvals) + additional)
    elif x_scale == "log":
        factor = (max(xvals) / min(xvals)) ** x_pad
        left = min(left, min(xvals) / factor)
        right = max(right, max(xvals) * factor)
    ax.set_xlim(left, right)
    fig.tight_layout()
    return fig, ax, (left, right), (bottom, top)


def cli_cost_vs_compliance_scatter(
    entries: list[tuple[str, AggregatedSimulationResults]],
    slo_objective: SloObjective,
) -> None:
    """Print a terminal scatter plot of violation vs cost."""
    x_label = slo_objective.slo_metric.to_plot_axis_label()

    labels: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for label, agg in entries:
        labels.append(label)
        xs.append(agg.primary_violation(slo_objective.slo_metric))
        ys.append(agg.cost)

    if len(xs) < 2:
        return

    markers = CLI_SCATTER_MARKERS

    plotext.clear_figure()
    plotext.plot_size(60, 20)
    for i in range(len(xs)):
        mk = markers[i % len(markers)]
        plotext.scatter([xs[i]], [ys[i]], marker=mk)
    plotext.xlabel(x_label)
    plotext.ylabel("Cost ($)")

    # Add a vertical line at the SLO threshold and make sure it is included
    # in the plot bounds with some padding.
    threshold = slo_objective.slo_threshold
    plotext.vline(
        threshold,
        color="gray",
    )

    x_lo, x_hi = min(min(xs), threshold), max(max(xs), threshold)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = max((x_hi - x_lo) * 0.15, x_hi * 0.05) or 0.01
    y_pad = max((y_hi - y_lo) * 0.15, y_hi * 0.05) or 0.01
    plotext.xlim(x_lo - x_pad, x_hi + x_pad)
    plotext.ylim(y_lo - y_pad, y_hi + y_pad)

    plotext.title("Violation vs. Cost")
    plotext.theme("clear")

    # Build the plot as a string and append a legend to the right.
    plot_str = plotext.build()
    plot_lines = plot_str.split("\n")

    legend_lines: list[str] = [""]  # blank line at top
    for i, lbl in enumerate(labels):
        mk = markers[i % len(markers)]
        legend_lines.append(f"  {mk} {lbl}")
    legend_lines.append("")

    # Vertically centre the legend against the plot.
    total_plot = len(plot_lines)
    total_legend = len(legend_lines)
    offset = max(0, (total_plot - total_legend) // 2)

    out_lines: list[str] = []
    for row, pline in enumerate(plot_lines):
        li = row - offset
        suffix = legend_lines[li] if 0 <= li < total_legend else ""
        out_lines.append(pline + suffix)

    print("\n".join(out_lines))
