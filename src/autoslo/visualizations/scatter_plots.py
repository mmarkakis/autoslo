from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import plotext
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Console

from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.visualizations.colors import Palette
from autoslo.workload_execution.aggregated_execution_results import (
    AggregatedExecutionResults,
)

_console = Console()


@dataclass
class ScatterPoint:
    """A single point on a cost-vs-compliance scatter plot."""

    formatting_id: str
    label: str
    x: float
    y: float


@dataclass
class ImprovementArrow:
    """Config for an annotated arrow showing improvement from one point to another.

    The arrow is drawn from ``base_label`` to ``target_label``, and annotated
    with the percentage change along both axes.  Specify these per-panel in the
    plotting manifest under the ``improvement_arrow`` key.
    """

    base_label: str
    target_label: str


FORMATTING = {
    "initial": (Palette.gray, "x"),
    "ground_truth": (Palette.dark_purple, "*"),
    "prev_day": (Palette.light_blue, "o"),
    "prev_week": (Palette.dark_blue, "o"),
    "prev_month": (Palette.dark_green, "o"),
    "base_16": (Palette.light_orange, "s"),
    "base_16_16": (Palette.light_orange, "D"),
    "base_32": (Palette.dark_orange, "s"),
    "base_32_32": (Palette.dark_orange, "D"),
    "base_64": (Palette.dark_red, "s"),
    "round_robin": (Palette.light_orange, "s"),
    "stage": (Palette.light_blue, "^"),
    "iconq": (Palette.light_green, "o"),
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
    improvement_arrow: ImprovementArrow | None = None,
    show_legend: bool = False,
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
    show_legend :
        Whether to show a legend for the points.

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

    # Plot points.
    for pt in points:
        if pt.formatting_id not in FORMATTING:
            _console.print(
                f"[yellow]Warning: unknown formatting_id '{pt.formatting_id}' "
                f"— falling back to default style.[/]"
            )
        color, marker = FORMATTING.get(pt.formatting_id, (Palette.gray, "o"))
        ax.scatter(pt.x, pt.y, label=pt.label, color=color, marker=marker, s=60)

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

    # Maybe add legend.
    if show_legend:
        ax.legend(loc="lower right")

    # Relative padding around x data.
    xvals = [pt.x for pt in points]
    left, right = (
        existing_xlims
        if existing_xlims
        else ((min(xvals), max(xvals)))
    )
    if x_scale == "linear" and len(points) > 0:
        additional = (max(xvals) - min(xvals)) * x_pad
        left = 0
        right = max(right, max(xvals) + additional)
    elif x_scale == "log":
        factor = (max(xvals) / min(xvals)) ** x_pad
        left = min(left, min(xvals) / factor)
        right = max(right, max(xvals) * factor)
    ax.set_xlim(left, right)

    # Draw improvement arrow if requested.
    if improvement_arrow is not None:
        base_pts = [
            pt for pt in points if pt.label == improvement_arrow.base_label
        ]
        target_pts = [
            pt for pt in points if pt.label == improvement_arrow.target_label
        ]
        if not base_pts:
            _console.print(
                f"[yellow]Warning: improvement_arrow base_label "
                f"'{improvement_arrow.base_label}' not found — skipping arrow.[/]"
            )
        elif not target_pts:
            _console.print(
                f"[yellow]Warning: improvement_arrow target_label "
                f"'{improvement_arrow.target_label}' not found — skipping arrow.[/]"
            )
        else:
            base = base_pts[0]
            target = target_pts[0]
            ax.annotate(
                "",
                xy=(target.x, target.y),
                xytext=(base.x, base.y),
                arrowprops=dict(
                    arrowstyle="->", color=Palette.gray, lw=1.5, linestyle="--"
                ),
                zorder=-10,
            )
            parts: list[str] = []
            if base.x != 0:
                x_change = (target.x - base.x) / abs(base.x)
                direction = "←" if x_change < 0 else "→"
                parts.append(f"Violation {direction} {abs(x_change):.1%}")
            if base.y != 0:
                y_change = (target.y - base.y) / abs(base.y)
                direction = "↓" if y_change < 0 else "↑"
                parts.append(f"Cost {direction} {abs(y_change):.1%}")
            if parts:
                ax.text(
                    0.1,
                    0.1,
                    "\n".join(parts),
                    color=Palette.gray,
                    ha="left",
                    va="bottom",
                    transform=ax.transAxes,
                )

    fig.tight_layout()
    return fig, ax, (left, right), (bottom, top)


def cli_cost_vs_compliance_scatter(
    entries: list[tuple[str, AggregatedExecutionResults]],
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
