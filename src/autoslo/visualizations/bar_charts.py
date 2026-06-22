from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from autoslo.slo.slo_metric import SloMetric
from autoslo.visualizations.colors import Palette
from autoslo.visualizations.scatter_plots import FORMATTING, ScatterPoint, ThresholdLine


def violation_rate_bar_chart(
    points: Sequence[ScatterPoint],
    *,
    x_metric: str | SloMetric,
    title: str | None = None,
    threshold_lines: Sequence[ThresholdLine] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal bar chart of SLO violation rates with cost annotations.

    Each method is a horizontal bar whose length equals its SLO violation
    rate.  A text annotation placed just to the right of each bar tip shows
    ``{rate:.3f},  ${cost:.2f}``.  Dashed vertical threshold lines from
    *threshold_lines* are drawn in the same style as :func:`cost_vs_compliance_scatter`.

    The y-axis labels are the method names; no legend is generated.

    Parameters
    ----------
    points:
        Data points to plot.  Order is preserved top-to-bottom.
    x_metric:
        The SLO metric or column name whose value determines bar length.
    title:
        Optional axes title.
    threshold_lines:
        Optional dashed vertical threshold lines.
    ax:
        Pre-existing axes to draw on.  When *None* a new figure is created.
    figsize:
        Figure size when creating a new figure (ignored when *ax* is given).
        Defaults to ``(8, max(2.0, 0.55 * n + 0.9))`` so the chart scales
        with the number of methods.

    Returns
    -------
    (fig, ax)
        The matplotlib figure and axes.
    """
    n = len(points)

    if ax is None:
        if figsize is None:
            figsize = (8, max(2.0, 0.55 * n + 0.9))
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(Palette.white)
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    ax.set_facecolor(Palette.white)

    x_metric_obj = SloMetric(x_metric) if isinstance(x_metric, str) else x_metric

    _palette_map = Palette.as_colormap()
    bar_height = 0.55

    # Draw bars; reverse the list so the first point appears at the top.
    for i, pt in enumerate(reversed(points)):
        color, _ = FORMATTING.get(pt.formatting_id, (Palette.gray, "o"))
        ax.barh(
            i,
            pt.x,
            height=bar_height,
            color=color,
            edgecolor=Palette.gray,
            linewidth=0.4,
            zorder=2,
        )
        # Tip annotation: violation rate + cost.
        ax.text(
            pt.x,
            i,
            f"  {pt.x:.3f},  ${pt.y:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=Palette.gray,
            zorder=3,
        )

    # Dashed vertical threshold lines (same style as cost_vs_compliance_scatter).
    if threshold_lines:
        for tl in threshold_lines:
            resolved_color = _palette_map.get(tl.color, tl.color)
            ax.axvline(
                tl.value,
                color=resolved_color,
                linestyle="--",
                linewidth=0.7,
                zorder=1,
            )
            if tl.label is not None:
                ax.text(
                    tl.value,
                    0.97,
                    tl.label,
                    color=resolved_color,
                    rotation=90,
                    ha="right",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    fontsize=9,
                )

    # Y-axis: method names, first point at the top.
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels([pt.label for pt in reversed(points)])
    ax.set_ylim(-0.5, n - 0.5)

    # X-axis: leave room to the right for the tip annotations.
    xmax = max((pt.x for pt in points), default=0.1)
    ax.set_xlim(0, xmax * 1.6)
    ax.set_xlabel(x_metric_obj.to_plot_axis_label())

    if title:
        ax.set_title(title)

    ax.tick_params(colors=Palette.gray)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, ax
