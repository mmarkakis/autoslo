from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from rich.console import Console

from autoslo.filesystem.structured_events import EventType
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.visualizations.colors import Palette
from autoslo.visualizations.scatter_plots import FORMATTING

_console = Console()


@dataclass
class SpinupRecord:
    """One cluster spin-up event pair (autoscaler decision → cluster ready)."""

    cluster_name: str
    rpu: int
    decision_time_s: float | None  # None for initial (pre-workload) clusters
    ready_time_s: float


@dataclass
class SpinupLane:
    """All spinup events for one method within a single timeline panel."""

    label: str
    formatting_id: str
    spinups: list[SpinupRecord] = field(default_factory=list)
    run_duration_s: float = 0.0
    last_arrival_s: float | None = None
    last_completion_s: float | None = None


def render_spinup_timeline(
    ax: Axes,
    lanes: list[SpinupLane],
    run_duration_s: float,
) -> None:
    """Draw horizontal spinup lanes on *ax* with the x-axis in minutes.

    Each lane corresponds to one method.  For every cluster spin-up the lane
    shows:

    * A downward-triangle marker (▼) at the autoscaler trigger time (omitted
      for initial clusters that have no corresponding decision event).
    * An arrow from trigger to ready.
    * A marker (shape from ``FORMATTING``) at the ready time.
    * The RPU size annotated just above the ready marker.

    Colors and marker shapes are taken from the same ``FORMATTING`` table used
    by the scatter plots, keyed by each lane's ``formatting_id``.
    """
    time_scale = 60.0  # convert seconds to minutes for the x-axis

    for lane_idx, lane in enumerate(lanes):
        y = float(lane_idx)
        ax.axhline(y, color="#cccccc", linewidth=0.5, zorder=0)

        color, method_marker = FORMATTING.get(
            lane.formatting_id, (Palette.gray, "o")
        )

        for spinup in lane.spinups:
            ready_x = spinup.ready_time_s / time_scale

            if spinup.decision_time_s is not None:
                dec_x = spinup.decision_time_s / time_scale
                # Arrow-line from trigger to ready.
                ax.annotate(
                    "",
                    xy=(ready_x, y),
                    xytext=(dec_x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4),
                    zorder=1,
                )
                # Trigger marker: fixed downward-triangle across all methods.
                ax.scatter(
                    dec_x, y, marker="v", color=color, s=45,
                    zorder=2, clip_on=False,
                )

            # Ready marker: uses the method's formatting marker.
            ax.scatter(
                ready_x, y, marker=method_marker, color=color, s=60,
                zorder=3, clip_on=False,
            )
            # RPU label just above the ready marker.
            ax.text(
                ready_x,
                y + 0.22,
                str(spinup.rpu),
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                fontweight="bold",
                zorder=4,
            )

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([lane.label for lane in lanes], fontsize=8)
    ax.set_ylim(-0.65, max(len(lanes) - 0.35, 0.35))
    ax.set_xlim(0, (run_duration_s / time_scale) * 1.03)
    ax.set_xlabel("Time (min)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)  # lane labels are sufficient; no tick marks needed


def load_spinup_lane(
    run_dir: Path, label: str, formatting_id: str
) -> SpinupLane | None:
    """Load spinup-event timeline data from *run_dir*'s structured log.

    Returns *None* when the log is absent or contains no useful events.
    """
    log_path = run_dir / "structured_log.parquet"
    if not log_path.exists():
        _console.print(f"[yellow]No structured log at {run_dir}[/]")
        return None

    df = StructuredLog.load(log_path).df

    # Workload duration.
    lifecycle = df[df["event_type"].isin(["run_start", "run_finish"])]
    if len(lifecycle) >= 2:
        run_duration_s = float(
            lifecycle["rel_time_s"].max() - lifecycle["rel_time_s"].min()
        )
    elif not lifecycle.empty:
        run_duration_s = float(lifecycle["rel_time_s"].max())
    else:
        run_duration_s = 0.0

    # Collect (time, rpu) tuples for each decision event.
    dec_rows = df[df["event_type"] == EventType.SPIN_UP_DECISION.value]
    decisions: list[tuple[float, int]] = []
    for _, row in dec_rows.iterrows():
        d = row.get("details") or {}
        rpu = d.get("rpu") if isinstance(d, dict) else None
        if rpu is not None:
            decisions.append((float(row["rel_time_s"]), int(rpu)))

    # Collect (time, rpu, cluster_name) tuples for each ready event.
    ready_rows = df[df["event_type"] == EventType.CLUSTER_READY.value]
    ready_events: list[tuple[float, int, str]] = []
    for _, row in ready_rows.iterrows():
        d = row.get("details") or {}
        rpu = d.get("rpu") if isinstance(d, dict) else None
        cluster_name = str(row.get("cluster_name") or "")
        if rpu is not None:
            ready_events.append((float(row["rel_time_s"]), int(rpu), cluster_name))

    # Pair each decision with the closest prior ready event of the same RPU.
    # Initial clusters have a ready event but no preceding decision.
    decision_pool: dict[int, list[float]] = {}
    for t, rpu in sorted(decisions):
        decision_pool.setdefault(rpu, []).append(t)

    spinups: list[SpinupRecord] = []
    for ready_time, rpu, cluster_name in sorted(ready_events):
        pool = decision_pool.get(rpu, [])
        eligible = [t for t in pool if t <= ready_time]
        if eligible:
            decision_time: float | None = max(eligible)
            assert decision_time is not None
            pool.remove(decision_time)
        else:
            decision_time = None  # initial cluster — no autoscaler decision
        spinups.append(
            SpinupRecord(
                cluster_name=cluster_name,
                rpu=rpu,
                decision_time_s=decision_time,
                ready_time_s=ready_time,
            )
        )

    spinups.sort(key=lambda s: s.ready_time_s)

    # Latest query arrival (covers both live and simulator event names).
    arrival_rows = df[
        df["event_type"].isin(
            [EventType.ARRIVAL.value, EventType.SIM_QUERY_ARRIVAL.value]
        )
    ]
    last_arrival_s: float | None = (
        float(arrival_rows["rel_time_s"].max()) if not arrival_rows.empty else None
    )

    # Latest query completion.
    completion_rows = df[df["event_type"] == EventType.COMPLETION.value]
    last_completion_s: float | None = (
        float(completion_rows["rel_time_s"].max())
        if not completion_rows.empty
        else None
    )

    return SpinupLane(
        label=label,
        formatting_id=formatting_id,
        spinups=spinups,
        run_duration_s=run_duration_s,
        last_arrival_s=last_arrival_s,
        last_completion_s=last_completion_s,
    )


def save_spinup_timeline_figure(
    panel_lanes: list[tuple[dict, list[SpinupLane]]],
    rows: int,
    cols: int,
    timeline_path: Path,
) -> None:
    """Render a grid of spinup-timeline panels and save to *timeline_path*.

    Parameters
    ----------
    panel_lanes :
        One entry per panel: ``(panel_spec_dict, lanes)``.  The panel spec
        dict must contain ``"row"`` and ``"col"`` positioning keys and an
        optional ``"title"`` string.
    rows, cols :
        Grid dimensions.
    timeline_path :
        Destination file path (PNG).
    """
    all_durations = [
        lane.run_duration_s for _, lanes in panel_lanes for lane in lanes
    ]
    shared_duration_s = max(all_durations) if all_durations else 600.0

    max_lanes = max((len(lanes) for _, lanes in panel_lanes), default=1)
    panel_height_in = max(2.0, max_lanes * 0.65 + 0.8)
    fig, axes_2d = plt.subplots(
        rows,
        cols,
        figsize=(6.5 * cols, panel_height_in * rows),
        squeeze=False,
    )

    seen_positions: set[tuple[int, int]] = set()
    for panel, lanes in panel_lanes:
        row = panel.get("row", 0)
        col = panel.get("col", 0)
        seen_positions.add((row, col))
        ax: Axes = axes_2d[row][col]
        render_spinup_timeline(ax, lanes, shared_duration_s)
        title = panel.get("title")
        if title:
            ax.set_title(title, fontsize=9)

        # Vertical dashed lines at the latest query arrival and completion
        # across all methods in this panel.
        time_scale = 60.0
        arrivals = [
            lane.last_arrival_s
            for lane in lanes
            if lane.last_arrival_s is not None
        ]
        completions = [
            lane.last_completion_s
            for lane in lanes
            if lane.last_completion_s is not None
        ]
        if arrivals:
            ax.axvline(
                max(arrivals) / time_scale,
                color="#888888",
                linestyle="--",
                linewidth=0.9,
                zorder=0,
            )
        if completions:
            ax.axvline(
                max(completions) / time_scale,
                color="#444444",
                linestyle=(0, (3, 1, 1, 1)),
                linewidth=0.9,
                zorder=0,
            )

    # Hide unused axes.
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in seen_positions:
                axes_2d[r][c].set_visible(False)

    # Legend: trigger/ready markers + workload-boundary lines.
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="v", linestyle="None", color="#555555",
            markersize=6, label="Trigger",
        ),
        plt.Line2D(
            [0], [0], marker="o", linestyle="None", color="#555555",
            markersize=6, label="Ready",
        ),
        plt.Line2D(
            [0], [0], color="#888888", linestyle="--",
            linewidth=0.9, label="Last arrival",
        ),
        plt.Line2D(
            [0], [0], color="#444444", linestyle=(0, (3, 1, 1, 1)),
            linewidth=0.9, label="Last completion",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(timeline_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _console.print(f"[green]Saved:[/] {timeline_path}")
