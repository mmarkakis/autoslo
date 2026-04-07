from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import plotly.graph_objects as go
import plotly.io as pio
from intervaltree import Interval

from autoslo.blueprints.cluster import Cluster
from autoslo.slo.slo_objective import SloMetric
from autoslo.slo.slo_resolver import (
    query_interval,
    slo_relative_violation,
    slo_violation_amount_s,
    violates_slo,
)
from autoslo.utils.billing import Billing
from autoslo.utils.colors import Palette
from autoslo.workload_definition.query import Query
from autoslo.workload_execution.trace import Trace


@dataclass(frozen=True)
class GanttSnapshot:
    label: str

    # cluster_name -> list of (begin, end, metadata dict)
    intervals_by_cluster: dict[str, list[tuple[float, float, dict[str, Any]]]]

    # Metrics captured at snapshot time
    total_queries: int
    violating_queries: int
    violation_rate: float
    cost_per_cluster: dict[str, float]
    total_cost: float
    violation_amount: float
    violation_relative: float


class GanttRecorder:

    MET_COLOR = Palette.light_green
    RUNNING_COLOR = Palette.light_gray
    MISSED_COLOR = Palette.dark_red

    def __init__(self) -> None:
        self.snapshots: list[GanttSnapshot] = []

    def snapshot(
        self,
        cost_per_second_per_cluster: dict[str, float],
        completed_queries_per_cluster: dict[str, list[Query]],
        active_queries_per_cluster: dict[str, list[Query]],
        label: str,
        slo_s: float,
        latencies: dict[str, float] | None = None,
    ) -> None:
        """
        Capture state to later draw the gantt chart.
        """
        if latencies is None:
            latencies = {}

        intervals_by_cluster: dict[
            str, list[tuple[float, float, dict[str, Any]]]
        ] = {}

        total_queries = 0
        violating_queries = 0
        violation_amount = 0.0
        violation_relative_sum = 0.0

        def _add_intervals(
            queries: Iterable[Query], state: str
        ) -> list[tuple[float, float, dict[str, Any]]]:
            nonlocal total_queries, violating_queries, violation_amount
            nonlocal violation_relative_sum

            intervals: list[tuple[float, float, dict[str, Any]]] = []
            for q in queries:
                total_queries += 1
                lat = latencies.get(q.query_id, 0.0)
                ref_interval = query_interval(
                    q.rel_start_time_s, lat, q.query_id
                )
                color = (
                    self.RUNNING_COLOR
                    if state == "RUNNING"
                    else (
                        self.MISSED_COLOR
                        if violates_slo(lat, slo_s)
                        else self.MET_COLOR
                    )
                )
                intervals.append(
                    (
                        ref_interval.begin,
                        ref_interval.end,
                        ref_interval.data | {"state": state, "color": color},
                    )
                )
                violating_queries += violates_slo(lat, slo_s)
                violation_amount += slo_violation_amount_s(lat, slo_s)
                violation_relative_sum += slo_relative_violation(lat, slo_s)

            intervals.sort(key=lambda x: (x[0], x[1], x[2].get("query_id", "")))
            return intervals

        for (
            cluster_name,
            completed_queries,
        ) in completed_queries_per_cluster.items():
            intervals_by_cluster[cluster_name] = _add_intervals(
                completed_queries,
                "COMPLETED",
            )

        for cluster_name, active_queries in active_queries_per_cluster.items():
            intervals = _add_intervals(
                active_queries,
                "RUNNING",
            )
            intervals_by_cluster.setdefault(cluster_name, []).extend(intervals)

        cost_per_cluster: dict[str, float] = {}
        total_cost = 0.0
        for cluster_name in intervals_by_cluster.keys():
            intervalized = [
                Interval(iv[0], iv[1], iv[2])
                for iv in intervals_by_cluster[cluster_name]
            ]
            billed_s = Billing.billed_s(intervalized)
            cost = cost_per_second_per_cluster[cluster_name] * billed_s
            cost_per_cluster[cluster_name] = cost
            total_cost += cost

        violation_rate = (
            (violating_queries / total_queries) if total_queries > 0 else 0.0
        )
        violation_relative = (
            (violation_relative_sum / total_queries)
            if total_queries > 0
            else 0.0
        )

        self.snapshots.append(
            GanttSnapshot(
                label=label,
                intervals_by_cluster=intervals_by_cluster,
                total_queries=total_queries,
                violating_queries=violating_queries,
                violation_rate=violation_rate,
                total_cost=total_cost,
                cost_per_cluster=cost_per_cluster,
                violation_amount=violation_amount,
                violation_relative=violation_relative,
            )
        )


def _pack_into_lanes(
    sorted_intervals: list[tuple[float, float, dict[str, Any]]],
) -> list[list[tuple[float, float, dict[str, Any]]]]:
    """
    Faster lane packing than your O(n^2) version:
    since intervals are sorted by start time, we only need to compare to the
    last end-time in each lane.
    """
    lanes: list[list[tuple[float, float, dict[str, Any]]]] = []
    lane_last_end: list[float] = []

    for s, e, meta in sorted_intervals:

        placed = False
        for i, last_end in enumerate(lane_last_end):
            if s >= last_end:
                lanes[i].append((s, e, meta))
                lane_last_end[i] = e
                placed = True
                break
        if not placed:
            lanes.append([(s, e, meta)])
            lane_last_end.append(e)

    return lanes


def _format_hover_text(
    cluster_name: str,
    s: float,
    e: float,
    meta: dict[str, Any],
    slo_s: float | dict[str, float],
) -> str:
    qid = meta.get("query_id", "")
    pure_qid = (
        Trace.redshift_query_id_from_query_id(qid) if type(qid) == str else qid
    )
    dur = e - s
    if isinstance(slo_s, dict):
        slo_s = slo_s.get(qid, -1)

    msg = (
        f"Cluster: {cluster_name}<br>"
        f"Query ID: {pure_qid}<br>"
        f"TPC-DS Temp and Q Idx: {meta['tpcds_temp_and_q_idx']}<br>"
        f"State: {meta['state']}<br>"
        f"Start: {s:.3f}s<br>"
        f"End: {e:.3f}s"
        + (" (Projected)<br>" if meta["state"] == "RUNNING" else "<br>")
        + f"Duration: {dur:.3f}s<br>"
        f"SLO: {slo_s:.3f}s<br>"
        f"Stage Prediction: {meta['stage_latency_prediction_s']:.3f}s<br>"
    )

    return msg


def _build_shapes_for_snapshot(
    snap: Any,
    slo_s: float | dict[str, float],
    layout_plan: dict[str, Any],
    snap_idx: int = 0,
):
    """Build rectangle shapes and hover points for a single snapshot."""
    shapes: list[dict[str, Any]] = []

    # Collect center points and hover text for each rectangle.
    # Using individual points (instead of polygons) ensures each rectangle
    # gets its own distinct hover text.
    hover_x: list[float] = []
    hover_y: list[float] = []
    hover_texts: list[str] = []

    # Only iterate clusters that have lanes to display.
    cluster_iter = [
        c
        for c in layout_plan["cluster_order"]
        if layout_plan["max_lanes_by_cluster"].get(c, 0) > 0
    ]

    lane_cache = layout_plan.get("lane_cache", {})
    for cluster_name in cluster_iter:
        intervals = snap.intervals_by_cluster.get(cluster_name, [])
        if not intervals:
            continue
        # Use cached lanes if available, otherwise compute
        cache_key = (snap_idx, cluster_name)
        if cache_key in lane_cache:
            lanes = lane_cache[cache_key]
        else:
            lanes = _pack_into_lanes(intervals)
        base = layout_plan["base_offset_by_cluster"][cluster_name]
        reference_time = layout_plan["reference_time"]

        for lane_idx, lane in enumerate(lanes):
            for s, e, meta in lane:
                rel_s = s - reference_time
                rel_e = e - reference_time
                y0 = base + lane_idx + 0.1
                y1 = base + lane_idx + 0.9

                # Visible colored rectangle (rendered as a layout shape).
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=rel_s,
                        x1=rel_e,
                        y0=y0,
                        y1=y1,
                        line=dict(width=0),
                        fillcolor=meta["color"],
                        layer="above",
                    )
                )

                # Invisible hover point at rectangle center.
                hover_x.append((rel_s + rel_e) / 2)
                hover_y.append((y0 + y1) / 2)
                hover_texts.append(
                    _format_hover_text(cluster_name, rel_s, rel_e, meta, slo_s)
                )

    # Single scatter trace with one marker per rectangle for hover detection.
    # Always create exactly one trace (even if empty) to ensure consistent
    # trace count across all frames for proper animation.
    hover_traces: list[go.Scatter] = [
        go.Scatter(
            x=hover_x if hover_x else [],
            y=hover_y if hover_y else [],
            mode="markers",
            marker=dict(size=1, opacity=0),
            text=hover_texts if hover_texts else [],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    ]
    # Build boundary ticks from layout plan
    grid_ticks: list[float] = []
    label_centers: list[float] = []
    lbls: list[str] = []
    c_names: list[str] = []
    for c in layout_plan["cluster_order"]:
        ml = layout_plan["max_lanes_by_cluster"].get(c, 0)
        if ml <= 0:
            continue
        bottom = layout_plan["base_offset_by_cluster"][c]
        top = bottom + ml
        grid_ticks.append(top)  # Only add top boundary
        label_centers.append(bottom + (ml / 2))
        lbls.append(c)
        c_names.append(c)
    # Add final bottom boundary
    if grid_ticks:
        grid_ticks.append(layout_plan["y_max"])
    return dict(
        shapes=shapes,
        y_ticks=grid_ticks,
        y_labels=lbls,
        y_label_centers=label_centers,
        cluster_names=c_names,
        x_max=layout_plan["x_max"],
        y_max=layout_plan["y_max"],
        hover_traces=hover_traces,
    )


def render_gantt_scrubber(
    snapshots: list[Any],
    slo_s: float | dict[str, float],
    title: str = "Cluster Query Assignments Over Time",
    slo_metric: SloMetric = SloMetric.RELATIVE,
    slo_threshold: float = 0.0,
    workload_name: Optional[str] = None,
    include_animation_frames: bool = False,
) -> go.Figure:
    if not snapshots:
        raise ValueError("No snapshots to render")

    # For subtitle, require slo_s to be a single float
    if workload_name is not None and not isinstance(slo_s, float):
        raise ValueError(
            "Subtitle can only be generated when slo_s is a single float value, not a dict"
        )

    # Build global layout plan
    cluster_order: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        for c in snap.intervals_by_cluster.keys():
            if c not in seen:
                cluster_order.append(c)
                seen.add(c)

    # Sort by RPU ascending (fewest RPU at bottom)
    cluster_order = sorted(
        cluster_order, key=lambda c: Cluster.rpu_for_cluster_name(c)
    )

    max_lanes_by_cluster: dict[str, int] = {c: 0 for c in cluster_order}
    # Cache lane packing results to avoid redundant computation
    _lane_cache: dict[
        tuple[int, str], list[list[tuple[float, float, dict[str, Any]]]]
    ] = {}
    for c in cluster_order:
        for snap_idx, snap in enumerate(snapshots):
            intervals = snap.intervals_by_cluster.get(c, [])
            if intervals:
                lanes = _pack_into_lanes(intervals)
                _lane_cache[(snap_idx, c)] = lanes
                lane_count = len(lanes)
            else:
                lane_count = 0
            if lane_count > max_lanes_by_cluster[c]:
                max_lanes_by_cluster[c] = lane_count

    # Compute fixed y-axis ticks/labels and base offsets
    y_ticks: list[float] = []
    y_labels: list[str] = []
    y_label_centers: list[float] = []
    base_offset_by_cluster: dict[str, int] = {}
    y_pos = 0
    for c in cluster_order:
        ml = max_lanes_by_cluster.get(c, 0)
        if ml <= 0:
            continue
        base_offset_by_cluster[c] = y_pos
        y_ticks.append(y_pos)
        y_labels.append(c)
        y_label_centers.append(y_pos + ml / 2)
        y_pos += ml

    # Also compute a fixed width for the x axis based on the global max time
    # across all snapshots, so that the scrubber doesn't resize as you move
    # through snapshots of different durations.
    max_time = max(
        max(
            (
                interval[1]
                for intervals in snap.intervals_by_cluster.values()
                for interval in intervals
            ),
            default=0,
        )
        for snap in snapshots
    )

    layout_plan = dict(
        cluster_order=cluster_order,
        max_lanes_by_cluster=max_lanes_by_cluster,
        base_offset_by_cluster=base_offset_by_cluster,
        y_label_centers=y_label_centers,
        y_ticks=y_ticks,
        y_labels=y_labels,
        y_max=y_pos,
        x_max=max_time,
        reference_time=0,
        lane_cache=_lane_cache,
    )

    specs = [
        _build_shapes_for_snapshot(
            s, slo_s, layout_plan=layout_plan, snap_idx=i
        )
        for i, s in enumerate(snapshots)
    ]
    base = specs[0]

    # Build per-snapshot annotations
    def _ann_for_snap(snap: GanttSnapshot):
        # Build the primary text line based on the active metric.
        if slo_metric is SloMetric.ABSOLUTE_S:
            txt = (
                f"<b>Cumulative SLO Violation Time:</b> {snap.violation_amount:.1f}s "
                f"• <b>Cost:</b> ${snap.total_cost:,.2f}"
            )
        elif slo_metric is SloMetric.RELATIVE:
            txt = (
                f"<b>Mean Relative SLO Violation:</b> {snap.violation_relative*100:.2f}% "
                f"• <b>Cost:</b> ${snap.total_cost:,.2f}"
            )
        else:
            txt = (
                f"<b>SLO Violation Rate:</b> {snap.violation_rate*100:.1f}% "
                f"({snap.violating_queries}/{snap.total_queries}) • "
                f"<b>Cost:</b> ${snap.total_cost:,.2f}"
            )

        # Determine red/green colouring based on the threshold.
        color = Palette.black
        if slo_threshold is not None:
            if slo_metric is SloMetric.ABSOLUTE_S:
                metric_val = snap.violation_amount
            elif slo_metric is SloMetric.RELATIVE:
                metric_val = snap.violation_relative
            else:
                metric_val = snap.violation_rate
            color = (
                GanttRecorder.MISSED_COLOR
                if metric_val > slo_threshold
                else GanttRecorder.MET_COLOR
            )

        return [
            dict(
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.12,
                xanchor="right",
                yanchor="bottom",
                text=txt,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1,
                font=dict(size=14, color=color),
            )
        ]

    # Build custom y-axis label traces with hover tooltips
    def _build_label_traces(
        spec: dict[str, Any], snap: GanttSnapshot
    ) -> list[go.Scatter]:
        traces: list[go.Scatter] = []

        for y, label, cluster_name in zip(
            spec["y_label_centers"], spec["y_labels"], spec["cluster_names"]
        ):
            intervals = snap.intervals_by_cluster.get(cluster_name, [])
            q_count = len(intervals)
            # cluster violation rate for this snapshot
            viol = 0
            for s, e, meta in intervals:
                qid = meta.get("query_id", "")
                slo_val = (
                    slo_s
                    if isinstance(slo_s, float)
                    else slo_s.get(qid, float("inf"))
                )
                if (e - s) > slo_val:
                    viol += 1
            viol_rate = (viol / q_count) if q_count > 0 else 0.0
            hover = (
                f"Cluster: {cluster_name}<br>"
                f"RPU: {Cluster.rpu_for_cluster_name(cluster_name)}<br>"
                f"Queries: {q_count}<br>"
                f"SLO Viol.: {viol_rate*100:.1f}% ({viol}/{q_count})<br>"
                f"Cost: ${snap.cost_per_cluster.get(cluster_name, 0.0):,.2f}"
            )
            color = "#444444"
            traces.append(
                go.Scatter(
                    x=[-20],
                    y=[y],
                    mode="text+markers",
                    text=[label],
                    textposition="middle left",
                    textfont=dict(color=color, size=12),
                    marker=dict(
                        size=20, color="rgba(255,255,255,0)", symbol="square"
                    ),
                    hoverinfo="text",
                    hovertext=hover,
                    showlegend=False,
                    xaxis="x",
                    yaxis="y",
                    cliponaxis=False,
                )
            )
        return traces

    # Pre-compute all label traces to avoid redundant computation in slider steps
    all_label_traces = [
        _build_label_traces(specs[i], snapshots[i])
        for i in range(len(snapshots))
    ]
    base_label_traces = all_label_traces[0]

    # Pre-extract data from traces for slider updates to avoid repeated attribute access
    precomputed_slider_data = []
    for i in range(len(snapshots)):
        combined_traces = specs[i]["hover_traces"] + all_label_traces[i]
        precomputed_slider_data.append(
            {
                "x": [t.x for t in combined_traces],
                "y": [t.y for t in combined_traces],
                "text": [t.text for t in combined_traces],
                "hovertext": [
                    getattr(t, "hovertext", None) for t in combined_traces
                ],
            }
        )

    # Build title dict with optional subtitle
    title_dict: dict[str, Any] = dict(text=title)
    if workload_name is not None and isinstance(slo_s, float):
        title_dict = dict(
            text=title,
            subtitle={"text": f"Workload: {workload_name} | SLO: {slo_s:.2f}s"},
        )

    fig = go.Figure(
        data=base["hover_traces"] + base_label_traces,
        layout=go.Layout(
            title=title_dict,
            template="plotly_white",
            xaxis=dict(
                title="Time since start (s)", range=[-500, base["x_max"] + 500]
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=base["y_ticks"],
                ticktext=base["y_labels"],
                range=[-1, base["y_max"]],
                showticklabels=False,
                fixedrange=True,
            ),
            shapes=base["shapes"],
            margin=dict(t=100, r=20, b=80, l=150),
            annotations=_ann_for_snap(snapshots[0]),
            sliders=[
                dict(
                    active=0,
                    x=0.08,
                    y=-0.08,
                    len=0.9,
                    currentvalue=dict(prefix="Snapshot: "),
                    steps=[
                        dict(
                            method="update",
                            args=[
                                # Update data traces (using precomputed data)
                                precomputed_slider_data[i],
                                # Update layout
                                {
                                    "shapes": specs[i]["shapes"],
                                    "annotations": _ann_for_snap(snapshots[i]),
                                },
                            ],
                            label=getattr(snapshots[i], "label", str(i)),
                        )
                        for i in range(len(snapshots))
                    ],
                )
            ],
        ),
    )

    # Only generate frames if animation playback is needed.
    # The slider works without frames (via the 'update' method in steps).
    # Frame generation is expensive for large numbers of snapshots.
    if include_animation_frames:
        fig.frames = [
            go.Frame(
                name=f"f{i}",
                data=spec["hover_traces"] + all_label_traces[i],
                layout=go.Layout(
                    shapes=spec["shapes"],
                    xaxis=dict(range=[-500, spec["x_max"] + 500]),
                    yaxis=dict(
                        tickmode="array",
                        tickvals=(base["y_ticks"]),
                        ticktext=(base["y_labels"]),
                        range=[
                            -1,
                            max(
                                1,
                                (base["y_max"]),
                            ),
                        ],
                        showticklabels=False,
                    ),
                    annotations=_ann_for_snap(snapshots[i]),
                ),
            )
            for i, spec in enumerate(specs)
        ]

    return fig


def export_gantt_video(
    snapshots: list[GanttSnapshot],
    slo_s: float | dict[str, float],
    output_path: str | Path,
    frame_duration: float = 1.0,
    title: str = "Cluster Query Assignments Over Time",
    constant_layout: bool = False,
    slo_metric: SloMetric = SloMetric.RELATIVE,
    slo_threshold: float = 0.0,
    workload_name: Optional[str] = None,
    width: int = 1400,
    height: int = 700,
) -> None:
    """
    Export snapshots as a video where each snapshot is a frame.

    Args:
        snapshots: List of GanttSnapshot objects
        slo_s: SLO threshold(s) in seconds
        output_path: Path to save the video file (e.g., "output.mp4")
        frame_duration: Duration each frame is displayed in seconds (default: 1.0)
        title: Title for the Gantt chart
        constant_layout: Whether to use constant layout across snapshots
        slo_metric: Which SLO-violation metric to highlight
        slo_threshold: Threshold for red/green colouring
        workload_name: Optional workload name for subtitle
        width: Video width in pixels (default: 1400)
        height: Video height in pixels (default: 700)

    Raises:
        ValueError: If snapshots list is empty or imageio is not installed
    """
    try:
        import imageio
    except ImportError:
        raise ValueError(
            "imageio is required for video export. "
            "Install it with: pip install imageio imageio-ffmpeg"
        )

    # Check if kaleido is available
    try:
        import kaleido
    except ImportError:
        raise ValueError(
            "kaleido is required for video export to render plots as images. "
            "Install it with: pip install kaleido"
        )

    if not snapshots:
        raise ValueError("No snapshots to render for video export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate FPS from frame duration
    fps = 1.0 / frame_duration if frame_duration > 0 else 1.0

    # Render each snapshot as a static image
    frames = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, snap in enumerate(snapshots):
            # Build a minimal figure for this snapshot using the same logic as render_gantt_scrubber
            fig = render_gantt_scrubber(
                snapshots=[snap],
                slo_s=slo_s,
                title=title,
                slo_metric=slo_metric,
                slo_threshold=slo_threshold,
                workload_name=workload_name,
            )

            # Convert to static image using plotly.io which handles kaleido properly
            temp_image_path = str(Path(temp_dir) / f"frame_{i:04d}.png")
            try:
                pio.write_image(
                    fig,
                    temp_image_path,
                    width=width,
                    height=height,
                    engine="kaleido",
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to render snapshot '{snap.label}' to image. "
                    f"Error: {e}. "
                    f"Make sure kaleido is properly installed: pip install --force-reinstall kaleido"
                )

            # Read the image back
            try:
                img = imageio.imread(temp_image_path)
                frames.append(img)
            except Exception as e:
                raise ValueError(
                    f"Failed to read rendered image for snapshot '{snap.label}'. Error: {e}"
                )

    # Write frames to video file
    try:
        imageio.mimsave(
            str(output_path),
            frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )
    except Exception as e:
        raise ValueError(
            f"Failed to write video file. Error: {e}. "
            f"Ensure ffmpeg is installed and imageio-ffmpeg is available."
        )

    print(f"Video exported successfully to {output_path}")
    print(f"  Snapshots: {len(snapshots)}")
    print(f"  Frame duration: {frame_duration}s")
    print(f"  Total duration: {len(snapshots) * frame_duration:.1f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
