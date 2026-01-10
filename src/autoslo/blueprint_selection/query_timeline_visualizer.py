from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import plotly.graph_objects as go

from autoslo.blueprints.cluster import Cluster
from autoslo.workload_execution.trace import Trace


@dataclass(frozen=True)
class GanttSnapshot:
    label: str
    # cluster_name -> list of (begin, end, metadata dict)
    intervals_by_cluster: dict[str, list[tuple[float, float, dict[str, Any]]]]
    # cluster_name -> display label (or RPU)
    cluster_label_by_name: dict[str, str]
    # Metrics captured at snapshot time
    total_queries: int
    violating_queries: int
    violation_rate: float
    cost_per_cluster: dict[str, float] = None
    total_cost: float = 0.0


class GanttRecorder:
    def __init__(self) -> None:
        self.snapshots: list[GanttSnapshot] = []

    def snapshot(
        self,
        obj: Any,
        label: Optional[str] = None,
        slo_s: Optional[float | dict[str, float]] = None,
    ) -> None:
        """
        Capture just enough state from `obj` to later draw the gantt chart.

        Expects:
          - obj.active_clusters: Iterable[str]
          - obj._interval_trees[cluster_name] yields intervals with .begin, .end, .data["query_id"]
          - Cluster.from_config(cluster_name).rpu exists (or adjust below)
        """
        intervals_by_cluster: dict[
            str, list[tuple[float, float, dict[str, Any]]]
        ] = {}
        cluster_label_by_name: dict[str, str] = {}

        # Freeze what we need right now
        active_clusters: Iterable[str] = list(obj.active_clusters)

        for cluster_name in active_clusters:
            tree = obj._interval_trees[cluster_name]
            intervals = sorted((iv.begin, iv.end, dict(iv.data)) for iv in tree)
            intervals_by_cluster[cluster_name] = intervals

            rpu = Cluster.from_config(cluster_name).rpu
            cluster_label_by_name[cluster_name] = f"{cluster_name}"

        if label is None:
            label = str(len(self.snapshots))

        # Compute snapshot metrics
        total_queries = len(getattr(obj, "query_ids", []))
        violating_queries = 0
        if slo_s is not None and total_queries > 0:
            for qid in obj.query_ids:
                interval = obj.interval_for_query_id(qid)
                latency = interval.end - interval.begin
                slo_val = (
                    slo_s
                    if isinstance(slo_s, float)
                    else slo_s.get(qid, float("inf"))
                )
                if latency > slo_val:
                    violating_queries += 1
            violation_rate = violating_queries / total_queries
        else:
            violation_rate = 0.0

        total_cost = 0.0
        try:
            total_cost = obj.total_cost()
        except Exception:
            total_cost = 0.0

        cost_per_cluster = {}
        try:
            cost_per_cluster = obj.cost_per_cluster()
        except Exception:
            cost_per_cluster = {}

        self.snapshots.append(
            GanttSnapshot(
                label=label,
                intervals_by_cluster=intervals_by_cluster,
                cluster_label_by_name=cluster_label_by_name,
                total_queries=total_queries,
                violating_queries=violating_queries,
                violation_rate=violation_rate,
                total_cost=total_cost,
                cost_per_cluster=cost_per_cluster,
            )
        )


def _pack_into_lanes(
    sorted_intervals: list[tuple[float, float, dict[str, Any]]],
):
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
    lane_idx: int,
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
    slo_val: Optional[float] = (
        slo_s
        if isinstance(slo_s, float)
        else (slo_s.get(qid) if isinstance(slo_s, dict) else None)
    )
    header = f"Cluster: {cluster_name}<br>Query: {pure_qid}"
    timing = f"Start: {s:.3f}s<br>End: {e:.3f}s<br>Duration: {dur:.3f}s"
    slo = (
        f"SLO: {slo_val:.3f}s"
        if isinstance(slo_val, (float, int))
        else "SLO: -"
    )
    meta_lines: list[str] = []
    # for k, v in meta.items():
    #     if k == "query_id":
    #         continue
    #     meta_lines.append(f"{k}: {v}")
    meta_block = "<br>".join(meta_lines)
    if meta_block:
        return f"{header}<br>{timing}<br>{slo}<br><br>{meta_block}"
    return f"{header}<br>{timing}<br>{slo}"


def _build_shapes_for_snapshot(
    snap: Any,
    slo_s: float | dict[str, float],
    layout_plan: Optional[dict[str, Any]] = None,
):
    # Global min/max time for relative axis
    min_time = float("inf")
    max_time = -float("inf")

    # Determine min/max across all clusters
    for intervals in snap.intervals_by_cluster.values():
        if not intervals:
            continue
        min_time = min(min_time, intervals[0][0])
        max_time = max(max_time, intervals[-1][1])

    if min_time == float("inf"):
        # No intervals in this snapshot; still respect constant layout if provided
        if layout_plan is not None:
            return dict(
                shapes=[],
                y_ticks=layout_plan["y_ticks"],
                y_labels=layout_plan["y_labels"],
                x_max=1,
                y_max=layout_plan["y_max"],
                hover_traces=[],
            )
        return dict(
            shapes=[],
            y_ticks=[],
            y_labels=[],
            x_max=1,
            y_max=1,
            hover_traces=[],
        )

    shapes: list[dict[str, Any]] = []
    hover_traces: list[go.Scatter] = []
    # y_ticks will be used for gridline positions (boundaries)
    y_ticks: list[float] = []
    # y_labels and y_label_centers for custom text traces
    y_labels: list[str] = []
    y_label_centers: list[float] = []
    cluster_names: list[str] = []
    y_pos = 0

    # Determine iteration order: dynamic (snapshot dict order) or global plan
    if layout_plan is not None:
        cluster_iter = [
            c
            for c in layout_plan["cluster_order"]
            if layout_plan["max_lanes_by_cluster"].get(c, 0) > 0
        ]
    else:
        # Sort clusters by RPU ascending so fewer RPU appear lower
        present = [c for c, ints in snap.intervals_by_cluster.items() if ints]
        cluster_iter = sorted(present, key=lambda c: Cluster.from_config(c).rpu)

    for cluster_name in cluster_iter:
        intervals = snap.intervals_by_cluster.get(cluster_name, [])
        lanes = _pack_into_lanes(intervals) if intervals else []

        for lane_idx, lane in enumerate(lanes):
            for s, e, meta in lane:
                rel_s = s - min_time
                rel_e = e - min_time

                qid = meta.get("query_id", "")
                slo_rel_e = rel_s + (
                    slo_s if isinstance(slo_s, float) else slo_s[qid]
                )

                if layout_plan is not None:
                    base = layout_plan["base_offset_by_cluster"][cluster_name]
                    y0 = base + lane_idx + 0.4
                    y1 = base + lane_idx + 1.2
                else:
                    y0 = y_pos + lane_idx + 0.4
                    y1 = y_pos + lane_idx + 1.2

                is_over_slo = rel_e > slo_rel_e
                fill = (
                    "rgba(255,0,0,1.0)" if is_over_slo else "rgba(0,128,0,0.6)"
                )
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=rel_s,
                        x1=rel_e,
                        y0=y0,
                        y1=y1,
                        line=dict(color="black", width=1),
                        fillcolor=fill,
                        layer="above",
                    )
                )

                hover_traces.append(
                    go.Scatter(
                        x=[rel_s, rel_e, rel_e, rel_s, rel_s],
                        y=[y0, y0, y1, y1, y0],
                        mode="lines",
                        fill="toself",
                        fillcolor="rgba(0,0,0,0.001)",
                        line=dict(width=0),
                        hoverinfo="text",
                        hovertemplate="%{text}",
                        text=_format_hover_text(
                            cluster_name, lane_idx, rel_s, rel_e, meta, slo_s
                        ),
                        showlegend=False,
                    )
                )

        if layout_plan is None:
            top = y_pos
            bottom = y_pos + len(lanes)
            y_ticks.append(top)  # Only add top boundary
            y_label_centers.append(top + (len(lanes) / 2))
            y_labels.append(
                snap.cluster_label_by_name.get(cluster_name, cluster_name)
            )
            cluster_names.append(cluster_name)
            y_pos = bottom

    # Add final bottom boundary for dynamic layout
    if layout_plan is None and y_pos > 0:
        y_ticks.append(y_pos)

    if layout_plan is not None:
        # Build boundary ticks from layout plan
        grid_ticks: list[float] = []
        label_centers: list[float] = []
        lbls: list[str] = []
        c_names: list[str] = []
        for c in layout_plan["cluster_order"]:
            ml = layout_plan["max_lanes_by_cluster"].get(c, 0)
            if ml <= 0:
                continue
            top = layout_plan["base_offset_by_cluster"][c]
            bottom = top + ml
            grid_ticks.append(top)  # Only add top boundary
            label_centers.append(top + (ml / 2))
            lbls.append(snap.cluster_label_by_name.get(c, c))
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
            x_max=(max_time - min_time + 1),
            y_max=layout_plan["y_max"],
            hover_traces=hover_traces,
        )
    else:
        return dict(
            shapes=shapes,
            y_ticks=y_ticks,
            y_labels=y_labels,
            y_label_centers=y_label_centers,
            cluster_names=cluster_names,
            x_max=(max_time - min_time + 1),
            y_max=y_pos,
            hover_traces=hover_traces,
        )


def render_gantt_scrubber(
    snapshots: list[Any],
    slo_s: float | dict[str, float],
    title: str = "Cluster Query Assignments Over Time",
    constant_layout: bool = False,
    violation_rate_threshold: Optional[float] = None,
) -> go.Figure:
    if not snapshots:
        raise ValueError("No snapshots to render")

    if constant_layout:
        # Build global layout plan
        cluster_order: list[str] = []
        seen: set[str] = set()
        label_by_name: dict[str, str] = {}
        for snap in snapshots:
            for c in snap.intervals_by_cluster.keys():
                if c not in seen:
                    cluster_order.append(c)
                    seen.add(c)
            label_by_name.update(snap.cluster_label_by_name)

        # Sort by RPU ascending (fewest RPU at bottom)
        cluster_order = sorted(
            cluster_order, key=lambda c: Cluster.from_config(c).rpu
        )

        max_lanes_by_cluster: dict[str, int] = {c: 0 for c in cluster_order}
        for c in cluster_order:
            for snap in snapshots:
                intervals = snap.intervals_by_cluster.get(c, [])
                lane_count = (
                    len(_pack_into_lanes(intervals)) if intervals else 0
                )
                if lane_count > max_lanes_by_cluster[c]:
                    max_lanes_by_cluster[c] = lane_count

        # Compute fixed y-axis ticks/labels and base offsets
        y_ticks: list[float] = []
        y_labels: list[str] = []
        base_offset_by_cluster: dict[str, int] = {}
        y_pos = 0
        for c in cluster_order:
            ml = max_lanes_by_cluster.get(c, 0)
            if ml <= 0:
                continue
            base_offset_by_cluster[c] = y_pos
            y_ticks.append(y_pos + (ml - 1) / 2)
            y_labels.append(label_by_name.get(c, c))
            y_pos += ml + 1

        layout_plan = dict(
            cluster_order=cluster_order,
            max_lanes_by_cluster=max_lanes_by_cluster,
            base_offset_by_cluster=base_offset_by_cluster,
            y_ticks=y_ticks,
            y_labels=y_labels,
            y_max=y_pos,
        )

        specs = [
            _build_shapes_for_snapshot(s, slo_s, layout_plan=layout_plan)
            for s in snapshots
        ]
        base = specs[0]
    else:
        specs = [_build_shapes_for_snapshot(s, slo_s) for s in snapshots]
        base = specs[0]

    # Build per-snapshot annotations
    def _ann_for_snap(snap: GanttSnapshot):
        txt = f"<b>SLO:</b> {snap.violation_rate*100:.1f}% ({snap.violating_queries}/{snap.total_queries}) • <b>Cost:</b> ${snap.total_cost:,.2f}"
        color = (
            (
                "red"
                if (
                    violation_rate_threshold is not None
                    and snap.violation_rate > violation_rate_threshold
                )
                else "green"
            )
            if violation_rate_threshold is not None
            else "black"
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
            hover = f"Cluster: {cluster_name}<br>RPU: {Cluster.from_config(cluster_name).rpu}<br>Queries: {q_count}<br>SLO Viol.: {viol_rate*100:.1f}% ({viol}/{q_count})<br>Cost: ${snap.cost_per_cluster.get(cluster_name, 0.0):,.2f}"
            color = "#cccccc" if q_count == 0 else "#444444"
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

    base_label_traces = _build_label_traces(base, snapshots[0])

    fig = go.Figure(
        data=base["hover_traces"] + base_label_traces,
        layout=go.Layout(
            title=title,
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
                            method="animate",
                            args=[
                                [f"f{i}"],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=getattr(snapshots[i], "label", str(i)),
                        )
                        for i in range(len(snapshots))
                    ],
                )
            ],
        ),
        frames=[
            go.Frame(
                name=f"f{i}",
                data=spec["hover_traces"]
                + _build_label_traces(
                    (base if constant_layout else spec), snapshots[i]
                ),
                layout=go.Layout(
                    shapes=spec["shapes"],
                    xaxis=dict(range=[-500, spec["x_max"] + 500]),
                    yaxis=dict(
                        tickmode="array",
                        tickvals=(
                            base["y_ticks"]
                            if constant_layout
                            else spec["y_ticks"]
                        ),
                        ticktext=(
                            base["y_labels"]
                            if constant_layout
                            else spec["y_labels"]
                        ),
                        range=[
                            -1,
                            max(
                                1,
                                (
                                    base["y_max"]
                                    if constant_layout
                                    else spec["y_max"]
                                ),
                            ),
                        ],
                        showticklabels=False,
                    ),
                    annotations=_ann_for_snap(snapshots[i]),
                ),
            )
            for i, spec in enumerate(specs)
        ],
    )
    return fig
