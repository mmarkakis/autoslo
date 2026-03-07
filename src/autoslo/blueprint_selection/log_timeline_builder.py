"""
log_timeline_builder.py
-----------------------
Reconstruct Gantt timeline snapshots from a simulator solve log, without
needing any live state capture during the simulation.

Primary entry points
--------------------
build_final_snapshot_from_log  – single final-state GanttSnapshot
build_scrubber_snapshots_from_log – one GanttSnapshot per query arrival
                                    (implemented but not yet wired to the UI)
render_run                     – convenience wrapper for notebooks / scripts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import yaml
from intervaltree import Interval

from autoslo.blueprint_selection.query_timeline_visualizer_2 import (
    GanttRecorder,
    GanttSnapshot,
    render_gantt_scrubber,
)
from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.utils.billing import Billing
from autoslo.utils.colors import Palette

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_run(run_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    """Load config.yml + the structured/solve log from a run directory."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.yml") as f:
        config = yaml.safe_load(f)
    log = pd.read_parquet(os.path.join(run_dir, "structured_log.parquet"))
    return config, log


def _color(
    state: str,
    duration: float,
    resolver: SloResolver,
    tpcds_temp_and_q_idx: str | None = None,
) -> str:
    if state == "RUNNING":
        return Palette.light_gray
    slo_s = resolver.resolve(tpcds_temp_and_q_idx)
    return Palette.dark_red if duration > slo_s else Palette.light_green


def _snapshot_from_query_dicts(
    query_rows: list[dict[str, Any]],
    resolver: SloResolver,
    cost_per_second_per_cluster: dict[str, float],
    label: str,
) -> GanttSnapshot:
    """
    Build a GanttSnapshot from a flat list of per-query dicts.  Each dict must
    have: query_id, tpcds_temp_and_q_idx, cluster_name, start_s, end_s, state.
    """
    intervals_by_cluster: dict[
        str, list[tuple[float, float, dict[str, Any]]]
    ] = {}
    total_queries = 0
    violating_queries = 0
    violation_amount = 0.0
    violation_relative_sum = 0.0

    for row in query_rows:
        cluster = row["cluster_name"]
        s = row["start_s"]
        e = row["end_s"]
        state = row["state"]
        duration = e - s
        tpcds = row.get("tpcds_temp_and_q_idx", "")
        color = _color(state, duration, resolver, tpcds)

        meta: dict[str, Any] = {
            "query_id": row["query_id"],
            "tpcds_temp_and_q_idx": tpcds,
            "stage_latency_prediction_s": duration,  # best approximation
            "state": state,
            "color": color,
        }
        intervals_by_cluster.setdefault(cluster, []).append((s, e, meta))

        total_queries += 1
        if state != "RUNNING":
            effective_slo_s = resolver.resolve(tpcds)
            if duration > effective_slo_s:
                violating_queries += 1
                violation_amount += duration - effective_slo_s
            if effective_slo_s > 0:
                violation_relative_sum += max(
                    0.0, (duration - effective_slo_s) / effective_slo_s
                )

    # sort each cluster's intervals
    for cluster in intervals_by_cluster:
        intervals_by_cluster[cluster].sort(
            key=lambda x: (x[0], x[1], x[2].get("query_id", ""))
        )

    # compute costs from billing intervals
    cost_per_cluster: dict[str, float] = {}
    total_cost = 0.0
    for cluster, ivs in intervals_by_cluster.items():
        billed = Billing.billed_s([Interval(iv[0], iv[1], iv[2]) for iv in ivs])
        cost = cost_per_second_per_cluster.get(cluster, 0.0) * billed
        cost_per_cluster[cluster] = cost
        total_cost += cost

    violation_rate = (
        (violating_queries / total_queries) if total_queries > 0 else 0.0
    )
    violation_relative = (
        (violation_relative_sum / total_queries)
        if total_queries > 0
        else 0.0
    )
    return GanttSnapshot(
        label=label,
        intervals_by_cluster=intervals_by_cluster,
        total_queries=total_queries,
        violating_queries=violating_queries,
        violation_rate=violation_rate,
        cost_per_cluster=cost_per_cluster,
        total_cost=total_cost,
        violation_amount=violation_amount,
        violation_relative=violation_relative,
    )


def _cost_per_second_from_config(config: dict) -> dict[str, float]:
    """
    Reconstruct cost_per_second_per_cluster from the blueprint stored in config.
    Reads the blueprint YAML directly — no Blueprint or Cluster.from_config needed.
    """
    import autoslo.utils.paths as pu
    from autoslo.blueprints.cluster import Cluster

    bp_configs = pu.get_blueprint_dicts_from_config()
    bp_name = config["blueprint_name"]
    cluster_names = bp_configs[bp_name]["cluster_names"]
    return {
        name: Cluster.cost_per_second_for_rpu(
            Cluster.rpu_for_cluster_name(name)
        )
        for name in cluster_names
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def build_final_snapshot_from_log(
    log_path: str | Path,
    config: dict,
) -> GanttSnapshot:
    """
    Reconstruct the single final-state GanttSnapshot from the solve log.

    For each query:
    - cluster_name   : from the 'routing' event
    - start_s        : timestamp from the 'routing' event (= rel_start_time_s)
    - tpcds_temp_and_q_idx: from the 'routing' event
    - final end_time_s: from the 'completion' event if present; otherwise the
                        last end_time_s seen in 'latency_update' events, falling
                        back to the routing end_time_s
    - state          : COMPLETED if there's a completion event, RUNNING otherwise

    Coloring: green ≤ slo_s, red > slo_s (RUNNING intervals use gray).
    """
    log = pd.read_parquet(log_path)
    resolver = SloResolver.from_dict(
        default_slo_s=config["slo_s"],
        slo_dict=config.get("slo_dict") or {},
        slo_dict_filename=config.get("slo_dict_filename"),
    )
    cost_per_second = _cost_per_second_from_config(config)

    # --- index relevant events by query_id ---
    routing = log[log["event_type"] == "routing"].set_index("query_id")[
        ["timestamp", "cluster_name", "end_time_s", "tpcds_temp_and_q_idx"]
    ]
    completions = (
        log[log["event_type"] == "completion"]
        .set_index("query_id")[["end_time_s"]]
        .rename(columns={"end_time_s": "completed_end_time_s"})
    )
    # Last latency_update end_time per query (for still-running queries)
    updates = log[log["event_type"] == "latency_update"].copy()
    if not updates.empty:
        last_updates = (
            updates.sort_values("timestamp")
            .groupby("query_id")["end_time_s"]
            .last()
            .rename("updated_end_time_s")
        )
    else:
        last_updates = pd.Series(dtype=float, name="updated_end_time_s")

    # --- join everything ---
    df = routing.join(completions, how="left").join(last_updates, how="left")

    # Resolve final end_time_s: completed > last_update > routing
    def _resolve_end(row: "pd.Series") -> float:
        if pd.notna(row.get("completed_end_time_s")):
            return float(row["completed_end_time_s"])
        if pd.notna(row.get("updated_end_time_s")):
            return float(row["updated_end_time_s"])
        return float(row["end_time_s"])

    df["final_end_s"] = df.apply(_resolve_end, axis=1)
    df["state"] = df["completed_end_time_s"].apply(
        lambda v: "COMPLETED" if pd.notna(v) else "RUNNING"
    )

    rows: list[dict[str, Any]] = [
        {
            "query_id": qid,
            "tpcds_temp_and_q_idx": row["tpcds_temp_and_q_idx"],
            "cluster_name": row["cluster_name"],
            "start_s": float(row["timestamp"]),
            "end_s": row["final_end_s"],
            "state": row["state"],
        }
        for qid, row in df.iterrows()
    ]

    return _snapshot_from_query_dicts(
        rows, resolver, cost_per_second, label="Final"
    )


def build_scrubber_snapshots_from_log(
    log_path: str | Path,
    config: dict,
) -> list[GanttSnapshot]:
    """
    Walk the arrival events in order.  For each arrival, compute a GanttSnapshot
    capturing the state of the system at that moment:

    - Queries whose completion event has timestamp ≤ current arrival time:
      COMPLETED with their actual end_time_s.
    - Queries that have been routed (routing event ≤ current arrival time) but
      not yet completed: RUNNING with their most recent predicted end_time_s.

    Returns one GanttSnapshot per arrival event (no stride / skipping).

    NOTE: Not wired to the UI yet.  This function is implemented here so the
    backend API surface is complete; the frontend scrubber can be added later.
    """
    log = pd.read_parquet(log_path)
    resolver = SloResolver.from_dict(
        default_slo_s=config["slo_s"],
        slo_dict=config.get("slo_dict") or {},
        slo_dict_filename=config.get("slo_dict_filename"),
    )
    cost_per_second = _cost_per_second_from_config(config)

    # Build a per-query event timeline for fast lookup.
    # query_state[qid] = dict tracking the latest known values
    arrivals = log[log["event_type"] == "arrival"].sort_values("timestamp")
    routings = log[log["event_type"] == "routing"]
    updates = log[log["event_type"] == "latency_update"]
    completions = log[log["event_type"] == "completion"]

    # Index by query_id
    route_map: dict = routings.set_index("query_id").to_dict("index")
    complete_map: dict = completions.set_index("query_id")[
        ["timestamp", "end_time_s"]
    ].to_dict("index")

    # last update per query sorted chronologically
    if not updates.empty:
        upd_sorted = updates.sort_values("timestamp")
        last_upd_map: dict = (
            upd_sorted.groupby("query_id").last().to_dict("index")
        )
    else:
        last_upd_map = {}

    snapshots: list[GanttSnapshot] = []

    for i, (_, arrival_row) in enumerate(arrivals.iterrows()):
        current_time = float(arrival_row["timestamp"])
        label = f"Q{i+1}: {arrival_row['query_id']}"

        routed_qids = [
            qid
            for qid, rr in route_map.items()
            if float(rr["timestamp"]) <= current_time
        ]

        rows: list[dict[str, Any]] = []
        for qid in routed_qids:
            rr = route_map[qid]
            start_s = float(rr["timestamp"])
            tpcds = rr.get("tpcds_temp_and_q_idx", "")
            cluster_name = rr["cluster_name"]

            if (
                qid in complete_map
                and float(complete_map[qid]["timestamp"]) <= current_time
            ):
                end_s = float(complete_map[qid]["end_time_s"])
                state = "COMPLETED"
            else:
                if (
                    qid in last_upd_map
                    and float(last_upd_map[qid]["timestamp"]) <= current_time
                ):
                    end_s = float(last_upd_map[qid]["end_time_s"])
                else:
                    end_s = float(rr["end_time_s"])
                state = "RUNNING"

            rows.append(
                {
                    "query_id": qid,
                    "tpcds_temp_and_q_idx": tpcds,
                    "cluster_name": cluster_name,
                    "start_s": start_s,
                    "end_s": end_s,
                    "state": state,
                }
            )

        snap = _snapshot_from_query_dicts(
            rows, resolver, cost_per_second, label=label
        )
        snapshots.append(snap)

    return snapshots


def render_run(
    run_dir: str | Path,
    scrubber: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """
    Load config.yml + the structured/solve log from run_dir and render a Gantt figure.

    Parameters
    ----------
    run_dir : path to a simulator run directory (must contain config.yml and
              structured_log.parquet).
    scrubber : if True, call build_scrubber_snapshots_from_log (one step per
               arrival); if False (default), show only the final-state Gantt.
    **kwargs : forwarded to render_gantt_scrubber.
    """
    config, _ = _load_run(run_dir)
    log_path = os.path.join(run_dir, "structured_log.parquet")

    if scrubber:
        snapshots = build_scrubber_snapshots_from_log(log_path, config)
    else:
        snapshots = [build_final_snapshot_from_log(log_path, config)]

    return render_gantt_scrubber(
        snapshots,
        slo_s=config["slo_s"],
        workload_name=config.get("workload_name"),
        **kwargs,
    )
