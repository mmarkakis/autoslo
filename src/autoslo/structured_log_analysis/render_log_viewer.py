#!/usr/bin/env python3
"""
render_log_viewer.py
--------------------
Standalone script that reads a ``structured_log.parquet`` file and generates a
self-contained HTML page for interactively scrubbing through the run timeline.

Usage
-----
    python render_log_viewer.py /path/to/structured_log.parquet

The SLO configuration is read from ``config.yml`` or ``runner_config.yml`` in
the same directory as the log file.  The HTML file is written next to the input
log file.

Supports both runner and simulator logs.  Both log kinds share the same
event vocabulary defined in :class:`~autoslo.utils.structured_events.EventType`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

import pandas as pd

from autoslo.clusters.cluster import Cluster
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.structured_events import EventType


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------


def _detect_log_kind(df: pd.DataFrame) -> str:
    """Return ``'simulator'`` or ``'runner'`` based on ``source`` column."""
    sources = set(df["source"].unique())
    if "WorkloadRunner" in sources:
        return "runner"
    if "WorkloadSimulator" in sources:
        return "simulator"
    raise ValueError(
        f"Cannot determine log kind from sources: {sources}. "
        f"Expected 'WorkloadRunner' or 'WorkloadSimulator'."
    )


def _load_slo_resolver(log_dir: Path) -> SloResolver:
    """Read the config file next to the log and build a SloResolver."""
    for name in ("config.yml", "runner_config.yml"):
        cfg_path = log_dir / name
        if cfg_path.exists():
            break
    else:
        raise FileNotFoundError(
            f"No config.yml or runner_config.yml found in {log_dir}"
        )

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    return SloResolver.from_config(cfg)


def _validate_rel_time(df: pd.DataFrame) -> None:
    """Assert ``rel_time_s`` is present and contains relative timestamps."""
    if "rel_time_s" not in df.columns:
        raise ValueError(
            "Column 'rel_time_s' not found in log. "
            f"Available columns: {list(df.columns)}"
        )
    bad = df[df["rel_time_s"] > 1_000_000]
    if not bad.empty:
        counts = bad.groupby("event_type").size().to_dict()
        raise ValueError(
            "rel_time_s values appear to be absolute epoch timestamps, "
            f"not relative. Offending event types: {counts}"
        )


def _parse_details(raw: Any) -> dict:
    """Parse a details field that may be a JSON string or already a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _safe_rpu(cluster_name: str) -> int | None:
    """Extract RPU from cluster name, returning None on failure."""
    if not cluster_name:
        return None
    try:
        return Cluster.rpu_for_cluster_name(cluster_name)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Unified log parser
# ---------------------------------------------------------------------------


def _parse_log(
    df: pd.DataFrame,
    slo_resolver: SloResolver,
    log_kind: str,
) -> dict:
    """Parse a structured log DataFrame into the JS data payload."""

    events = df.sort_values("rel_time_s")

    # --- Event type value sets (strings) for filtering ---
    query_lifecycle_values = {e.value for e in EventType.query_lifecycle_types()}
    routing_values = {e.value for e in EventType.routing_types()}
    cluster_lifecycle_values = {e.value for e in EventType.cluster_lifecycle_types()}
    autoscaler_values = {e.value for e in EventType.autoscaler_types()}

    # --- Build per-query event timeline ---
    query_events: dict[str, list[dict]] = defaultdict(list)
    for _, row in events.iterrows():
        qid = row.get("query_id")
        if pd.isna(qid) or not qid:
            continue
        et = row["event_type"]
        if et not in query_lifecycle_values and et not in routing_values:
            continue
        query_events[qid].append({
            "rel_time_s": float(row["rel_time_s"]),
            "event_type": et,
            "cluster_name": row.get("cluster_name", ""),
            "query_text_id": str(row.get("query_text_id", "")),
            "details": _parse_details(row.get("details", "")),
        })

    # --- Reconstruct queries ---
    queries = []
    for qid, evts in query_events.items():
        evts.sort(key=lambda e: e["rel_time_s"])

        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in evts:
            by_type[e["event_type"]].append(e)

        # Arrival
        arrival_evts = by_type.get(EventType.ARRIVAL.value, [])
        arrival_s = arrival_evts[0]["rel_time_s"] if arrival_evts else None

        # Execution start (required)
        exec_start_evts = by_type.get(EventType.QUERY_EXECUTION_START.value, [])
        if not exec_start_evts:
            raise ValueError(
                f"Query {qid!r} is missing a QUERY_EXECUTION_START event. "
                "Check emission sites."
            )
        exec_start_s = exec_start_evts[0]["rel_time_s"]

        # Execution finish (required)
        exec_finish_evts = by_type.get(EventType.QUERY_EXECUTION_FINISH.value, [])
        if not exec_finish_evts:
            raise ValueError(
                f"Query {qid!r} is missing a QUERY_EXECUTION_FINISH event. "
                "Check emission sites."
            )
        exec_finish_s = exec_finish_evts[0]["rel_time_s"]

        # Completion (optional — run may have been interrupted)
        completion_evts = by_type.get(EventType.COMPLETION.value, [])
        completion_s: float | None = None
        success: bool | None = None
        if completion_evts:
            completion_s = completion_evts[0]["rel_time_s"]
            details = completion_evts[0]["details"]
            success = details.get("success")

        # Cluster name from QUERY_ROUTED or execution events
        routed_evts = by_type.get(EventType.QUERY_ROUTED.value, [])
        if routed_evts:
            cluster_name = routed_evts[0]["cluster_name"]
        else:
            cluster_name = exec_start_evts[0]["cluster_name"]

        # Query text id from any event
        query_text_id = ""
        for e in evts:
            qtid = e.get("query_text_id", "")
            if qtid and str(qtid) != "nan":
                query_text_id = str(qtid)
                break

        latency_s = exec_finish_s - exec_start_s
        slo_s = slo_resolver.resolve(query_text_id if query_text_id else None)
        rpu = _safe_rpu(cluster_name)

        # Use arrival_s if available, otherwise exec_start_s
        if arrival_s is None:
            arrival_s = exec_start_s

        # For overall bar extent
        end_s = completion_s if completion_s is not None else exec_finish_s

        completed = completion_s is not None
        violates_slo = (latency_s > slo_s) if (completed and success is not False) else False

        queries.append({
            "query_id": qid,
            "query_text_id": query_text_id,
            "cluster_name": cluster_name,
            "rpu": rpu,
            "arrival_s": arrival_s,
            "exec_start_s": exec_start_s,
            "exec_finish_s": exec_finish_s,
            "completion_s": completion_s,
            "start_s": arrival_s,
            "end_s": end_s,
            "latency_s": latency_s,
            "slo_s": slo_s,
            "success": success,
            "violates_slo": violates_slo,
            "state": "completed" if completed else "running",
        })

    # --- Cluster lifecycle events ---
    cluster_events_list = []
    cl_mask = events["event_type"].isin(cluster_lifecycle_values)
    for _, row in events[cl_mask].iterrows():
        cname = row.get("cluster_name", "")
        details = _parse_details(row.get("details", ""))
        cluster_events_list.append({
            "rel_time_s": float(row["rel_time_s"]),
            "event_type": row["event_type"],
            "cluster_name": cname,
            "rpu": _safe_rpu(cname),
            "reason": details.get("reason", ""),
        })

    # --- Autoscaler events ---
    autoscaler_events = []
    as_mask = events["event_type"].isin(autoscaler_values)
    for _, row in events[as_mask].iterrows():
        details = _parse_details(row.get("details", ""))
        cname = row.get("cluster_name", "")
        autoscaler_events.append({
            "rel_time_s": float(row["rel_time_s"]),
            "event_type": row["event_type"],
            "cluster_name": cname,
            "rpu": _safe_rpu(cname),
            "slo_violation": details.get("slo_violation"),
            "cost": details.get("cost"),
            "slo_threshold": details.get("slo_threshold"),
        })

    # --- Routing score events (grouped by query_id) ---
    routing_scores: dict[str, list[dict]] = defaultdict(list)
    rs_mask = events["event_type"] == EventType.ROUTING_SCORE.value
    for _, row in events[rs_mask].iterrows():
        qid = row.get("query_id", "")
        details = _parse_details(row.get("details", ""))
        routing_scores[qid].append({
            "rel_time_s": float(row["rel_time_s"]),
            "cluster_name": row.get("cluster_name", ""),
            "rpu": _safe_rpu(row.get("cluster_name", "")),
            "latency_s": details.get("latency_s"),
            "slo_violation": details.get("slo_violation"),
            "cost": details.get("cost"),
        })

    # --- Latency update events ---
    latency_update_events = []
    lu_mask = events["event_type"] == EventType.LATENCY_UPDATE.value
    for _, row in events[lu_mask].iterrows():
        details = _parse_details(row.get("details", ""))
        latency_update_events.append({
            "rel_time_s": float(row["rel_time_s"]),
            "query_id": row.get("query_id", ""),
            "cluster_name": row.get("cluster_name", ""),
            "old_latency_s": details.get("old_latency_s"),
            "latency_s": details.get("latency_s"),
        })

    # --- Run metadata ---
    run_meta: dict[str, Any] = {}
    rs_rows = events[events["event_type"] == EventType.RUN_START.value]
    if not rs_rows.empty:
        d = _parse_details(rs_rows.iloc[0].get("details", ""))
        run_meta = {
            "workload_name": d.get("workload_name", ""),
            "num_queries": d.get("num_queries"),
            "routing_policy": d.get("routing_policy", ""),
            "closed_loop": d.get("closed_loop"),
        }

    # --- Run finish time ---
    run_finish_events = []
    rf_rows = events[events["event_type"] == EventType.RUN_FINISH.value]
    for _, row in rf_rows.iterrows():
        run_finish_events.append({
            "rel_time_s": float(row["rel_time_s"]),
            "event_type": EventType.RUN_FINISH.value,
        })

    # --- Arrival times for scrubber ---
    arrivals = events[events["event_type"] == EventType.ARRIVAL.value].sort_values("rel_time_s")
    arrival_times = [float(t) for t in arrivals["rel_time_s"]]

    # --- Time range ---
    time_range = [float(events["rel_time_s"].min()), float(events["rel_time_s"].max())]

    return {
        "kind": log_kind,
        "queries": queries,
        "cluster_events": cluster_events_list,
        "autoscaler_events": autoscaler_events,
        "routing_scores": dict(routing_scores),
        "latency_update_events": latency_update_events,
        "run_meta": run_meta,
        "run_finish_events": run_finish_events,
        "arrival_times": arrival_times,
        "time_range": time_range,
        "default_slo_s": slo_resolver.default_slo_s,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Structured Log Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: #1a1a2e; color: #e0e0e0; overflow: hidden; }

/* Layout */
.container { display: flex; flex-direction: column; height: 100vh; }
.header { padding: 8px 16px; background: #16213e; border-bottom: 1px solid #0f3460; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
.header h1 { font-size: 14px; font-weight: 600; color: #e94560; }
.header .meta { font-size: 11px; color: #888; margin-left: 16px; }
.header .stats { font-size: 12px; color: #a0a0a0; }
.header .stats span { margin-left: 16px; }
.header .stats .violation { color: #e94560; }
.header .stats .met { color: #4ecca3; }

.main { display: flex; flex: 1; overflow: hidden; }

/* Gantt panel */
.gantt-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.gantt-controls { padding: 6px 16px; background: #16213e; border-bottom: 1px solid #0f3460; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
/* Scrubber row: sits between controls and viewport; spacer aligns thumb with canvas timeline */
.scrubber-row { display: flex; align-items: center; background: #16213e; border-bottom: 2px solid #0f3460; padding: 3px 0 3px 0; flex-shrink: 0; }
.scrubber-spacer { flex-shrink: 0; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px; }  /* width set by JS to match CLUSTER_LABEL_WIDTH */
.scrubber-tail { flex-shrink: 0; }  /* width set by JS to match viewport scrollbar gutter */
.scrubber-row input[type=range] { flex: 1; accent-color: #e94560; margin: 0; }
.scrubber-row .time-display { font-size: 12px; color: #e94560; font-weight: 600; min-width: 72px; text-align: right; }
.gantt-viewport { flex: 1; overflow: auto; position: relative; }
.gantt-canvas-wrap { position: relative; min-height: 100%; }
canvas#gantt { display: block; }

/* Event log panel */
.event-panel { width: 360px; border-left: 1px solid #0f3460; display: flex; flex-direction: column; background: #16213e; flex-shrink: 0; }
.event-panel h2 { font-size: 12px; padding: 8px 12px; border-bottom: 1px solid #0f3460; color: #a0a0a0; text-transform: uppercase; letter-spacing: 1px; }
.event-list { flex: 1; overflow-y: auto; font-size: 11px; }
.event-item { padding: 4px 12px; border-bottom: 1px solid #0f3460; }
.event-item.highlight { background: #0f3460; }
.event-item .event-time { color: #e94560; font-weight: 600; }
.event-item .event-type { color: #4ecca3; margin-left: 6px; }
.event-item .event-detail { color: #888; margin-left: 4px; }
.event-item .score-toggle { cursor: pointer; color: #4ecca3; margin-left: 4px; text-decoration: underline; }
.event-item .score-details { display: none; margin-top: 2px; padding-left: 12px; color: #888; }
.event-item .score-details.open { display: block; }

/* Tooltip */
.tooltip { position: fixed; background: #16213e; border: 1px solid #0f3460; padding: 8px 12px; font-size: 11px; pointer-events: none; z-index: 100; border-radius: 4px; max-width: 400px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: none; }
.tooltip .tt-row { margin: 2px 0; }
.tooltip .tt-label { color: #a0a0a0; }
.tooltip .tt-value { color: #e0e0e0; font-weight: 600; }
.tooltip .tt-violation { color: #e94560; }
.tooltip .tt-met { color: #4ecca3; }
.tooltip .tt-failed { color: #f0a500; }

/* Zoom controls */
.zoom-controls { display: flex; gap: 4px; }
.zoom-controls button { background: #0f3460; border: 1px solid #0f3460; color: #e0e0e0; padding: 2px 10px; cursor: pointer; font-size: 12px; border-radius: 3px; }
.zoom-controls button:hover { background: #e94560; }

/* Playback */
.playback-controls { display: flex; gap: 4px; align-items: center; }
.playback-controls button { background: #0f3460; border: 1px solid #0f3460; color: #e0e0e0; padding: 2px 8px; cursor: pointer; font-size: 12px; border-radius: 3px; }
.playback-controls button:hover { background: #e94560; }
.playback-controls button.active { background: #e94560; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
      <h1>Structured Log Viewer</h1>
      <span class="meta" id="run-meta"></span>
      <div class="stats">
          <span>Queries: <b id="stat-total">0</b></span>
          <span>Completed: <b id="stat-completed">0</b></span>
          <span class="violation">Violations: <b id="stat-violations">0</b></span>
          <span class="violation">Rate: <b id="stat-viol-rate">0%</b></span>
          <span>Clusters: <b id="stat-clusters">0</b></span>
      </div>
  </div>
  <div class="main">
    <div class="gantt-panel">
      <div class="gantt-controls">
        <div class="playback-controls">
          <button id="btn-play" title="Play/Pause">&#9654;</button>
          <button id="btn-reset" title="Reset">&#9632;</button>
        </div>
        <div class="zoom-controls">
          <button id="btn-zoom-in" title="Zoom in (=)">+</button>
          <button id="btn-zoom-out" title="Zoom out (-)">-</button>
          <button id="btn-zoom-fit" title="Zoom to fit (0)">Fit</button>
        </div>
      </div>
            <div class="scrubber-row" id="scrubber-row">
                <div class="scrubber-spacer" id="scrubber-spacer">
                    <div class="time-display" id="time-display">0.0s</div>
                </div>
                <input type="range" id="time-slider" min="0" max="1000" value="1000" step="1">
                <div class="scrubber-tail" id="scrubber-tail"></div>
            </div>
      <div class="gantt-viewport" id="gantt-viewport">
        <div class="gantt-canvas-wrap">
          <canvas id="gantt"></canvas>
        </div>
      </div>
    </div>
    <div class="event-panel">
      <h2>Events (<span id="event-count">0</span>)</h2>
      <div class="event-list" id="event-list"></div>
    </div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
// ===========================================================================
// DATA (injected by Python)
// ===========================================================================
const DATA = __DATA_PLACEHOLDER__;

// ===========================================================================
// HELPERS
// ===========================================================================

function escapeHtml(s) {
    if (s == null) return "";
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
                    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function formatMaybeNumber(v, digits = 2) {
    if (v == null || v === "") return "?";
    const n = Number(v);
    if (!Number.isFinite(n)) return escapeHtml(v);
    return n.toFixed(digits);
}

// ===========================================================================
// STATE
// ===========================================================================
const state = {
    currentTime: DATA.time_range[1],
    zoom: 1.0,
    panX: 0,
    playing: false,
    playTimer: null,
    playSpeed: 50,
    arrivalIdx: DATA.arrival_times.length,
    hoveredQuery: null,
};

// ===========================================================================
// CONSTANTS
// ===========================================================================
const COLORS = {
    met: "#4ecca3",
    metLight: "rgba(78,204,163,0.35)",
    violated: "#e94560",
    violatedLight: "rgba(233,69,96,0.35)",
    failed: "#f0a500",
    failedLight: "rgba(240,165,0,0.35)",
    running: "#555577",
    clusterBg: "#1e1e3a",
    clusterLine: "#0f3460",
    pending: "#ffcc00",
    pendingBg: "rgba(255,204,0,0.08)",
    text: "#a0a0a0",
    timeline: "#e94560",
};

const ROW_HEIGHT = 18;
const LANE_GAP = 2;
const CLUSTER_PADDING = 6;
// CLUSTER_LABEL_WIDTH is computed dynamically after realClusters is built.
const HEADER_HEIGHT = 56;  // two 28px sub-rows: top = minutes, bottom = seconds
const HEADER_MID = HEADER_HEIGHT / 2;  // y of divider between the two sub-rows
const MARKER_STRIP_HEIGHT = 10;  // reserved px above query lanes for lifecycle markers
// Minimum row height to ensure all label lines (name + RPU + queries + active) always fit.
const CLUSTER_LABEL_MIN_HEIGHT = CLUSTER_PADDING + MARKER_STRIP_HEIGHT + 42 + CLUSTER_PADDING; // 42px covers 3 label lines at 11px line-height

// Off-screen canvas for the diagonal hatch pattern (dead-zone fill)
const _hatchCanvas = document.createElement("canvas");
_hatchCanvas.width = 8;
_hatchCanvas.height = 8;
(function() {
    const hctx = _hatchCanvas.getContext("2d");
    hctx.clearRect(0, 0, 8, 8);
    hctx.strokeStyle = "rgba(255,255,255,0.06)";
    hctx.lineWidth = 1;
    // two diagonal stripes per tile (top-left to bottom-right)
    hctx.beginPath();
    hctx.moveTo(0, 0); hctx.lineTo(8, 8);
    hctx.moveTo(-4, 4); hctx.lineTo(4, -4);
    hctx.moveTo(4, 12); hctx.lineTo(12, 4);
    hctx.stroke();
}());
let _hatchPattern = null;  // lazily created per canvas context

// ===========================================================================
// DERIVED DATA
// ===========================================================================

// Build cluster lifecycle from cluster events and queries.
const clusterLifecycle = {};

DATA.queries.forEach(q => {
    const name = q.cluster_name;
    if (!name) return;
    if (!(name in clusterLifecycle)) {
        clusterLifecycle[name] = { firstSeen: q.start_s, lastSeen: q.end_s, rpu: q.rpu, spinUpStarted: null, readyTime: null };
    }
    clusterLifecycle[name].firstSeen = Math.min(clusterLifecycle[name].firstSeen, q.start_s);
    clusterLifecycle[name].lastSeen = Math.max(clusterLifecycle[name].lastSeen, q.end_s);
    if (q.rpu != null) clusterLifecycle[name].rpu = q.rpu;
});

DATA.cluster_events.forEach(e => {
    const name = e.cluster_name;
    if (!name) return;
    if (!(name in clusterLifecycle)) {
        clusterLifecycle[name] = { firstSeen: e.rel_time_s, lastSeen: e.rel_time_s, rpu: e.rpu, spinUpStarted: null, readyTime: null };
    }
    clusterLifecycle[name].firstSeen = Math.min(clusterLifecycle[name].firstSeen, e.rel_time_s);
    clusterLifecycle[name].lastSeen = Math.max(clusterLifecycle[name].lastSeen, e.rel_time_s);
    if (e.rpu != null) clusterLifecycle[name].rpu = e.rpu;
    if (e.event_type === "spin_up_started") {
        clusterLifecycle[name].spinUpStarted = e.rel_time_s;
    }
    if (e.event_type === "cluster_ready") {
        clusterLifecycle[name].readyTime = e.rel_time_s;
    }
    if (e.event_type === "cluster_removed") {
        // Keep earliest removal time in case of multiple events
        if (clusterLifecycle[name].removedTime == null ||
            e.rel_time_s < clusterLifecycle[name].removedTime) {
            clusterLifecycle[name].removedTime = e.rel_time_s;
        }
    }
});

// Filter out hypothetical clusters; sort by ready time
const realClusters = Object.keys(clusterLifecycle)
    .filter(n => !n.includes("hypothetical"))
    .sort((a, b) => {
        const readyA = clusterLifecycle[a].readyTime ?? clusterLifecycle[a].firstSeen;
        const readyB = clusterLifecycle[b].readyTime ?? clusterLifecycle[b].firstSeen;
        return readyA - readyB;
    });

const realQueries = DATA.queries.filter(q => !q.cluster_name.includes("hypothetical"));

// Pack queries into lanes per cluster
function packLanes(queries) {
    const sorted = [...queries].sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
    const lanes = [];
    sorted.forEach(q => {
        let placed = false;
        for (let i = 0; i < lanes.length; i++) {
            if (q.start_s >= lanes[i].endTime) {
                lanes[i].endTime = q.end_s;
                q._lane = i;
                placed = true;
                break;
            }
        }
        if (!placed) {
            q._lane = lanes.length;
            lanes.push({ endTime: q.end_s });
        }
    });
    return lanes.length;
}

const queriesByCluster = {};
const lanesPerCluster = {};
realClusters.forEach(name => { queriesByCluster[name] = []; });
realQueries.forEach(q => {
    if (q.cluster_name in queriesByCluster) {
        queriesByCluster[q.cluster_name].push(q);
    }
});
realClusters.forEach(name => {
    lanesPerCluster[name] = Math.max(1, packLanes(queriesByCluster[name]));
});

// Compute CLUSTER_LABEL_WIDTH dynamically so the widest cluster name always fits.
// We use an off-screen canvas to measure the actual rendered pixel width of the
// bold 11px monospace font used for the cluster name label.
(function() {
    const _mc = document.createElement("canvas").getContext("2d");
    _mc.font = "bold 11px monospace";
    let maxW = 100;
    realClusters.forEach(name => {
        const w = Math.ceil(_mc.measureText(name).width);
        if (w > maxW) maxW = w;
    });
    // 16px horizontal padding (8px each side)
    window.CLUSTER_LABEL_WIDTH = maxW + 16;
}());

// Align scrubber spacer width with the cluster label column so the
// slider thumb position corresponds to the time position on the canvas.
function syncScrubberLayout() {
    document.getElementById("scrubber-spacer").style.width = CLUSTER_LABEL_WIDTH + "px";
    const scrollbarWidth = Math.max(0, viewport.offsetWidth - viewport.clientWidth);
    document.getElementById("scrubber-tail").style.width = scrollbarWidth + "px";
}

// Compute cluster Y positions
// Each row: CLUSTER_PADDING + MARKER_STRIP_HEIGHT + max(lanes, min_label_lines) + CLUSTER_PADDING
const clusterYPositions = {};
let currentY = HEADER_HEIGHT;
realClusters.forEach(name => {
    const numLanes = lanesPerCluster[name];
    const lanesHeight = numLanes * (ROW_HEIGHT + LANE_GAP);
    const height = CLUSTER_PADDING + MARKER_STRIP_HEIGHT + Math.max(lanesHeight, CLUSTER_LABEL_MIN_HEIGHT - CLUSTER_PADDING - MARKER_STRIP_HEIGHT - CLUSTER_PADDING) + CLUSTER_PADDING;
    clusterYPositions[name] = { y: currentY, height };
    currentY += height + 1;
});
const totalHeight = currentY + 20;

// Build unified event list for the event panel
const allEvents = [];

// RUN_START
if (DATA.run_meta && DATA.run_meta.workload_name) {
    allEvents.push({ timestamp: DATA.time_range[0], type: "run_start", detail: escapeHtml(DATA.run_meta.workload_name) + (DATA.run_meta.routing_policy ? " / " + escapeHtml(DATA.run_meta.routing_policy) : "") });
}

DATA.cluster_events.forEach(e => {
    allEvents.push({ timestamp: e.rel_time_s, type: e.event_type, detail: escapeHtml(e.cluster_name) + (e.rpu != null ? " (" + e.rpu + " RPU)" : "") + (e.reason ? " \u2014 " + escapeHtml(e.reason) : "") });
});

DATA.autoscaler_events.forEach(e => {
    let detail = "";
    if (e.rpu != null) detail = e.rpu + " RPU";
    if (e.slo_violation != null) detail += " viol=" + e.slo_violation;
    if (e.cost != null) detail += " cost=" + e.cost;
    allEvents.push({ timestamp: e.rel_time_s, type: e.event_type, detail: escapeHtml(detail) });
});

(DATA.latency_update_events || []).forEach(e => {
    allEvents.push({ timestamp: e.rel_time_s, type: "latency_update", detail: escapeHtml(e.query_id) + " on " + escapeHtml(e.cluster_name) + " " + formatMaybeNumber(e.old_latency_s, 2) + "s\u2192" + formatMaybeNumber(e.latency_s, 2) + "s" });
});

DATA.arrival_times.forEach((t, i) => {
    allEvents.push({ timestamp: t, type: "arrival", detail: "query #" + (i + 1) });
});

realQueries.filter(q => q.state === "completed").forEach(q => {
    allEvents.push({ timestamp: q.end_s, type: "completion", detail: escapeHtml(q.query_id) + " on " + escapeHtml(q.cluster_name) + " (" + q.latency_s.toFixed(1) + "s)" });
});

realQueries.forEach(q => {
    const scores = DATA.routing_scores[q.query_id];
    allEvents.push({
        timestamp: q.exec_start_s,
        type: "query_routed",
        detail: escapeHtml(q.query_id) + " \u2192 " + escapeHtml(q.cluster_name),
        routingScores: scores || null,
    });
});

(DATA.run_finish_events || []).forEach(e => {
    allEvents.push({ timestamp: e.rel_time_s, type: "run_finish", detail: "" });
});

allEvents.sort((a, b) => a.timestamp - b.timestamp);

// Populate run metadata in header
if (DATA.run_meta && DATA.run_meta.workload_name) {
    const m = DATA.run_meta;
    let parts = [escapeHtml(m.workload_name)];
    if (m.routing_policy) parts.push(escapeHtml(m.routing_policy));
    if (m.closed_loop != null) parts.push(m.closed_loop ? "closed-loop" : "open-loop");
    if (m.num_queries != null) parts.push(m.num_queries + " queries");
    document.getElementById("run-meta").innerHTML = parts.join(" \u00b7 ");
}

// ===========================================================================
// CANVAS RENDERING
// ===========================================================================

const canvas = document.getElementById("gantt");
const ctx = canvas.getContext("2d");
const viewport = document.getElementById("gantt-viewport");

function timeToX(t) {
    return CLUSTER_LABEL_WIDTH + (t - DATA.time_range[0]) * state.zoom + state.panX;
}

function xToTime(x) {
    return (x - CLUSTER_LABEL_WIDTH - state.panX) / state.zoom + DATA.time_range[0];
}

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const viewWidth = viewport.clientWidth;
    const contentWidth = Math.max(viewWidth, CLUSTER_LABEL_WIDTH + (DATA.time_range[1] - DATA.time_range[0]) * state.zoom + 40);
    canvas.width = contentWidth * dpr;
    canvas.height = totalHeight * dpr;
    canvas.style.width = contentWidth + "px";
    canvas.style.height = totalHeight + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function getQueryColor(q) {
    if (q.success === false) return { full: COLORS.failed, light: COLORS.failedLight };
    if (q.violates_slo) return { full: COLORS.violated, light: COLORS.violatedLight };
    return { full: COLORS.met, light: COLORS.metLight };
}

function drawTimeline() {
    resizeCanvas();
    const W = parseFloat(canvas.style.width);
    const H = totalHeight;

    // Ensure hatch pattern is created for this canvas context
    if (!_hatchPattern) {
        _hatchPattern = ctx.createPattern(_hatchCanvas, "repeat");
    }

    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, W, H);

    // Time axis header — two sub-rows
    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, W, HEADER_HEIGHT);
    // Divider between top (minutes) and bottom (seconds) rows
    ctx.strokeStyle = "#1e2d50";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, HEADER_MID - 0.5);
    ctx.lineTo(W, HEADER_MID - 0.5);
    ctx.stroke();
    // Bottom border of header
    ctx.strokeStyle = COLORS.clusterLine;
    ctx.beginPath();
    ctx.moveTo(0, HEADER_HEIGHT - 0.5);
    ctx.lineTo(W, HEADER_HEIGHT - 0.5);
    ctx.stroke();

    // --- Interval selection ---
    const niceIntervals = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    // Major ticks: target ~80px spacing — shown in top row with minute labels
    let majorIdx = niceIntervals.findIndex(n => n >= 80 / state.zoom);
    if (majorIdx < 0) majorIdx = niceIntervals.length - 1;
    const majorInterval = niceIntervals[majorIdx];
    // Minor ticks: target ~20px spacing — shown in bottom row with second labels
    let minorIdx = niceIntervals.findIndex(n => n >= 20 / state.zoom);
    if (minorIdx < 0) minorIdx = niceIntervals.length - 1;
    const minorInterval = niceIntervals[Math.min(minorIdx, majorIdx)];  // never coarser than major

    ctx.textAlign = "center";

    // --- Major ticks (top row: minutes) ---
    const firstMajor = Math.ceil(DATA.time_range[0] / majorInterval) * majorInterval;
    for (let t = firstMajor; t <= DATA.time_range[1]; t += majorInterval) {
        const x = timeToX(t);
        if (x < CLUSTER_LABEL_WIDTH || x > W) continue;
        // Full-height grid line into cluster area
        ctx.strokeStyle = "#0f3460";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, HEADER_HEIGHT);
        ctx.lineTo(x, H);
        ctx.stroke();
        // Tick mark in top row
        ctx.strokeStyle = "#445";
        ctx.beginPath();
        ctx.moveTo(x, HEADER_MID - 1);
        ctx.lineTo(x, HEADER_MID - 6);
        ctx.stroke();
        // Label in top row
        const minLabel = majorInterval >= 60
            ? (t / 60).toFixed(0) + "m"
            : t.toFixed(0) + "s";
        ctx.fillStyle = "#c0c0c0";
        ctx.font = "bold 10px monospace";
        ctx.fillText(minLabel, x, HEADER_MID - 9);
    }

    // --- Minor ticks (bottom row: seconds) ---
    const firstMinor = Math.ceil(DATA.time_range[0] / minorInterval) * minorInterval;
    const showMinorLabels = minorInterval * state.zoom >= 25;  // only label if ticks are ≥25px apart
    for (let t = firstMinor; t <= DATA.time_range[1]; t += minorInterval) {
        const x = timeToX(t);
        if (x < CLUSTER_LABEL_WIDTH || x > W) continue;
        // Tick mark in bottom row
        ctx.strokeStyle = "#334";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, HEADER_MID + 1);
        ctx.lineTo(x, HEADER_MID + 5);
        ctx.stroke();
        // Label in bottom row
        if (showMinorLabels) {
            const secLabel = t.toFixed(0) + "s";
            ctx.fillStyle = COLORS.text;
            ctx.font = "10px monospace";
            ctx.fillText(secLabel, x, HEADER_HEIGHT - 5);
        }
    }

    // Current time line
    const curX = timeToX(state.currentTime);
    if (curX >= CLUSTER_LABEL_WIDTH) {
        ctx.strokeStyle = COLORS.timeline;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(curX, HEADER_HEIGHT);
        ctx.lineTo(curX, H);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
    }

    // Draw cluster rows
    realClusters.forEach(name => {
        const cl = clusterLifecycle[name];
        const pos = clusterYPositions[name];

        ctx.fillStyle = COLORS.clusterBg;
        ctx.fillRect(0, pos.y, W, pos.height);

        // Dead zone BEFORE spin_up_started (cluster does not yet exist)
        const deadStart = DATA.time_range[0];
        const aliveStart = cl.spinUpStarted ?? cl.readyTime ?? cl.firstSeen;
        if (aliveStart > deadStart) {
            const dx0 = Math.max(CLUSTER_LABEL_WIDTH, timeToX(deadStart));
            const dx1 = timeToX(aliveStart);
            if (dx1 > CLUSTER_LABEL_WIDTH) {
                ctx.fillStyle = "rgba(0,0,0,0.55)";
                ctx.fillRect(dx0, pos.y, dx1 - dx0, pos.height);
                if (_hatchPattern) {
                    ctx.fillStyle = _hatchPattern;
                    ctx.fillRect(dx0, pos.y, dx1 - dx0, pos.height);
                }
                // Right edge boundary line
                ctx.strokeStyle = "rgba(255,255,255,0.18)";
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(dx1, pos.y);
                ctx.lineTo(dx1, pos.y + pos.height);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }

        // Pending period: spin_up_started -> cluster_ready
        if (cl.readyTime != null && cl.spinUpStarted != null && cl.spinUpStarted < cl.readyTime) {
            const px0 = Math.max(CLUSTER_LABEL_WIDTH, timeToX(cl.spinUpStarted));
            const px1 = timeToX(cl.readyTime);
            if (px1 > CLUSTER_LABEL_WIDTH) {
                ctx.fillStyle = COLORS.pendingBg;
                ctx.fillRect(px0, pos.y, px1 - px0, pos.height);
                ctx.strokeStyle = COLORS.pending;
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 2]);
                ctx.beginPath();
                ctx.moveTo(px1, pos.y);
                ctx.lineTo(px1, pos.y + pos.height);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }

        // Dead zone AFTER cluster_removed (cluster no longer exists)
        if (cl.removedTime != null) {
            const rx0 = Math.max(CLUSTER_LABEL_WIDTH, timeToX(cl.removedTime));
            const rx1 = timeToX(DATA.time_range[1]);
            if (rx1 > CLUSTER_LABEL_WIDTH && rx0 < W) {
                ctx.fillStyle = "rgba(0,0,0,0.55)";
                ctx.fillRect(rx0, pos.y, rx1 - rx0, pos.height);
                if (_hatchPattern) {
                    ctx.fillStyle = _hatchPattern;
                    ctx.fillRect(rx0, pos.y, rx1 - rx0, pos.height);
                }
                // Left edge boundary line
                ctx.strokeStyle = "rgba(255,255,255,0.18)";
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(rx0, pos.y);
                ctx.lineTo(rx0, pos.y + pos.height);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }

        // Separator line
        ctx.strokeStyle = COLORS.clusterLine;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, pos.y + pos.height + 0.5);
        ctx.lineTo(W, pos.y + pos.height + 0.5);
        ctx.stroke();

        // Cluster label
        ctx.fillStyle = "#16213e";
        ctx.fillRect(0, pos.y, CLUSTER_LABEL_WIDTH, pos.height);
        ctx.strokeStyle = COLORS.clusterLine;
        ctx.beginPath();
        ctx.moveTo(CLUSTER_LABEL_WIDTH - 0.5, pos.y);
        ctx.lineTo(CLUSTER_LABEL_WIDTH - 0.5, pos.y + pos.height);
        ctx.stroke();

        ctx.save();
        ctx.rect(0, pos.y, CLUSTER_LABEL_WIDTH - 2, pos.height);
        ctx.clip();

        ctx.fillStyle = "#e0e0e0";
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "left";
        ctx.fillText(name, 6, pos.y + 13);

        ctx.fillStyle = COLORS.text;
        ctx.font = "10px monospace";
        const numQs = queriesByCluster[name].filter(q => q.start_s <= state.currentTime).length;
        const activeQs = queriesByCluster[name].filter(q => q.start_s <= state.currentTime && q.end_s > state.currentTime).length;
        if (cl.rpu != null) ctx.fillText(cl.rpu + " RPU", 6, pos.y + 25);
        ctx.fillText(numQs + " queries", 6, pos.y + 36);
        ctx.fillText(activeQs + " active", 6, pos.y + 47);

        ctx.restore();
    });

    // Draw query bars (multi-segment)
    realClusters.forEach(clusterName => {
        const pos = clusterYPositions[clusterName];
        const queries = queriesByCluster[clusterName];

        queries.forEach(q => {
            if (q.start_s > state.currentTime) return;

            const laneY = pos.y + CLUSTER_PADDING + MARKER_STRIP_HEIGHT + q._lane * (ROW_HEIGHT + LANE_GAP);
            const barHeight = ROW_HEIGHT;
            const colors = getQueryColor(q);
            const isCompleted = q.state === "completed" && q.end_s <= state.currentTime;

            // Segment 1: arrival_s -> exec_start_s (queue time)
            if (q.arrival_s < q.exec_start_s) {
                const x0 = timeToX(q.arrival_s);
                const segEnd = Math.min(q.exec_start_s, state.currentTime);
                const x1 = timeToX(segEnd);
                if (x1 > CLUSTER_LABEL_WIDTH && x0 < parseFloat(canvas.style.width)) {
                    ctx.fillStyle = isCompleted ? colors.light : COLORS.running;
                    ctx.fillRect(Math.max(CLUSTER_LABEL_WIDTH, x0), laneY, Math.max(1, x1 - Math.max(CLUSTER_LABEL_WIDTH, x0)), barHeight);
                }
            }

            // Segment 2: exec_start_s -> exec_finish_s (execution time)
            if (state.currentTime > q.exec_start_s) {
                const x0 = timeToX(q.exec_start_s);
                const segEnd = Math.min(q.exec_finish_s, state.currentTime);
                const x1 = timeToX(segEnd);
                if (x1 > CLUSTER_LABEL_WIDTH && x0 < parseFloat(canvas.style.width)) {
                    ctx.fillStyle = isCompleted ? colors.full : COLORS.running;
                    ctx.fillRect(Math.max(CLUSTER_LABEL_WIDTH, x0), laneY, Math.max(1, x1 - Math.max(CLUSTER_LABEL_WIDTH, x0)), barHeight);
                }
            }

            // Segment 3: exec_finish_s -> completion_s (post-exec)
            if (q.completion_s != null && q.completion_s > q.exec_finish_s && state.currentTime > q.exec_finish_s) {
                const x0 = timeToX(q.exec_finish_s);
                const segEnd = Math.min(q.completion_s, state.currentTime);
                const x1 = timeToX(segEnd);
                if (x1 > CLUSTER_LABEL_WIDTH && x0 < parseFloat(canvas.style.width)) {
                    ctx.fillStyle = isCompleted ? colors.light : COLORS.running;
                    ctx.fillRect(Math.max(CLUSTER_LABEL_WIDTH, x0), laneY, Math.max(1, x1 - Math.max(CLUSTER_LABEL_WIDTH, x0)), barHeight);
                }
            }
        });
    });

    // Draw cluster lifecycle markers
    DATA.cluster_events.forEach(e => {
        if (e.rel_time_s > state.currentTime) return;
        const x = timeToX(e.rel_time_s);
        if (x < CLUSTER_LABEL_WIDTH) return;
        const clName = e.cluster_name;
        if (!(clName in clusterYPositions)) return;
        const pos = clusterYPositions[clName];
        // Draw markers in the dedicated strip at top of the cluster row
        const my = pos.y + CLUSTER_PADDING + Math.floor(MARKER_STRIP_HEIGHT / 2);

        switch (e.event_type) {
            case "spin_up_decision":
                ctx.fillStyle = COLORS.met;
                ctx.beginPath(); ctx.moveTo(x, my-4); ctx.lineTo(x+4, my); ctx.lineTo(x, my+4); ctx.lineTo(x-4, my); ctx.fill();
                break;
            case "spin_up_requested":
                ctx.fillStyle = COLORS.met;
                ctx.beginPath(); ctx.moveTo(x, my-4); ctx.lineTo(x+4, my+2); ctx.lineTo(x-4, my+2); ctx.fill();
                break;
            case "spin_up_started":
                ctx.fillStyle = COLORS.met;
                ctx.beginPath(); ctx.moveTo(x-3, my-4); ctx.lineTo(x+3, my); ctx.lineTo(x-3, my+4); ctx.fill();
                break;
            case "cluster_ready":
                ctx.fillStyle = COLORS.met;
                ctx.beginPath(); ctx.arc(x, my, 3, 0, Math.PI*2); ctx.fill();
                break;
            case "tear_down_decision":
                ctx.fillStyle = COLORS.violated;
                ctx.beginPath(); ctx.moveTo(x, my-4); ctx.lineTo(x+4, my); ctx.lineTo(x, my+4); ctx.lineTo(x-4, my); ctx.fill();
                break;
            case "tear_down_requested":
                ctx.fillStyle = COLORS.violated;
                ctx.beginPath(); ctx.moveTo(x, my+4); ctx.lineTo(x+4, my-2); ctx.lineTo(x-4, my-2); ctx.fill();
                break;
            case "tear_down_blocked":
                ctx.strokeStyle = COLORS.violated; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.moveTo(x-3, my-3); ctx.lineTo(x+3, my+3); ctx.moveTo(x+3, my-3); ctx.lineTo(x-3, my+3); ctx.stroke();
                ctx.lineWidth = 1;
                break;
            case "tear_down_started":
                ctx.fillStyle = COLORS.violated;
                ctx.beginPath(); ctx.moveTo(x-3, my-4); ctx.lineTo(x+3, my); ctx.lineTo(x-3, my+4); ctx.fill();
                break;
            case "stats_collected":
                ctx.fillStyle = "#888";
                ctx.beginPath(); ctx.arc(x, my, 2.5, 0, Math.PI*2); ctx.fill();
                break;
            case "cluster_removed":
                ctx.fillStyle = COLORS.violated;
                ctx.beginPath(); ctx.arc(x, my, 3, 0, Math.PI*2); ctx.fill();
                break;
        }
    });

}

// ===========================================================================
// STATS
// ===========================================================================

function updateStats() {
    const visible = realQueries.filter(q => q.start_s <= state.currentTime);
    const completed = visible.filter(q => q.state === "completed" && q.end_s <= state.currentTime);
    const violations = completed.filter(q => q.violates_slo);
    const activeClusters = new Set(visible.map(q => q.cluster_name));

    document.getElementById("stat-total").textContent = visible.length;
    document.getElementById("stat-completed").textContent = completed.length;
    document.getElementById("stat-violations").textContent = violations.length;
    document.getElementById("stat-viol-rate").textContent = completed.length > 0
        ? (100 * violations.length / completed.length).toFixed(1) + "%"
        : "0%";
    document.getElementById("stat-clusters").textContent = activeClusters.size;
}

// ===========================================================================
// EVENT PANEL
// ===========================================================================

function updateEventPanel() {
    const container = document.getElementById("event-list");
    const visible = allEvents.filter(e => e.timestamp <= state.currentTime);
    const toShow = visible.slice(-200);

    document.getElementById("event-count").textContent = visible.length;

    let html = "";
    toShow.forEach((e, idx) => {
        const tStr = e.timestamp.toFixed(1);
        html += '<div class="event-item"><span class="event-time">' + escapeHtml(tStr) + 's</span><span class="event-type">' + escapeHtml(e.type) + '</span><span class="event-detail">' + (e.detail || "") + '</span>';
        if (e.routingScores && e.routingScores.length > 0) {
            const detailId = "score-detail-" + idx;
            html += ' <span class="score-toggle" onclick="document.getElementById(\'' + detailId + '\').classList.toggle(\'open\')">[scores]</span>';
            html += '<div class="score-details" id="' + detailId + '">';
            e.routingScores.forEach(s => {
                html += '<div>' + escapeHtml(s.cluster_name) + (s.rpu != null ? ' (' + s.rpu + ' RPU)' : '') + ' lat=' + formatMaybeNumber(s.latency_s, 2) + 's cost=' + (s.cost != null ? s.cost : '?') + '</div>';
            });
            html += '</div>';
        }
        html += '</div>';
    });
    container.innerHTML = html;
    container.scrollTop = container.scrollHeight;
}

// ===========================================================================
// TOOLTIP
// ===========================================================================

const tooltipEl = document.getElementById("tooltip");

function showTooltip(x, y, q) {
    const sloLabel = q.success === false ? "tt-failed" : (q.violates_slo ? "tt-violation" : "tt-met");
    const sloText = q.success === false ? "FAILED" : (q.violates_slo ? "VIOLATED" : "MET");
    tooltipEl.innerHTML =
        '<div class="tt-row"><span class="tt-label">Query:</span> <span class="tt-value">' + escapeHtml(q.query_id) + '</span></div>' +
        '<div class="tt-row"><span class="tt-label">Template:</span> <span class="tt-value">' + escapeHtml(q.query_text_id) + '</span></div>' +
        '<div class="tt-row"><span class="tt-label">Cluster:</span> <span class="tt-value">' + escapeHtml(q.cluster_name) + (q.rpu != null ? " (" + q.rpu + " RPU)" : "") + '</span></div>' +
        '<div class="tt-row"><span class="tt-label">Arrival:</span> <span class="tt-value">' + q.arrival_s.toFixed(1) + 's</span></div>' +
        '<div class="tt-row"><span class="tt-label">Exec start:</span> <span class="tt-value">' + q.exec_start_s.toFixed(1) + 's</span></div>' +
        '<div class="tt-row"><span class="tt-label">Exec finish:</span> <span class="tt-value">' + q.exec_finish_s.toFixed(1) + 's</span></div>' +
        '<div class="tt-row"><span class="tt-label">Latency:</span> <span class="tt-value">' + q.latency_s.toFixed(2) + 's</span></div>' +
        '<div class="tt-row"><span class="tt-label">SLO:</span> <span class="tt-value">' + q.slo_s.toFixed(1) + 's</span> <span class="' + sloLabel + '">(' + sloText + ')</span></div>' +
        (q.completion_s != null ? '<div class="tt-row"><span class="tt-label">Completion:</span> <span class="tt-value">' + q.completion_s.toFixed(1) + 's</span></div>' : '') +
        '<div class="tt-row"><span class="tt-label">State:</span> <span class="tt-value">' + escapeHtml(q.state) + '</span></div>';
    tooltipEl.style.display = "block";
    tooltipEl.style.left = Math.min(x + 10, window.innerWidth - 420) + "px";
    tooltipEl.style.top = Math.min(y + 10, window.innerHeight - 200) + "px";
}

function hideTooltip() {
    tooltipEl.style.display = "none";
    state.hoveredQuery = null;
}

// ===========================================================================
// HIT TESTING
// ===========================================================================

function hitTest(canvasX, canvasY) {
    for (const clusterName of realClusters) {
        const pos = clusterYPositions[clusterName];
        if (canvasY < pos.y || canvasY > pos.y + pos.height) continue;

        for (const q of queriesByCluster[clusterName]) {
            if (q.start_s > state.currentTime) continue;
            const x0 = timeToX(q.start_s);
            const effectiveEnd = Math.min(q.end_s, state.currentTime);
            const x1 = timeToX(effectiveEnd);
            const laneY = pos.y + CLUSTER_PADDING + MARKER_STRIP_HEIGHT + q._lane * (ROW_HEIGHT + LANE_GAP);

            if (canvasX >= x0 && canvasX <= x1 && canvasY >= laneY && canvasY <= laneY + ROW_HEIGHT) {
                return q;
            }
        }
    }
    return null;
}

// ===========================================================================
// INTERACTIONS
// ===========================================================================

const slider = document.getElementById("time-slider");
const timeDisplay = document.getElementById("time-display");

function setTime(t) {
    state.currentTime = Math.max(DATA.time_range[0], Math.min(DATA.time_range[1], t));
    slider.value = Math.round(1000 * (state.currentTime - DATA.time_range[0]) / (DATA.time_range[1] - DATA.time_range[0]));
    timeDisplay.textContent = state.currentTime.toFixed(1) + "s";
    drawTimeline();
    updateStats();
    updateEventPanel();
}

slider.addEventListener("input", () => {
    const frac = parseInt(slider.value) / 1000;
    setTime(DATA.time_range[0] + frac * (DATA.time_range[1] - DATA.time_range[0]));
});

// Zoom helpers
function zoomBy(factor, centerX) {
    if (centerX == null) centerX = viewport.clientWidth / 2;
    const timeBefore = xToTime(centerX);
    state.zoom = Math.max(0.01, state.zoom * factor);
    const timeAfter = xToTime(centerX);
    state.panX += (timeAfter - timeBefore) * state.zoom;
    drawTimeline();
}

document.getElementById("btn-zoom-in").addEventListener("click", () => { zoomBy(1.5); });
document.getElementById("btn-zoom-out").addEventListener("click", () => { zoomBy(1 / 1.5); });
document.getElementById("btn-zoom-fit").addEventListener("click", () => {
    const viewW = viewport.clientWidth - CLUSTER_LABEL_WIDTH - 40;
    const timeSpan = DATA.time_range[1] - DATA.time_range[0];
    state.zoom = timeSpan > 0 ? viewW / timeSpan : 1;
    state.panX = 0;
    drawTimeline();
});

// Ctrl+scroll / Shift+scroll zoom
viewport.addEventListener("wheel", (e) => {
    if (e.ctrlKey || e.shiftKey) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        zoomBy(factor, cx);
    }
}, { passive: false });

// Playback
document.getElementById("btn-play").addEventListener("click", () => {
    if (state.playing) {
        clearInterval(state.playTimer);
        state.playing = false;
        document.getElementById("btn-play").classList.remove("active");
    } else {
        if (state.arrivalIdx >= DATA.arrival_times.length) {
            state.arrivalIdx = 0;
        }
        state.playing = true;
        document.getElementById("btn-play").classList.add("active");
        state.playTimer = setInterval(() => {
            if (state.arrivalIdx < DATA.arrival_times.length) {
                setTime(DATA.arrival_times[state.arrivalIdx]);
                state.arrivalIdx++;
            } else {
                setTime(DATA.time_range[1]);
                clearInterval(state.playTimer);
                state.playing = false;
                document.getElementById("btn-play").classList.remove("active");
            }
        }, 1000 / state.playSpeed);
    }
});

document.getElementById("btn-reset").addEventListener("click", () => {
    if (state.playing) {
        clearInterval(state.playTimer);
        state.playing = false;
        document.getElementById("btn-play").classList.remove("active");
    }
    state.arrivalIdx = 0;
    setTime(DATA.time_range[0]);
});

// Mouse hover for tooltip
canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const hit = hitTest(cx, cy);
    if (hit) {
        state.hoveredQuery = hit;
        showTooltip(e.clientX, e.clientY, hit);
        canvas.style.cursor = "pointer";
        drawTimeline();
    } else {
        if (state.hoveredQuery) {
            state.hoveredQuery = null;
            drawTimeline();
        }
        hideTooltip();
        canvas.style.cursor = "default";
    }
});
canvas.addEventListener("mouseleave", () => {
    hideTooltip();
    if (state.hoveredQuery) {
        state.hoveredQuery = null;
        drawTimeline();
    }
});

// Keyboard
document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") {
        const prevIdx = DATA.arrival_times.findIndex(t => t >= state.currentTime) - 1;
        if (prevIdx >= 0) setTime(DATA.arrival_times[prevIdx]);
        else if (DATA.arrival_times.length > 0) setTime(DATA.arrival_times[0]);
    } else if (e.key === "ArrowRight") {
        const nextIdx = DATA.arrival_times.findIndex(t => t > state.currentTime);
        if (nextIdx >= 0) setTime(DATA.arrival_times[nextIdx]);
        else setTime(DATA.time_range[1]);
    } else if (e.key === " ") {
        e.preventDefault();
        document.getElementById("btn-play").click();
    } else if (e.key === "=" || e.key === "+") {
        zoomBy(1.5);
    } else if (e.key === "-") {
        zoomBy(1 / 1.5);
    } else if (e.key === "0") {
        document.getElementById("btn-zoom-fit").click();
    }
});

// ===========================================================================
// INIT
// ===========================================================================

window.addEventListener("resize", () => {
    syncScrubberLayout();
    drawTimeline();
});

{
    const viewW = viewport.clientWidth - CLUSTER_LABEL_WIDTH - 40;
    const timeSpan = DATA.time_range[1] - DATA.time_range[0];
    state.zoom = timeSpan > 0 ? viewW / timeSpan : 1;
}
setTime(DATA.time_range[1]);
syncScrubberLayout();  // measure scrollbar after canvas has been sized

</script>
</body>
</html>"""


def generate_html(data: dict) -> str:
    """Inject timeline data into the HTML template."""
    data_json = json.dumps(data, default=str).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a structured log as an interactive HTML timeline viewer.",
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to structured_log.parquet",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output HTML path (default: same directory as log file)",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path).resolve()
    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load SLO config from the same directory
    slo_resolver = _load_slo_resolver(log_path.parent)
    print(f"SLO config: default={slo_resolver.default_slo_s}s"
          f"{', per-template overrides loaded' if slo_resolver.has_overrides() else ''}")

    print(f"Reading {log_path} ...")
    df = pd.read_parquet(log_path)
    print(f"  {len(df)} events, {df['event_type'].nunique()} event types")

    _validate_rel_time(df)

    kind = _detect_log_kind(df)
    print(f"  Detected log kind: {kind}")

    data = _parse_log(df, slo_resolver, kind)

    print(f"  {len(data['queries'])} queries across {len(set(q['cluster_name'] for q in data['queries']))} clusters")

    html_content = generate_html(data)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = log_path.parent / "log_viewer.html"

    out_path.write_text(html_content)
    print(f"  Written to {out_path}")
    print(f"  Open in browser: file://{out_path}")


if __name__ == "__main__":
    main()
