#!/usr/bin/env python3
"""
render_log_viewer.py
--------------------
Standalone script that reads a structured_log.parquet file and generates a
self-contained HTML page for interactively scrubbing through the run timeline.

Usage
-----
    python render_log_viewer.py /path/to/structured_log.parquet

The SLO configuration is read from config.yml or runner_config.yml in the
same directory as the log file.  The HTML file is written next to the input
log file.

Supports both simulator logs (event_types: arrival, routing, latency_update,
completion, spin_up_scheduled, cluster_ready, tear_down_*, ...) and runner
logs (event_types: query_routed, query_execution_start, query_execution_finish,
run_start, run_finish, ...).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import yaml

import pandas as pd

from autoslo.slo.slo_resolver import SloResolver


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _detect_log_kind(df: pd.DataFrame) -> str:
    """Return 'simulator' or 'runner' based on event_types present."""
    event_types = set(df["event_type"].unique())
    if "arrival" in event_types or "completion" in event_types:
        return "simulator"
    if "query_routed" in event_types or "query_execution_finish" in event_types:
        return "runner"
    # fallback: look at sources
    sources = set(df["source"].unique())
    if "WorkloadSimulator" in sources:
        return "simulator"
    return "runner"


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

    slo_cfg = cfg.get("slo_config", {})
    slo_s = float(slo_cfg.get("slo_s", 10.0))
    slo_dict_filename = slo_cfg.get("slo_dict_filename")

    return SloResolver(default_slo_s=slo_s, slo_dict_filename=slo_dict_filename)


def _extract_rpu_from_cluster_name(name: str) -> int:
    """Extract RPU from cluster names like 'autoslo-16-...'."""
    parts = name.split("-")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def _parse_simulator_log(df: pd.DataFrame, slo_resolver: SloResolver) -> dict:
    """Parse simulator log into structured timeline data."""
    events = df.sort_values("timestamp")

    # --- Query intervals ---
    routings = events[events["event_type"].isin(["routing", "query_routed"])]
    completions = events[events["event_type"] == "completion"]
    latency_updates = events[events["event_type"] == "latency_update"]

    # Build routing index
    route_map: dict[str, dict] = {}
    for _, row in routings.iterrows():
        qid = row["query_id"]
        route_map[qid] = {
            "start_s": float(row["timestamp"]),
            "cluster_name": row["cluster_name"],
            "query_text_id": row.get("query_text_id", ""),
            "end_time_s": float(row["end_time_s"]) if pd.notna(row.get("end_time_s")) else None,
        }

    # Build completion index
    complete_map: dict[str, float] = {}
    for _, row in completions.iterrows():
        qid = row["query_id"]
        if pd.notna(row.get("latency_s")):
            complete_map[qid] = float(row["latency_s"])
        elif pd.notna(row.get("end_time_s")):
            complete_map[qid] = float(row["end_time_s"]) - route_map.get(qid, {}).get("start_s", 0.0)

    # Build last latency update index
    update_map: dict[str, float] = {}
    if not latency_updates.empty:
        for _, row in latency_updates.sort_values("timestamp").iterrows():
            qid = row["query_id"]
            if pd.notna(row.get("latency_s")):
                update_map[qid] = float(row["latency_s"])
            elif pd.notna(row.get("end_time_s")) and qid in route_map:
                update_map[qid] = float(row["end_time_s"]) - route_map[qid]["start_s"]

    # Build query list
    queries = []
    for qid, rr in route_map.items():
        if qid in complete_map:
            latency_s = complete_map[qid]
            state = "completed"
        elif qid in update_map:
            latency_s = update_map[qid]
            state = "running"
        elif rr["end_time_s"] is not None:
            latency_s = rr["end_time_s"] - rr["start_s"]
            state = "running"
        else:
            continue

        query_text_id = rr.get("query_text_id", "")
        if pd.isna(query_text_id):
            query_text_id = ""

        query_slo_s = slo_resolver.resolve(query_text_id if query_text_id else None)
        queries.append({
            "query_id": qid,
            "query_text_id": str(query_text_id),
            "cluster_name": rr["cluster_name"],
            "start_s": rr["start_s"],
            "latency_s": latency_s,
            "end_s": rr["start_s"] + latency_s,
            "state": state,
            "slo_s": query_slo_s,
            "violates_slo": (latency_s > query_slo_s) if state == "completed" else False,
        })

    # --- Cluster lifecycle events ---
    cluster_events = []
    lifecycle_types = {
        "spin_up_scheduled", "request_spin_up", "spin_up",
        "cluster_ready",
        "tear_down_decision", "tear_down_requested",
        "cluster_removed", "stats_collected",
        "capacity_checkpoint_reconciliation",
    }
    for _, row in events[events["event_type"].isin(lifecycle_types)].iterrows():
        evt: dict = {
            "timestamp": float(row["timestamp"]),
            "event_type": row["event_type"],
            "cluster_name": row.get("cluster_name", ""),
        }
        if pd.notna(row.get("rpu")):
            evt["rpu"] = int(row["rpu"])
        if pd.notna(row.get("reason")):
            evt["reason"] = str(row["reason"])
        cluster_events.append(evt)

    # --- Autoscaler events ---
    autoscaler_events = []
    as_types = {"rpu_counterfactual", "rpu_selection"}
    for _, row in events[events["event_type"].isin(as_types)].iterrows():
        evt = {
            "timestamp": float(row["timestamp"]),
            "event_type": row["event_type"],
        }
        if pd.notna(row.get("candidate_rpu")):
            evt["candidate_rpu"] = int(row["candidate_rpu"])
        if pd.notna(row.get("selected_rpu")):
            evt["selected_rpu"] = int(row["selected_rpu"])
        if pd.notna(row.get("metric_and_cost")):
            evt["metric_and_cost"] = str(row["metric_and_cost"])
        autoscaler_events.append(evt)

    # --- Arrival timestamps for scrubber ---
    arrivals = events[events["event_type"] == "arrival"].sort_values("timestamp")
    arrival_times = [float(t) for t in arrivals["timestamp"]]

    return {
        "kind": "simulator",
        "queries": queries,
        "cluster_events": cluster_events,
        "autoscaler_events": autoscaler_events,
        "arrival_times": arrival_times,
        "time_range": [float(events["timestamp"].min()), float(events["timestamp"].max())],
        "default_slo_s": slo_resolver.default_slo_s,
    }


def _parse_runner_log(df: pd.DataFrame, slo_resolver: SloResolver) -> dict:
    """Parse runner log into structured timeline data."""
    events = df.sort_values("timestamp")

    # Runner uses absolute timestamps — normalize to relative
    t0 = float(events["timestamp"].min())

    # --- Query intervals ---
    routed = events[events["event_type"] == "query_routed"]
    finished = events[events["event_type"] == "query_execution_finish"]

    finish_map: dict[str, dict] = {}
    for _, row in finished.iterrows():
        qid = row["query_id"]
        finish_map[qid] = {
            "end_ts": float(row["timestamp"]),
            "latency_s": float(row["latency_s"]) if pd.notna(row.get("latency_s")) else None,
        }

    queries = []
    for _, row in routed.iterrows():
        qid = row["query_id"]
        start_s = float(row["timestamp"]) - t0

        query_text_id = row.get("query_text_id", "")
        if pd.isna(query_text_id):
            query_text_id = ""

        if qid in finish_map:
            fm = finish_map[qid]
            latency_s = fm["latency_s"] if fm["latency_s"] is not None else (fm["end_ts"] - float(row["timestamp"]))
            state = "completed"
        else:
            latency_s = 0.0
            state = "running"

        query_slo_s = slo_resolver.resolve(query_text_id if query_text_id else None)
        queries.append({
            "query_id": qid,
            "query_text_id": str(query_text_id),
            "cluster_name": row["cluster_name"],
            "start_s": start_s,
            "latency_s": latency_s,
            "end_s": start_s + latency_s,
            "state": state,
            "slo_s": query_slo_s,
            "violates_slo": (latency_s > query_slo_s) if state == "completed" else False,
        })

    # Cluster lifecycle
    cluster_events = []
    for _, row in events[events["event_type"].isin([
        "cluster_tear_down_started", "cluster_tear_down_completed",
    ])].iterrows():
        cluster_events.append({
            "timestamp": float(row["timestamp"]) - t0,
            "event_type": row["event_type"],
            "cluster_name": row.get("cluster_name", ""),
        })

    # Arrival times — use query_routed timestamps
    arrival_times = sorted(float(row["timestamp"]) - t0 for _, row in routed.iterrows())

    return {
        "kind": "runner",
        "queries": queries,
        "cluster_events": cluster_events,
        "autoscaler_events": [],
        "arrival_times": arrival_times,
        "time_range": [0.0, float(events["timestamp"].max()) - t0],
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
.header .stats { font-size: 12px; color: #a0a0a0; }
.header .stats span { margin-left: 16px; }
.header .stats .violation { color: #e94560; }
.header .stats .met { color: #4ecca3; }

.main { display: flex; flex: 1; overflow: hidden; }

/* Gantt panel */
.gantt-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.gantt-controls { padding: 8px 16px; background: #16213e; border-bottom: 1px solid #0f3460; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
.gantt-controls label { font-size: 12px; color: #a0a0a0; }
.gantt-controls input[type=range] { flex: 1; accent-color: #e94560; }
.gantt-controls .time-display { font-size: 12px; color: #e94560; font-weight: 600; min-width: 80px; text-align: right; }
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

/* Tooltip */
.tooltip { position: fixed; background: #16213e; border: 1px solid #0f3460; padding: 8px 12px; font-size: 11px; pointer-events: none; z-index: 100; border-radius: 4px; max-width: 400px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); display: none; }
.tooltip .tt-row { margin: 2px 0; }
.tooltip .tt-label { color: #a0a0a0; }
.tooltip .tt-value { color: #e0e0e0; font-weight: 600; }
.tooltip .tt-violation { color: #e94560; }
.tooltip .tt-met { color: #4ecca3; }

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
        <label>Time:</label>
        <input type="range" id="time-slider" min="0" max="1000" value="1000" step="1">
        <div class="time-display" id="time-display">0.0s</div>
        <div class="zoom-controls">
          <button id="btn-zoom-in">+</button>
          <button id="btn-zoom-out">-</button>
          <button id="btn-zoom-fit">Fit</button>
        </div>
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
// STATE
// ===========================================================================
const state = {
    currentTime: DATA.time_range[1],
    zoom: 1.0,
    panX: 0,
    playing: false,
    playTimer: null,
    playSpeed: 50,  // queries per second
    arrivalIdx: DATA.arrival_times.length,
};

// ===========================================================================
// CONSTANTS
// ===========================================================================
const COLORS = {
    met: "#4ecca3",
    violated: "#e94560",
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
const CLUSTER_LABEL_WIDTH = 140;
const HEADER_HEIGHT = 30;

// ===========================================================================
// DERIVED DATA
// ===========================================================================

// Collect all cluster names and sort by RPU ascending, then name
function extractRpu(name) {
    if (!name) return 0;
    const m = name.match(/cluster_(\d+)_/);
    if (m) return parseInt(m[1]);
    const m2 = name.match(/(\d+)rpu/);
    if (m2) return parseInt(m2[1]);
    return 0;
}

// Build cluster lifecycle: for each cluster, when it appeared and disappeared.
const clusterLifecycle = {};  // name -> { firstSeen, lastSeen, rpu, pending_until }

// From queries
DATA.queries.forEach(q => {
    const name = q.cluster_name;
    if (!name) return;
    if (!(name in clusterLifecycle)) {
        clusterLifecycle[name] = { firstSeen: q.start_s, lastSeen: q.end_s, rpu: extractRpu(name), pending_until: null };
    }
    clusterLifecycle[name].firstSeen = Math.min(clusterLifecycle[name].firstSeen, q.start_s);
    clusterLifecycle[name].lastSeen = Math.max(clusterLifecycle[name].lastSeen, q.end_s);
});

// From cluster events
DATA.cluster_events.forEach(e => {
    const name = e.cluster_name;
    if (!name) return;
    if (!(name in clusterLifecycle)) {
        clusterLifecycle[name] = { firstSeen: e.timestamp, lastSeen: e.timestamp, rpu: e.rpu || extractRpu(name), pending_until: null };
    }
    clusterLifecycle[name].firstSeen = Math.min(clusterLifecycle[name].firstSeen, e.timestamp);
    clusterLifecycle[name].lastSeen = Math.max(clusterLifecycle[name].lastSeen, e.timestamp);
    if (e.event_type === "spin_up_scheduled" || e.event_type === "request_spin_up" || e.event_type === "spin_up") {
        clusterLifecycle[name].pending_until = null;  // will be set by cluster_ready
    }
    if (e.event_type === "cluster_ready") {
        clusterLifecycle[name].pending_until = e.timestamp;
    }
});

// Filter out hypothetical clusters from the Gantt view
const realClusters = Object.keys(clusterLifecycle)
    .filter(n => !n.includes("hypothetical"))
    .sort((a, b) => {
        const rpuDiff = clusterLifecycle[a].rpu - clusterLifecycle[b].rpu;
        if (rpuDiff !== 0) return rpuDiff;
        return clusterLifecycle[a].firstSeen - clusterLifecycle[b].firstSeen;
    });

// Filter out hypothetical queries
const realQueries = DATA.queries.filter(q => !q.cluster_name.includes("hypothetical"));

// Pack queries into lanes per cluster
function packLanes(queries) {
    // Sort by start time
    const sorted = [...queries].sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
    const lanes = [];  // each lane has a "endTime" for greedy packing

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

// Group queries by cluster and pack lanes
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

// Compute cluster Y positions
const clusterYPositions = {};  // name -> { y, height }
let currentY = HEADER_HEIGHT;
realClusters.forEach(name => {
    const numLanes = lanesPerCluster[name];
    const height = numLanes * (ROW_HEIGHT + LANE_GAP) + CLUSTER_PADDING * 2;
    clusterYPositions[name] = { y: currentY, height };
    currentY += height + 1;  // 1px separator
});
const totalHeight = currentY + 20;

// Build unified event list for the event panel
const allEvents = [];
DATA.cluster_events.forEach(e => {
    allEvents.push({ timestamp: e.timestamp, type: e.event_type, detail: e.cluster_name + (e.rpu ? ` (${e.rpu} RPU)` : "") + (e.reason ? ` — ${e.reason}` : "") });
});
DATA.autoscaler_events.forEach(e => {
    let detail = "";
    if (e.selected_rpu) detail = `selected ${e.selected_rpu} RPU`;
    else if (e.candidate_rpu) detail = `candidate ${e.candidate_rpu} RPU`;
    if (e.metric_and_cost) detail += ` [${e.metric_and_cost}]`;
    allEvents.push({ timestamp: e.timestamp, type: e.event_type, detail });
});
// Add arrival events (sparse — only label every Nth)
DATA.arrival_times.forEach((t, i) => {
    allEvents.push({ timestamp: t, type: "arrival", detail: `query #${i + 1}` });
});
// Add completion events from queries
realQueries.filter(q => q.state === "completed").forEach(q => {
    allEvents.push({ timestamp: q.end_s, type: "completion", detail: `${q.query_id} on ${q.cluster_name} (${q.latency_s.toFixed(1)}s)` });
});
allEvents.sort((a, b) => a.timestamp - b.timestamp);

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

function drawTimeline() {
    resizeCanvas();
    const W = parseFloat(canvas.style.width);
    const H = totalHeight;

    // Clear
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, W, H);

    // Time axis header
    ctx.fillStyle = "#16213e";
    ctx.fillRect(0, 0, W, HEADER_HEIGHT);
    ctx.strokeStyle = COLORS.clusterLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, HEADER_HEIGHT - 0.5);
    ctx.lineTo(W, HEADER_HEIGHT - 0.5);
    ctx.stroke();

    // Time ticks
    const timeSpan = DATA.time_range[1] - DATA.time_range[0];
    const pixelsPerSecond = state.zoom;
    // Choose tick interval based on zoom
    const targetPixelsPerTick = 80;
    const rawInterval = targetPixelsPerTick / pixelsPerSecond;
    const niceIntervals = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    let tickInterval = niceIntervals.find(n => n >= rawInterval) || 3600;

    ctx.fillStyle = COLORS.text;
    ctx.font = "10px monospace";
    ctx.textAlign = "center";

    const firstTick = Math.ceil(DATA.time_range[0] / tickInterval) * tickInterval;
    for (let t = firstTick; t <= DATA.time_range[1]; t += tickInterval) {
        const x = timeToX(t);
        if (x < CLUSTER_LABEL_WIDTH || x > W) continue;

        ctx.strokeStyle = "#0f3460";
        ctx.beginPath();
        ctx.moveTo(x, HEADER_HEIGHT);
        ctx.lineTo(x, H);
        ctx.stroke();

        let label;
        if (tickInterval >= 60) label = (t / 60).toFixed(0) + "m";
        else label = t.toFixed(0) + "s";
        ctx.fillStyle = COLORS.text;
        ctx.fillText(label, x, HEADER_HEIGHT - 8);
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

        // Cluster background
        ctx.fillStyle = COLORS.clusterBg;
        ctx.fillRect(0, pos.y, W, pos.height);

        // Pending period background highlight
        if (cl.pending_until && cl.firstSeen < cl.pending_until) {
            const px0 = Math.max(CLUSTER_LABEL_WIDTH, timeToX(cl.firstSeen));
            const px1 = timeToX(cl.pending_until);
            if (px1 > CLUSTER_LABEL_WIDTH) {
                ctx.fillStyle = COLORS.pendingBg;
                ctx.fillRect(px0, pos.y, px1 - px0, pos.height);
                // Pending marker line
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

        ctx.fillStyle = "#e0e0e0";
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "left";
        // Shorten cluster name
        let displayName = name;
        if (displayName.length > 18) displayName = displayName.substring(0, 18) + "…";
        ctx.fillText(displayName, 6, pos.y + 14);

        ctx.fillStyle = COLORS.text;
        ctx.font = "10px monospace";
        const numQs = queriesByCluster[name].filter(q => q.start_s <= state.currentTime).length;
        const activeQs = queriesByCluster[name].filter(q => q.start_s <= state.currentTime && q.end_s > state.currentTime && q.state !== "completed" || (q.start_s <= state.currentTime && q.end_s > state.currentTime)).length;
        ctx.fillText(`${cl.rpu} RPU · ${numQs} queries`, 6, pos.y + 28);
    });

    // Draw query bars
    realClusters.forEach(clusterName => {
        const pos = clusterYPositions[clusterName];
        const queries = queriesByCluster[clusterName];

        queries.forEach(q => {
            // Only draw queries that have started by current time
            if (q.start_s > state.currentTime) return;

            const effectiveEnd = Math.min(q.end_s, q.state === "completed" ? q.end_s : state.currentTime);
            const x0 = timeToX(q.start_s);
            const x1 = timeToX(effectiveEnd);
            const barWidth = Math.max(1, x1 - x0);

            const laneY = pos.y + CLUSTER_PADDING + q._lane * (ROW_HEIGHT + LANE_GAP);
            const barHeight = ROW_HEIGHT;

            // Color based on state
            let color;
            if (q.state !== "completed" || q.end_s > state.currentTime) {
                color = COLORS.running;
            } else if (q.violates_slo) {
                color = COLORS.violated;
            } else {
                color = COLORS.met;
            }

            // Skip if entirely outside viewport
            if (x1 < CLUSTER_LABEL_WIDTH || x0 > parseFloat(canvas.style.width)) return;

            ctx.fillStyle = color;
            ctx.fillRect(Math.max(CLUSTER_LABEL_WIDTH, x0), laneY, barWidth - Math.max(0, CLUSTER_LABEL_WIDTH - x0), barHeight);
        });
    });

    // Draw scaling events as markers
    DATA.cluster_events.forEach(e => {
        if (e.timestamp > state.currentTime) return;
        const x = timeToX(e.timestamp);
        if (x < CLUSTER_LABEL_WIDTH) return;

        if (e.event_type === "cluster_ready") {
            // Green triangle on the cluster row
            const clName = e.cluster_name;
            if (clName in clusterYPositions) {
                const pos = clusterYPositions[clName];
                ctx.fillStyle = COLORS.met;
                ctx.beginPath();
                ctx.moveTo(x, pos.y + 2);
                ctx.lineTo(x + 5, pos.y + 8);
                ctx.lineTo(x - 5, pos.y + 8);
                ctx.fill();
            }
        }
        if (e.event_type === "tear_down_requested" || e.event_type === "tear_down_decision") {
            const clName = e.cluster_name;
            if (clName in clusterYPositions) {
                const pos = clusterYPositions[clName];
                ctx.fillStyle = COLORS.violated;
                ctx.beginPath();
                ctx.moveTo(x - 4, pos.y + 2);
                ctx.lineTo(x + 4, pos.y + 2);
                ctx.lineTo(x, pos.y + 8);
                ctx.fill();
            }
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
    // Show events up to current time, last 200
    const visible = allEvents.filter(e => e.timestamp <= state.currentTime);
    const toShow = visible.slice(-200);

    document.getElementById("event-count").textContent = visible.length;

    // Build HTML
    let html = "";
    toShow.forEach(e => {
        const tStr = e.timestamp.toFixed(1);
        let typeClass = "";
        html += `<div class="event-item"><span class="event-time">${tStr}s</span><span class="event-type">${e.type}</span><span class="event-detail">${e.detail || ""}</span></div>`;
    });
    container.innerHTML = html;

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

// ===========================================================================
// TOOLTIP
// ===========================================================================

const tooltipEl = document.getElementById("tooltip");

function showTooltip(x, y, q) {
    const sloLabel = q.violates_slo ? "tt-violation" : "tt-met";
    const sloText = q.violates_slo ? "VIOLATED" : "MET";
    tooltipEl.innerHTML = `
        <div class="tt-row"><span class="tt-label">Query:</span> <span class="tt-value">${q.query_id}</span></div>
        <div class="tt-row"><span class="tt-label">Template:</span> <span class="tt-value">${q.query_text_id}</span></div>
        <div class="tt-row"><span class="tt-label">Cluster:</span> <span class="tt-value">${q.cluster_name}</span></div>
        <div class="tt-row"><span class="tt-label">Start:</span> <span class="tt-value">${q.start_s.toFixed(1)}s</span></div>
        <div class="tt-row"><span class="tt-label">Latency:</span> <span class="tt-value">${q.latency_s.toFixed(2)}s</span></div>
        <div class="tt-row"><span class="tt-label">SLO:</span> <span class="tt-value">${q.slo_s.toFixed(1)}s</span> <span class="${sloLabel}">(${sloText})</span></div>
        <div class="tt-row"><span class="tt-label">State:</span> <span class="tt-value">${q.state}</span></div>
    `;
    tooltipEl.style.display = "block";
    tooltipEl.style.left = Math.min(x + 10, window.innerWidth - 420) + "px";
    tooltipEl.style.top = Math.min(y + 10, window.innerHeight - 200) + "px";
}

function hideTooltip() {
    tooltipEl.style.display = "none";
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
            const effectiveEnd = Math.min(q.end_s, q.state === "completed" ? q.end_s : state.currentTime);
            const x1 = timeToX(effectiveEnd);
            const laneY = pos.y + CLUSTER_PADDING + q._lane * (ROW_HEIGHT + LANE_GAP);

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

// Zoom
document.getElementById("btn-zoom-in").addEventListener("click", () => {
    state.zoom *= 1.5;
    drawTimeline();
});
document.getElementById("btn-zoom-out").addEventListener("click", () => {
    state.zoom = Math.max(0.01, state.zoom / 1.5);
    drawTimeline();
});
document.getElementById("btn-zoom-fit").addEventListener("click", () => {
    const viewW = viewport.clientWidth - CLUSTER_LABEL_WIDTH - 40;
    const timeSpan = DATA.time_range[1] - DATA.time_range[0];
    state.zoom = timeSpan > 0 ? viewW / timeSpan : 1;
    state.panX = 0;
    drawTimeline();
});

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
    const dpr = window.devicePixelRatio || 1;
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const hit = hitTest(cx, cy);
    if (hit) {
        showTooltip(e.clientX, e.clientY, hit);
        canvas.style.cursor = "pointer";
    } else {
        hideTooltip();
        canvas.style.cursor = "default";
    }
});
canvas.addEventListener("mouseleave", hideTooltip);

// Keyboard
document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") {
        // Step back one arrival
        const prevIdx = DATA.arrival_times.findIndex(t => t >= state.currentTime) - 1;
        if (prevIdx >= 0) setTime(DATA.arrival_times[prevIdx]);
        else if (DATA.arrival_times.length > 0) setTime(DATA.arrival_times[0]);
    } else if (e.key === "ArrowRight") {
        // Step forward one arrival
        const nextIdx = DATA.arrival_times.findIndex(t => t > state.currentTime);
        if (nextIdx >= 0) setTime(DATA.arrival_times[nextIdx]);
        else setTime(DATA.time_range[1]);
    } else if (e.key === " ") {
        e.preventDefault();
        document.getElementById("btn-play").click();
    }
});

// ===========================================================================
// INIT
// ===========================================================================

window.addEventListener("resize", () => { drawTimeline(); });

// Initial zoom to fit
{
    const viewW = viewport.clientWidth - CLUSTER_LABEL_WIDTH - 40;
    const timeSpan = DATA.time_range[1] - DATA.time_range[0];
    state.zoom = timeSpan > 0 ? viewW / timeSpan : 1;
}
setTime(DATA.time_range[1]);

</script>
</body>
</html>"""


def generate_html(data: dict) -> str:
    """Inject timeline data into the HTML template."""
    data_json = json.dumps(data, default=str)
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

    kind = _detect_log_kind(df)
    print(f"  Detected log kind: {kind}")

    if kind == "simulator":
        data = _parse_simulator_log(df, slo_resolver)
    else:
        data = _parse_runner_log(df, slo_resolver)

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
