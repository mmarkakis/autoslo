"""
Analyze cluster spin-up latency from an autoslo structured log (or all logs
referenced by an execution manifest).

For every cluster that was spun up, builds a timeline of the key events:
  SPIN_UP_DECISION → SPIN_UP_REQUESTED → SPIN_UP_STARTED
  → (SPIN_UP_BLOCKED) → CLUSTER_READY

Prints one row per cluster with the wall-clock timestamp of each milestone
and the total span (in seconds) from the earliest available event to CLUSTER_READY.

Usage:
    # Single log:
    python analyze_spinup_latency.py --log <path/to/structured_log.parquet>

    # All runs in a manifest (simulator runs only):
    python analyze_spinup_latency.py --manifest autoscaler_eval_v2
    python analyze_spinup_latency.py --manifest /abs/path/to/manifest.yml
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.table import Table

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import Cluster
from autoslo.config.component_configs import WorkloadConfig
from autoslo.config.utils import make_run_id
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.path_utils import find_most_recent_live_run_id
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.filesystem.yaml_helpers import load_yaml

SPINUP_EVENTS = {
    "spin_up_decision",
    "spin_up_requested",
    "spin_up_started",
    "spin_up_blocked",
    "cluster_ready",
}


@dataclass
class _Input:
    log_path: Path
    run_id: str


@dataclass
class _Args:
    inputs: list["_Input"]
    only_summary: bool


def _resolve_inputs_from_manifest(manifest_path: Path) -> list[_Input]:
    """Return (log_path, run_id) pairs for every entry in the manifest."""
    manifest = load_yaml(manifest_path)
    entries = manifest.get("main_content", [])

    inputs: list[_Input] = []
    for entry in entries:
        workload_config = WorkloadConfig.from_config(entry)
        exec_cfg_path = resolve_config(entry["execution_config"])
        params = entry.get("params", {})
        config_label = make_run_id([exec_cfg_path.stem], params)
        wid = workload_config.id()
        run_id = find_most_recent_live_run_id(config_label, wid)
        if run_id is None:
            continue
        log_path = pu.get_runs_dir() / run_id / "structured_log.parquet"
        if log_path.exists():
            inputs.append(_Input(log_path=log_path, run_id=run_id))
        else:
            Console().print(
                f"[yellow]Warning: structured_log.parquet not found for run {run_id} at {log_path}. Skipping.[/]"
            )
    return inputs


def parse_args() -> _Args:
    parser = argparse.ArgumentParser(
        description="Analyze cluster spin-up latency from a structured log."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--log",
        metavar="structured_log.parquet",
        type=Path,
        help="Path to a single structured_log.parquet file.",
    )
    group.add_argument(
        "--manifest",
        metavar="NAME_OR_PATH",
        help=(
            "Name of an execution manifest (resolved under "
            "data/manifests/execution) or an explicit path to a .yml file. "
            "Processes all live runs referenced by the manifest."
        ),
    )
    parser.add_argument(
        "--only_summary",
        action="store_true",
        help="Only print the per-RPU summary table, skipping the per-cluster detail table.",
    )
    args = parser.parse_args()

    if args.log is not None:
        inputs = [_Input(log_path=args.log, run_id=args.log.parent.name)]
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute() and not manifest_path.exists():
            manifest_path = (
                pu.get_data_dir()
                / "manifests"
                / "execution"
                / manifest_path.with_suffix(".yml")
            )
        if not manifest_path.exists():
            parser.error(f"Manifest not found: {manifest_path}")
        inputs = _resolve_inputs_from_manifest(manifest_path)

    return _Args(inputs=inputs, only_summary=args.only_summary)


def fmt_ts(epoch: Optional[float]) -> str:
    """Format an epoch-seconds wall-clock value as a UTC datetime string."""
    if epoch is None or (isinstance(epoch, float) and pd.isna(epoch)):
        return "[dim]—[/dim]"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def fmt_s(value: Optional[float], decimals: int = 1) -> str:
    """Format a float number of seconds, or a dim dash for missing values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "[dim]—[/dim]"
    return f"{value:.{decimals}f} s"


def records_from_log(log_path: Path, run_id: str) -> list[dict]:
    """Extract one dict per spun-up cluster from a structured log."""
    df = StructuredLog.load(log_path).df

    ev = (
        df[df["event_type"].isin(SPINUP_EVENTS)]
        .sort_values("rel_time_s")
        .reset_index(drop=True)
    )

    started_rows = ev[ev["event_type"] == "spin_up_started"].copy()
    started_times = [float("-inf")] + started_rows["rel_time_s"].tolist()

    records = []
    for i, (_, started_row) in enumerate(started_rows.iterrows()):
        t_start = started_row["rel_time_s"]
        t_window_begin = started_times[i]
        cluster_name = started_row["cluster_name"]

        window = ev[
            (ev["rel_time_s"] > t_window_begin)
            & (ev["rel_time_s"] <= t_start)
            & ev["event_type"].isin(
                {"spin_up_decision", "spin_up_requested", "spin_up_blocked"}
            )
        ]

        def first_time(event_type: str) -> Optional[float]:
            rows = window[window["event_type"] == event_type]
            return float(rows["rel_time_s"].iloc[0]) if len(rows) else None

        def first_wall(event_type: str) -> Optional[float]:
            rows = window[window["event_type"] == event_type]
            return float(rows["wall_clock_s"].iloc[0]) if len(rows) else None

        def first_detail(event_type: str, key: str):
            rows = window[window["event_type"] == event_type]
            return rows["details"].iloc[0].get(key) if len(rows) else None

        t_decision = first_time("spin_up_decision")
        t_requested = first_time("spin_up_requested")
        t_blocked = first_time("spin_up_blocked")
        t_started = float(t_start)

        w_decision = first_wall("spin_up_decision")
        w_requested = first_wall("spin_up_requested")
        w_blocked = first_wall("spin_up_blocked")
        w_started = float(started_row["wall_clock_s"])

        ready_rows = ev[
            (ev["event_type"] == "cluster_ready")
            & (ev["cluster_name"] == cluster_name)
            & (ev["rel_time_s"] >= t_started)
            & (ev["source"] == "ManagedClusterPool")
        ]
        w_ready = (
            float(ready_rows["wall_clock_s"].iloc[0])
            if len(ready_rows)
            else None
        )

        w_first = next(
            (w for w in [w_decision, w_requested, w_started] if w is not None),
            None,
        )
        span = (
            (w_ready - w_first)
            if (w_ready is not None and w_first is not None)
            else None
        )

        records.append(
            {
                "run_id": run_id,
                "cluster": cluster_name,
                "rpu": Cluster.rpu_for_cluster_name(cluster_name),
                "w_decision": w_decision,
                "w_requested": w_requested,
                "w_started": w_started,
                "w_blocked": w_blocked,
                "w_ready": w_ready,
                "span_s": span,
            }
        )

    return records


def main() -> None:
    parsed = parse_args()
    inputs = parsed.inputs
    only_summary = parsed.only_summary

    all_records: list[dict] = []
    for inp in inputs:
        all_records.extend(records_from_log(inp.log_path, inp.run_id))

    console = Console()
    console.print()
    if len(inputs) == 1:
        console.print(
            f"[bold]Cluster spin-up latency[/bold]  –  "
            f"[cyan]{inputs[0].log_path}[/cyan]"
        )
    else:
        console.print(
            f"[bold]Cluster spin-up latency[/bold]  –  "
            f"[cyan]{len(inputs)} runs[/cyan]"
        )
    console.print(f"Total clusters spun up: [bold]{len(all_records)}[/bold]")
    console.print()

    if not only_summary:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Run ID", justify="left", no_wrap=True)
        table.add_column("Cluster", justify="left", no_wrap=True)
        table.add_column("RPU", justify="right")
        table.add_column("Decision (UTC)", justify="right", no_wrap=True)
        table.add_column("Requested (UTC)", justify="right", no_wrap=True)
        table.add_column("Started (UTC)", justify="right", no_wrap=True)
        table.add_column("Blocked (UTC)", justify="right", no_wrap=True)
        table.add_column("Ready (UTC)", justify="right", no_wrap=True)
        table.add_column("Span", justify="right", no_wrap=True)

        prev_run_id: Optional[str] = None
        for rec in all_records:
            # Dim the run_id when it repeats consecutively for readability.
            run_id_cell = (
                f"[dim]{rec['run_id']}[/dim]"
                if rec["run_id"] == prev_run_id
                else rec["run_id"]
            )
            prev_run_id = rec["run_id"]
            table.add_row(
                run_id_cell,
                rec["cluster"],
                str(rec["rpu"]) if rec["rpu"] is not None else "[dim]—[/dim]",
                fmt_ts(rec["w_decision"]),
                fmt_ts(rec["w_requested"]),
                fmt_ts(rec["w_started"]),
                fmt_ts(rec["w_blocked"]),
                fmt_ts(rec["w_ready"]),
                fmt_s(rec["span_s"]),
            )

        console.print(table)
        console.print()

    # --- Summary table: group by RPU ---
    import statistics

    rpu_spans: dict[Optional[int], list[float]] = {}
    for rec in all_records:
        rpu = rec["rpu"]
        span = rec["span_s"]
        if span is not None:
            rpu_spans.setdefault(rpu, []).append(span)

    if rpu_spans:
        console.print("[bold]Span summary by RPU[/bold]")
        console.print()
        summary = Table(show_header=True, header_style="bold magenta")
        summary.add_column("RPU", justify="right")
        summary.add_column("Count", justify="right")
        summary.add_column("Min", justify="right", no_wrap=True)
        summary.add_column("Max", justify="right", no_wrap=True)
        summary.add_column("Mean", justify="right", no_wrap=True)
        summary.add_column("Median", justify="right", no_wrap=True)

        for rpu in sorted(rpu_spans.keys(), key=lambda x: (x is None, x)):
            spans = rpu_spans[rpu]
            summary.add_row(
                str(rpu) if rpu is not None else "[dim]—[/dim]",
                str(len(spans)),
                fmt_s(min(spans)),
                fmt_s(max(spans)),
                fmt_s(statistics.mean(spans)),
                fmt_s(statistics.median(spans)),
            )

        console.print(summary)
        console.print()


if __name__ == "__main__":
    main()
