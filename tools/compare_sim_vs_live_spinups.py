"""
compare_sim_vs_live_spinups.py
----------------------
For each entry in an execution manifest, prints a rich side-by-side table
comparing the most-recent simulator run (left) with the most-recent live run
(right).

Each half has one row per cluster and shows:
  • Cluster name (color-coded by spin-up type)
  • Spin-up time (rel_time_s when SPIN_UP_STARTED)
  • Ready time  (rel_time_s when CLUSTER_READY)
  • Tear-down time (rel_time_s of CLUSTER_REMOVED, or TEAR_DOWN_STARTED)
  • # queries executed
  • # and fraction of queries that met their SLO
  • Per-cluster cost

Color convention for spin-up type (shown as a prefix in the cluster name):
  [cyan]   ● initial
  [yellow] ◆ scheduled
  [red]    ▲ reactive  (autoscaler-triggered)

Footer rows per half show totals / sums.

Usage:
    python tools/compare_sim_vs_live_spinups.py --manifest autoscaler_eval_v2
    python tools/compare_sim_vs_live_spinups.py --manifest /abs/path/to/manifest.yml
    python tools/compare_sim_vs_live_spinups.py --manifest my_manifest --no_sim
    python tools/compare_sim_vs_live_spinups.py --manifest my_manifest --no_live
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from rich import box
from rich.console import Console
from rich.table import Table

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import Cluster
from autoslo.config.component_configs import SloResolverConfig, WorkloadConfig
from autoslo.config.utils import make_run_id
from autoslo.filesystem.config_resolver import resolve_config
from autoslo.filesystem.path_utils import find_most_recent_live_run_id
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_execution.trace import Trace

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_t(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "[dim]—[/]"
    return f"{v:,.1f}s"


def _fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "[dim]—[/]"
    return f"{num / den:.1%}"


def _fmt_cost(v: float) -> str:
    return f"${v:.4f}"


# ---------------------------------------------------------------------------
# Per-cluster stats extraction
# ---------------------------------------------------------------------------


@dataclass
class ClusterStats:
    name: str
    spinup_type: str  # "initial" | "scheduled" | "reactive" | "unknown"
    rpu: int
    spinup_started_s: Optional[float]
    ready_s: Optional[float]
    teardown_s: Optional[float]
    n_queries: int
    n_slo_met: int
    cost: float


def _extract_cluster_stats(
    log_path: Path,
    slo_resolver: SloResolver,
    cost_by_cluster: dict[str, float],
) -> list[ClusterStats]:
    """
    Parse a structured_log.parquet and return one ClusterStats per cluster
    that was spun up (SPIN_UP_STARTED present).
    """
    sl = StructuredLog.load(log_path)
    df = sl.df

    # --- Cluster lifecycle events -----------------------------------------
    lifecycle = df[
        df["event_type"].isin(
            {
                "spin_up_requested",
                "spin_up_started",
                "cluster_ready",
                "cluster_removed",
                "tear_down_started",
            }
        )
    ].copy()
    lifecycle = lifecycle.sort_values("rel_time_s").reset_index(drop=True)

    started_rows = lifecycle[lifecycle["event_type"] == "spin_up_started"]

    # Boundary times so we can window each cluster's pre-start events.
    started_times_list: list[float] = started_rows["rel_time_s"].tolist()
    window_starts = [float("-inf")] + started_times_list  # one lookahead guard

    # --- Per-query data: cluster assignment and latency -------------------
    # Build query_id → cluster_name from COMPLETION events (they carry cluster_name).
    completion_df = df[df["event_type"] == "completion"].copy()
    completion_df = completion_df.dropna(subset=["query_id"])
    completion_map: dict[str, str] = {}  # query_id → cluster_name
    for _, row in completion_df.iterrows():
        qid = str(row["query_id"])
        cname = row.get("cluster_name", "")
        if cname and cname not in ("", None):
            completion_map[qid] = str(cname)

    # Arrival times.
    arrival_df = df[df["event_type"] == "arrival"].copy()
    arrival_df = arrival_df.dropna(subset=["query_id"])
    arrival_time: dict[str, float] = {
        str(row["query_id"]): float(row["rel_time_s"])
        for _, row in arrival_df.iterrows()
    }
    # Completion times and success.
    completion_time: dict[str, float] = {}
    completion_success: dict[str, bool] = {}
    completion_text_id: dict[str, str] = {}
    for _, row in completion_df.iterrows():
        qid = str(row["query_id"])
        completion_time[qid] = float(row["rel_time_s"])
        details = row.get("details", {})
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except Exception:
                details = {}
        completion_success[qid] = bool(details.get("success", True))
        completion_text_id[qid] = str(row.get("query_text_id", ""))

    # Per-cluster query accumulator.
    cluster_queries: dict[str, list[tuple[float, float, str, bool]]] = {}
    # (latency_s, slo_s, query_text_id, success) per query per cluster

    for qid, cname in completion_map.items():
        if cname not in cluster_queries:
            cluster_queries[cname] = []
        arr = arrival_time.get(qid)
        comp = completion_time.get(qid)
        success = completion_success.get(qid, True)
        text_id = completion_text_id.get(qid, "")
        if arr is not None and comp is not None:
            latency_s = comp - arr
            slo_s = slo_resolver.resolve(text_id) or 0.0
            cluster_queries[cname].append((latency_s, slo_s, text_id, success))

    # --- Build one ClusterStats per SPIN_UP_STARTED row ------------------
    results: list[ClusterStats] = []
    for i, (_, started_row) in enumerate(started_rows.iterrows()):
        t_start: float = float(started_row["rel_time_s"])
        t_window_begin: float = window_starts[i]
        cluster_name: str = str(started_row.get("cluster_name", ""))

        # Find the SPIN_UP_REQUESTED that preceded this SPIN_UP_STARTED.
        # SPIN_UP_REQUESTED carries the same action.reason as SPIN_UP_DECISION
        # and is emitted immediately before the cluster is added to the pool.
        requested_window = lifecycle[
            (lifecycle["event_type"] == "spin_up_requested")
            & (lifecycle["rel_time_s"] > t_window_begin)
            & (lifecycle["rel_time_s"] <= t_start)
        ]
        if not requested_window.empty:
            requested_row = requested_window.iloc[-1]
            details = requested_row.get("details", {})
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {}
            reason: str = str(details.get("reason", ""))
            if reason == "initial":
                spinup_type = "initial"
            elif reason.startswith("scheduled_spinup"):
                spinup_type = "scheduled"
            else:
                spinup_type = "reactive"
        else:
            spinup_type = "unknown"

        # CLUSTER_READY for this cluster (first after spin_up_started).
        ready_rows = lifecycle[
            (lifecycle["event_type"] == "cluster_ready")
            & (lifecycle["cluster_name"] == cluster_name)
            & (lifecycle["rel_time_s"] >= t_start)
        ]
        ready_s: Optional[float] = (
            float(ready_rows["rel_time_s"].iloc[0])
            if not ready_rows.empty
            else None
        )

        # CLUSTER_REMOVED or TEAR_DOWN_STARTED (first after ready/start).
        after_start = lifecycle[
            (lifecycle["cluster_name"] == cluster_name)
            & (lifecycle["rel_time_s"] >= t_start)
        ]
        removed_rows = after_start[
            after_start["event_type"].isin(
                {"cluster_removed", "tear_down_started"}
            )
        ]
        teardown_s: Optional[float] = (
            float(removed_rows["rel_time_s"].iloc[0])
            if not removed_rows.empty
            else None
        )

        # Query SLO stats for this cluster.
        queries = cluster_queries.get(cluster_name, [])
        n_queries = len(queries)
        n_slo_met = 0
        for latency_s, slo_s, text_id, success in queries:
            if success and (slo_s == 0 or latency_s <= slo_s):
                n_slo_met += 1

        cost = cost_by_cluster.get(cluster_name, 0.0)

        try:
            rpu = Cluster.rpu_for_cluster_name(cluster_name)
        except ValueError:
            rpu = 0

        results.append(
            ClusterStats(
                name=cluster_name,
                spinup_type=spinup_type,
                rpu=rpu,
                spinup_started_s=t_start,
                ready_s=ready_s,
                teardown_s=teardown_s,
                n_queries=n_queries,
                n_slo_met=n_slo_met,
                cost=cost,
            )
        )

    return results


# ---------------------------------------------------------------------------
# SloResolver loader
# ---------------------------------------------------------------------------


def _slo_resolver_for_dir(run_dir: Path) -> SloResolver:
    config_path = run_dir / "execution_config.yml"
    with open(config_path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return SloResolver(SloResolverConfig.from_config(cfg))


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

_SPINUP_MARKUP = {
    "initial": "[cyan]●[/]",
    "scheduled": "[yellow]◆[/]",
    "reactive": "[red]▲[/]",
    "unknown": "[dim]?[/]",
}

_SPINUP_COLOR = {
    "initial": "cyan",
    "scheduled": "yellow",
    "reactive": "red",
    "unknown": "dim",
}


def _half_rows(
    stats: list[ClusterStats],
) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """
    Build table cell strings for a run half.

    Returns (data_rows, rpu_rows, footer_row).
    Each element of data_rows: [name, spinup_s, ready_s, teardown_s, n_q, n_met, n_viol, pct, cost]
    rpu_rows: one row per unique RPU size, aggregated over clusters with that RPU.
    footer_row: same shape, with totals.
    """
    rows: list[list[str]] = []
    for cs in stats:
        marker = _SPINUP_MARKUP.get(cs.spinup_type, "")
        color = _SPINUP_COLOR.get(cs.spinup_type, "white")
        name_cell = f"{marker} [{color}]{cs.name}[/]"
        rows.append(
            [
                name_cell,
                _fmt_t(cs.spinup_started_s),
                _fmt_t(cs.ready_s),
                _fmt_t(cs.teardown_s),
                str(cs.n_queries),
                str(cs.n_slo_met),
                str(cs.n_queries - cs.n_slo_met),
                _fmt_pct(cs.n_queries - cs.n_slo_met, cs.n_queries),
                _fmt_cost(cs.cost),
            ]
        )

    # RPU-aggregated rows
    rpu_groups: dict[int, list[ClusterStats]] = {}
    for cs in stats:
        rpu_groups.setdefault(cs.rpu, []).append(cs)
    rpu_rows: list[list[str]] = []
    for rpu_size in sorted(rpu_groups.keys()):
        group = rpu_groups[rpu_size]
        n_q = sum(cs.n_queries for cs in group)
        n_met = sum(cs.n_slo_met for cs in group)
        n_viol = n_q - n_met
        cost = sum(cs.cost for cs in group)
        n_clusters = len(group)
        label = (
            f"[bold magenta]{rpu_size} RPU[/] "
            f"[dim]({n_clusters} cluster{'s' if n_clusters != 1 else ''})[/]"
        )
        rpu_rows.append(
            [
                label,
                "",
                "",
                "",
                str(n_q),
                str(n_met),
                str(n_viol),
                _fmt_pct(n_viol, n_q),
                _fmt_cost(cost),
            ]
        )

    total_q = sum(cs.n_queries for cs in stats)
    total_met = sum(cs.n_slo_met for cs in stats)
    total_cost = sum(cs.cost for cs in stats)
    footer = [
        "[bold]TOTAL[/]",
        "",
        "",
        "",
        f"[bold]{total_q}[/]",
        f"[bold]{total_met}[/]",
        f"[bold]{total_q - total_met}[/]",
        f"[bold]{_fmt_pct(total_q - total_met, total_q)}[/]",
        f"[bold]{_fmt_cost(total_cost)}[/]",
    ]
    return rows, rpu_rows, footer


_HALF_HEADERS = [
    "Cluster",
    "Spin-up",
    "Ready",
    "Tear-down",
    "Queries",
    "SLO met",
    "SLO viol.",
    "SLO viol. rate",
    "Cost",
]

_N_COLS = len(_HALF_HEADERS)


def _render_entry_table(
    title: str,
    sim_stats: Optional[list[ClusterStats]],
    live_stats: Optional[list[ClusterStats]],
) -> None:
    """Print one rich table for a single manifest entry."""
    table = Table(
        title=title,
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        title_style="bold magenta",
        expand=False,
    )

    # Build column headers: left half (sim) | divider | right half (live)
    for h in _HALF_HEADERS:
        table.add_column(f"[blue]{h}[/]", no_wrap=True)
    table.add_column(" ", no_wrap=True, width=1)  # divider
    for h in _HALF_HEADERS:
        table.add_column(f"[green]{h}[/]", no_wrap=True)

    sim_rows, sim_rpu_rows, sim_footer = (
        _half_rows(sim_stats)
        if sim_stats is not None
        else ([], [], ["[dim]—[/]"] * _N_COLS)
    )
    live_rows, live_rpu_rows, live_footer = (
        _half_rows(live_stats)
        if live_stats is not None
        else ([], [], ["[dim]—[/]"] * _N_COLS)
    )

    _blank = ["[dim]—[/]"] * _N_COLS
    _empty = [""] * _N_COLS

    # Per-cluster rows
    n_rows = max(len(sim_rows), len(live_rows))
    for i in range(n_rows):
        sim_r = sim_rows[i] if i < len(sim_rows) else _blank
        live_r = live_rows[i] if i < len(live_rows) else _blank
        table.add_row(*(sim_r + ["|"] + live_r))

    # RPU-aggregated rows
    if sim_rpu_rows or live_rpu_rows:
        table.add_section()
        n_rpu_rows = max(len(sim_rpu_rows), len(live_rpu_rows))
        for i in range(n_rpu_rows):
            sim_r = sim_rpu_rows[i] if i < len(sim_rpu_rows) else _empty
            live_r = live_rpu_rows[i] if i < len(live_rpu_rows) else _empty
            table.add_row(*(sim_r + ["|"] + live_r))

    # Footer separator + totals
    if sim_stats is not None or live_stats is not None:
        table.add_section()
        sim_f = sim_footer if sim_stats is not None else _blank
        live_f = live_footer if live_stats is not None else _blank
        table.add_row(*(sim_f + ["|"] + live_f), style="bold")

    console.print()
    console.rule()
    console.print(table)

    # Legend
    console.print(
        "  Spin-up type: "
        + "  ".join(
            f"[{_SPINUP_COLOR[st]}]{_SPINUP_MARKUP[st]} {st}[/]"
            for st in _SPINUP_MARKUP.keys()
        )
        + "  Columns: [blue]Simulator[/] | [green]Live[/]"
    )


# ---------------------------------------------------------------------------
# Manifest resolution
# ---------------------------------------------------------------------------


@dataclass
class _EntryPaths:
    label: str
    wid: str
    sim_dir: Optional[Path]
    live_dir: Optional[Path]


def _resolve_entries(
    manifest_path: Path,
    include_sim: bool,
    include_live: bool,
) -> list[_EntryPaths]:
    manifest = load_yaml(manifest_path)
    entries = manifest.get("main_content", [])

    result: list[_EntryPaths] = []
    for entry in entries:
        workload_config = WorkloadConfig.from_config(entry)
        exec_cfg_path = resolve_config(entry["execution_config"])
        params = entry.get("params", {})
        config_label = make_run_id([exec_cfg_path.stem], params)
        wid = workload_config.id()

        sim_dir: Optional[Path] = None
        if include_sim:
            candidate = (
                pu.get_data_dir() / "simulator_runs" / wid / config_label
            )
            if (candidate / "execution_config.yml").exists():
                sim_dir = candidate
            else:
                console.print(
                    f"[dim]No simulator run found for '{config_label}' / '{wid}'[/]"
                )

        live_dir: Optional[Path] = None
        if include_live:
            run_id = find_most_recent_live_run_id(config_label, wid)
            if run_id is not None:
                candidate = pu.get_runs_dir() / run_id
                if (candidate / "execution_config.yml").exists():
                    live_dir = candidate
            if live_dir is None:
                console.print(
                    f"[dim]No live run found for '{config_label}' / '{wid}'[/]"
                )

        result.append(
            _EntryPaths(
                label=config_label,
                wid=wid,
                sim_dir=sim_dir,
                live_dir=live_dir,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare simulator vs live run results side-by-side, "
            "per cluster, for each entry in an execution manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        metavar="NAME_OR_PATH",
        help=(
            "Name of an execution manifest (resolved under "
            "data/manifests/execution) or an explicit path to a .yml file."
        ),
    )
    parser.add_argument(
        "--no_sim",
        action="store_true",
        help="Skip simulator runs (show only live).",
    )
    parser.add_argument(
        "--no_live",
        action="store_true",
        help="Skip live runs (show only simulator).",
    )
    args = parser.parse_args()

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

    include_sim = not args.no_sim
    include_live = not args.no_live

    entries = _resolve_entries(manifest_path, include_sim, include_live)

    for ep in entries:
        sim_stats: Optional[list[ClusterStats]] = None
        live_stats: Optional[list[ClusterStats]] = None

        if ep.sim_dir is not None:
            log_path = ep.sim_dir / "structured_log.parquet"
            if log_path.exists():
                try:
                    resolver = _slo_resolver_for_dir(ep.sim_dir)
                    billing_path = ep.sim_dir / "billing_interval_analysis.yml"
                    cost_map = (
                        {
                            name: float(data.get("total_billed_cost", 0.0))
                            for name, data in load_yaml(billing_path).items()
                        }
                        if billing_path.exists()
                        else {}
                    )
                    sim_stats = _extract_cluster_stats(
                        log_path, resolver, cost_map
                    )
                except Exception as exc:
                    console.print(
                        f"[red]Error reading simulator run {ep.sim_dir}: {exc}[/]"
                    )
            else:
                console.print(
                    f"[yellow]structured_log.parquet not found in {ep.sim_dir}[/]"
                )

        if ep.live_dir is not None:
            log_path = ep.live_dir / "structured_log.parquet"
            if log_path.exists():
                try:
                    resolver = _slo_resolver_for_dir(ep.live_dir)
                    trace = Trace(ep.live_dir.name)
                    cost_map = {
                        c: trace.cost_of_cluster(c) for c in trace.seen_clusters
                    }
                    live_stats = _extract_cluster_stats(
                        log_path, resolver, cost_map
                    )
                except Exception as exc:
                    console.print(
                        f"[red]Error reading live run {ep.live_dir}: {exc}[/]"
                    )
            else:
                console.print(
                    f"[yellow]structured_log.parquet not found in {ep.live_dir}[/]"
                )

        if sim_stats is None and live_stats is None:
            console.print(
                f"[yellow]No data for '{ep.label}' / '{ep.wid}' — skipping.[/]"
            )
            continue

        title = f"{ep.wid}  ·  {ep.label}"
        _render_entry_table(title, sim_stats, live_stats)


if __name__ == "__main__":
    main()
