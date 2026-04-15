"""
runner_router.py
----------------
FastAPI router for live WorkloadRunner runs stored in ``data/runs/``.

Endpoints
---------
GET /api/runner/experiments
    List experiment names (grouped from runner_config.yml experiment_name).

GET /api/runner/experiments/{name}
    Return run summaries for the named experiment.

GET /api/runner/runs/{experiment}/{run_id}/timeline
    Return a TimelineData response built from Trace (sys_query_history).

GET /api/runner/runs/{experiment}/{run_id}/template_stats
    Return per-template compliance statistics.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException

import autoslo.utils.paths as pu
from autoslo.workload_execution.trace import Trace

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared Pydantic schemas (same contracts as simulator_router)
# ---------------------------------------------------------------------------

from autoslo.api.routers.simulator_router import (
    ExperimentSummary,
    RunSummary,
    TemplateStats,
    TimelineData,
    TimelineInterval,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _runs_dir() -> str:
    return pu.get_runs_path()


def _run_dir(run_id: str) -> str:
    return os.path.join(_runs_dir(), run_id)


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{label} not found")


def _rpu_for_runner_cluster(cluster_name: str) -> int:
    """Extract RPU from runner cluster names.

    Supports ``autoslo-{rpu}-{ts}-{seq}``.
    """
    parts = cluster_name.split("-")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    raise ValueError(f"Cannot parse RPU from cluster name: {cluster_name!r}")


# ---------------------------------------------------------------------------
# Trace cache
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_trace(run_id: str) -> Trace:
    """Build and cache a Trace for the given run_id."""
    return Trace(run_id)


# ---------------------------------------------------------------------------
# Experiment index (run_id → experiment_name mapping)
# ---------------------------------------------------------------------------


def _build_experiment_index() -> dict[str, list[str]]:
    """Scan data/runs/ and group run_ids by experiment_name from
    runner_config.yml.

    Returns a dict mapping experiment_name → [run_id, ...] sorted by run_id.
    """
    base = _runs_dir()
    if not os.path.exists(base):
        return {}

    index: dict[str, list[str]] = defaultdict(list)
    for entry in os.listdir(base):
        entry_path = os.path.join(base, entry)
        if not os.path.isdir(entry_path):
            continue
        config_path = os.path.join(entry_path, "runner_config.yml")
        if not os.path.exists(config_path):
            continue
        # Also require sys_query_history to exist (skip incomplete runs)
        has_history = any(
            f.startswith("sys_query_history+") and f.endswith(".parquet")
            for f in os.listdir(entry_path)
        )
        if not has_history:
            continue
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            experiment_name = (
                cfg.get("basic_config", {}).get("experiment_name")
            )
            if experiment_name:
                index[experiment_name].append(entry)
        except Exception:
            logger.warning("Skipping run %s: cannot read runner_config.yml", entry)
            continue

    # Sort run_ids within each experiment
    for k in index:
        index[k].sort()
    return dict(index)


_experiment_index_cache: dict[str, list[str]] | None = None


def _get_experiment_index(refresh: bool = False) -> dict[str, list[str]]:
    global _experiment_index_cache
    if _experiment_index_cache is None or refresh:
        _experiment_index_cache = _build_experiment_index()
    return _experiment_index_cache


# ---------------------------------------------------------------------------
# Per-run summary computation
# ---------------------------------------------------------------------------


def _compute_run_summary(run_id: str, trace: Trace) -> RunSummary:
    """Build a RunSummary for one runner run."""
    compliance = trace.slo_compliance()
    slo_config = trace.slo_config
    resolver = trace.slo_resolver

    total_queries = trace.num_queries
    completed = compliance[~compliance["is_aborted"]]
    completed_count = len(completed)
    violating = int(compliance["violates_slo"].sum())

    viol_rate = violating / total_queries if total_queries > 0 else 0.0
    avg_viol_amount = (
        round(float(completed["violation_amount_s"].mean()), 4) if completed_count > 0 else 0.0
    )
    avg_viol_relative = (
        round(float(completed["violation_relative"].mean()), 6) if completed_count > 0 else 0.0
    )

    # Cost
    total_cost = 0.0
    for cluster_name in trace.seen_clusters:
        try:
            total_cost += trace.cost_of_cluster(cluster_name)
        except ValueError:
            # RPU cannot be parsed — compute manually
            try:
                rpu = _rpu_for_runner_cluster(cluster_name)
                from autoslo.clusters.cluster import Cluster
                from intervaltree import Interval

                df = trace._dfs["sys_query_history"][cluster_name]
                from autoslo.utils.billing import Billing

                intervals = [
                    Interval(begin=s.timestamp(), end=e.timestamp())
                    for s, e in zip(df["start_time"], df["end_time"])
                ]
                billed_s = Billing.billed_s(query_intervals=intervals)
                total_cost += billed_s * Cluster.cost_per_second_for_rpu(rpu)
            except Exception:
                logger.warning(
                    "Cannot compute cost for cluster %s in run %s",
                    cluster_name,
                    run_id,
                )

    return RunSummary(
        run_id=run_id,
        seed=None,
        slo_s=slo_config.get("slo_s"),
        slo_metric=slo_config.get("slo_metric"),
        slo_threshold=slo_config.get("slo_threshold"),
        slo_dict_filename=resolver.slo_dict_filename,
        slo_dict=resolver.slo_dict or None,
        blueprint_name=None,
        violation_rate=round(viol_rate, 6),
        violation_amount_s=avg_viol_amount,
        violation_relative_mean=avg_viol_relative,
        violating_queries=violating,
        total_queries=total_queries,
        total_cost=round(total_cost, 4),
        num_queries=total_queries,
        completed_at=None,
    )


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@router.get("/runner/experiments", response_model=list[str])
def list_runner_experiments(refresh: bool = False):
    """Return the names of all runner experiments (grouped by
    experiment_name from runner_config.yml)."""
    idx = _get_experiment_index(refresh=refresh)
    return sorted(idx.keys())


@router.get("/runner/experiments/{name}", response_model=ExperimentSummary)
def get_runner_experiment(name: str):
    """Return run summaries for the named runner experiment."""
    idx = _get_experiment_index()
    run_ids = idx.get(name)
    if not run_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Runner experiment '{name}' not found",
        )

    summaries: list[RunSummary] = []
    for run_id in run_ids:
        try:
            trace = _get_trace(run_id)
            summaries.append(_compute_run_summary(run_id, trace))
        except Exception:
            logger.exception("Error building summary for run %s", run_id)
            continue

    return ExperimentSummary(experiment_name=name, runs=summaries)


@router.get(
    "/runner/runs/{experiment}/{run_id}/timeline",
    response_model=TimelineData,
)
def get_runner_timeline(
    experiment: str,
    run_id: str,
):
    """Build and return the Gantt timeline for a runner run using Trace."""
    # Validate the run belongs to this experiment
    idx = _get_experiment_index()
    if experiment not in idx or run_id not in idx[experiment]:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in experiment '{experiment}'",
        )

    trace = _get_trace(run_id)
    slo_config = trace.slo_config

    effective_metric = slo_config.get("slo_metric")
    effective_threshold = slo_config.get("slo_threshold")

    resolver = trace.slo_resolver

    # Compute t0 for normalization
    t0 = trace.earliest_query_start_time

    arrival = trace.arrival_times()
    completion = trace.completion_times()
    routing = trace.routing_decisions
    compliance = trace.slo_compliance(resolver=resolver)

    total_queries = trace.num_queries
    completed_df = compliance[~compliance["is_aborted"]]
    completed_count = len(completed_df)
    violating_queries = int(compliance["violates_slo"].sum())
    violation_rate = violating_queries / total_queries if total_queries > 0 else 0.0
    agg_violation_amount = (
        round(float(completed_df["violation_amount_s"].mean()), 4)
        if completed_count > 0 else 0.0
    )
    agg_violation_relative = (
        round(float(completed_df["violation_relative"].mean()), 6)
        if completed_count > 0 else 0.0
    )

    intervals: list[TimelineInterval] = []
    for qid in trace.query_ids:
        arr = arrival[qid]
        comp = completion[qid]
        start_s = (arr - t0).total_seconds()
        end_s = (comp - t0).total_seconds()
        row = compliance.loc[qid]
        state = "ABORTED" if row["is_aborted"] else "COMPLETED"
        intervals.append(
            TimelineInterval(
                cluster_name=str(routing[qid]),
                query_id=qid,
                query_text_id=row["query_text_id"],
                start_s=round(start_s, 4),
                end_s=round(end_s, 4),
                latency_s=round(float(row["latency_s"]), 4),
                state=state,
                violates_slo=bool(row["violates_slo"]),
                slo_s=float(row["slo_s"]),
                violation_amount_s=round(float(row["violation_amount_s"]), 4),
                violation_relative=round(float(row["violation_relative"]), 6),
            )
        )

    # Total cost
    total_cost = 0.0
    for cluster_name in trace.seen_clusters:
        try:
            total_cost += trace.cost_of_cluster(cluster_name)
        except ValueError:
            try:
                rpu = _rpu_for_runner_cluster(cluster_name)
                from autoslo.clusters.cluster import Cluster
                from intervaltree import Interval
                from autoslo.utils.billing import Billing

                df = trace._dfs["sys_query_history"][cluster_name]
                query_intervals = [
                    Interval(begin=s.timestamp(), end=e.timestamp())
                    for s, e in zip(df["start_time"], df["end_time"])
                ]
                billed_s = Billing.billed_s(query_intervals=query_intervals)
                total_cost += billed_s * Cluster.cost_per_second_for_rpu(rpu)
            except Exception:
                logger.warning(
                    "Cannot compute cost for cluster %s", cluster_name
                )

    return TimelineData(
        run_id=run_id,
        experiment_name=experiment,
        default_slo_s=resolver.default_slo_s,
        slo_metric=effective_metric,
        slo_threshold=effective_threshold,
        slo_dict=resolver.slo_dict,
        slo_dict_filename=resolver.slo_dict_filename,
        total_queries=total_queries,
        violating_queries=violating_queries,
        violation_rate=round(violation_rate, 6),
        violation_amount_s=agg_violation_amount,
        violation_relative_mean=agg_violation_relative,
        total_cost=round(total_cost, 4),
        intervals=intervals,
    )


@router.get(
    "/runner/runs/{experiment}/{run_id}/template_stats",
    response_model=list[TemplateStats],
)
def get_runner_template_stats(
    experiment: str,
    run_id: str,
):
    """Return per-template compliance statistics for a runner run."""
    idx = _get_experiment_index()
    if experiment not in idx or run_id not in idx[experiment]:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in experiment '{experiment}'",
        )

    trace = _get_trace(run_id)
    compliance = trace.slo_compliance()

    # Restrict to completed queries and extract template_id from query_text_id
    completed = compliance[~compliance["is_aborted"]].copy()
    completed["template_id"] = completed["query_text_id"].apply(
        lambda s: int(s.split("#")[1])
        if isinstance(s, str) and "#" in s
        else None
    )

    df = completed.dropna(subset=["template_id"])
    if df.empty:
        return []
    df = df.astype({"template_id": int})
    results: list[TemplateStats] = []

    for tid, group in df.groupby("template_id"):
        lats = group["latency_s"].to_numpy(dtype=float)
        slo_val = float(group["slo_s"].iloc[0])
        n = len(lats)

        results.append(
            TemplateStats(
                template_id=int(tid),
                occurrences=n,
                slo_s=round(slo_val, 4),
                p50_latency_s=round(float(np.percentile(lats, 50)), 4),
                p90_latency_s=round(float(np.percentile(lats, 90)), 4),
                p95_latency_s=round(float(np.percentile(lats, 95)), 4),
                violation_rate=round(int(group["violates_slo"].sum()) / n, 6),
                total_violation_amount_s=round(float(group["violation_amount_s"].sum()), 4),
                mean_relative_violation=round(float(group["violation_relative"].mean()), 6),
            )
        )

    results.sort(key=lambda s: s.template_id)
    return results
