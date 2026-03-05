"""
simulator_router.py
-------------------
FastAPI router for the simulator experiments/runs API.

Endpoints
---------
GET /api/simulator/experiments
    List experiment names (directories containing experiment_meta.json).

GET /api/simulator/experiments/{name}
    Return the full experiment_meta.json for the named experiment.

GET /api/simulator/runs/{experiment}/{run_id}/timeline
    Return a TimelineData response built from the run's solve_log.parquet.

GET /api/simulator/runs/{experiment}/{run_id}/log
    Return the raw solve log as a JSON array (enables future scrubber).

GET /api/simulator/runs/{experiment}/{run_id}/config
    Return the run's config.yml as a dict.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.slo_resolver import SloResolver

router = APIRouter()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _simulator_runs_dir() -> str:
    return os.path.join(pu.get_data_path(), "simulator_runs")


def _experiment_dir(experiment: str) -> str:
    return os.path.join(_simulator_runs_dir(), experiment)


def _run_dir(experiment: str, run_id: str) -> str:
    return os.path.join(_experiment_dir(experiment), run_id)


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{label} not found")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    run_id: str
    seed: int | None = None
    slo_s: float | None = None
    slo_dict_filename: str | None = None
    slo_dict: dict | None = None
    blueprint_name: str | None = None
    violation_rate: float | None = None
    violating_queries: int | None = None
    total_queries: int | None = None
    total_cost: float | None = None
    violation_amount_s: float | None = None
    num_queries: int | None = None
    completed_at: str | None = None


class ExperimentSummary(BaseModel):
    experiment_name: str
    runs: list[RunSummary]


class TimelineInterval(BaseModel):
    cluster_name: str
    query_id: Any
    query_text_id: Any
    start_s: float
    end_s: float
    state: str  # "COMPLETED" | "RUNNING"
    violates_slo: bool
    slo_s: float


class TimelineData(BaseModel):
    run_id: str
    experiment_name: str | None
    default_slo_s: float
    slo_dict: dict
    slo_dict_filename: str | None
    total_queries: int
    violating_queries: int
    violation_rate: float
    total_cost: float
    intervals: list[TimelineInterval]


class TemplateStats(BaseModel):
    template_id: int
    slo_s: float
    occurrences: int
    compliant: int
    compliance_rate: float
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    avg_excess_s: float


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@router.get("/simulator/experiments", response_model=list[str])
def list_experiments():
    """
    Return the names of all experiment directories that contain an
    experiment_meta.json file.  Flat-layout run directories (without
    experiment_meta.json) are intentionally ignored.
    """
    base = _simulator_runs_dir()
    if not os.path.exists(base):
        return []
    names = []
    for entry in sorted(os.listdir(base)):
        meta_path = os.path.join(base, entry, "experiment_meta.json")
        if os.path.isdir(os.path.join(base, entry)) and os.path.exists(meta_path):
            names.append(entry)
    return names


@router.get("/simulator/experiments/{name}", response_model=ExperimentSummary)
def get_experiment(name: str):
    """
    Return the full experiment summary (metadata + per-run stats) for the
    named experiment.
    """
    meta_path = os.path.join(_experiment_dir(name), "experiment_meta.json")
    _require_file(meta_path, f"Experiment '{name}'")
    with open(meta_path) as f:
        meta = json.load(f)
    return ExperimentSummary(
        experiment_name=meta.get("experiment_name", name),
        runs=[RunSummary(**r) for r in meta.get("runs", [])],
    )


@router.get(
    "/simulator/runs/{experiment}/{run_id}/timeline",
    response_model=TimelineData,
)
def get_run_timeline(experiment: str, run_id: str):
    """
    Build and return the final-state Gantt timeline for the specified run.
    The response is a flat list of interval records ready for browser-side
    rendering — no pre-computed Plotly shapes are included.
    """
    rdir = _run_dir(experiment, run_id)
    log_path = os.path.join(rdir, "solve_log.parquet")
    config_path = os.path.join(rdir, "config.yml")
    billing_path = os.path.join(rdir, "billing_interval_analysis.yml")

    _require_file(log_path, f"solve_log for run '{run_id}'")
    _require_file(config_path, f"config for run '{run_id}'")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    resolver = SloResolver.from_dict(
        default_slo_s=config.get("slo_s", 0.0),
        slo_dict=config.get("slo_dict") or {},
        slo_dict_filename=config.get("slo_dict_filename"),
    )

    # --- reconstruct timeline from log ---
    log = pd.read_parquet(log_path)

    routing = (
        log[log["event_type"] == "routing"]
        .set_index("query_id")[["timestamp", "cluster_name", "end_time_s", "query_text_id"]]
    )
    completions = (
        log[log["event_type"] == "completion"]
        .set_index("query_id")[["end_time_s"]]
        .rename(columns={"end_time_s": "completed_end_time_s"})
    )
    updates = log[log["event_type"] == "latency_update"]
    if not updates.empty:
        last_updates = (
            updates.sort_values("timestamp")
            .groupby("query_id")["end_time_s"]
            .last()
            .rename("updated_end_time_s")
        )
    else:
        last_updates = pd.Series(dtype=float, name="updated_end_time_s")

    df = routing.join(completions, how="left").join(last_updates, how="left")

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

    total_queries = len(df)
    violating_queries = 0
    intervals: list[TimelineInterval] = []

    for qid, row in df.iterrows():
        start_s = float(row["timestamp"])
        end_s = float(row["final_end_s"])
        duration = end_s - start_s
        state = str(row["state"])
        tpcds = row.get("query_text_id")
        row_slo_s = resolver.resolve(tpcds)
        viol = state == "COMPLETED" and duration > row_slo_s
        if viol:
            violating_queries += 1
        intervals.append(
            TimelineInterval(
                cluster_name=str(row["cluster_name"]),
                query_id=qid,
                query_text_id=tpcds,
                start_s=start_s,
                end_s=end_s,
                state=state,
                violates_slo=viol,
                slo_s=row_slo_s,
            )
        )

    violation_rate = violating_queries / total_queries if total_queries > 0 else 0.0

    # total cost from billing file
    total_cost = 0.0
    if os.path.exists(billing_path):
        with open(billing_path) as f:
            billing = yaml.safe_load(f) or {}
        for cluster_data in billing.values():
            total_cost += cluster_data.get("total_billed_cost", 0.0)

    return TimelineData(
        run_id=run_id,
        experiment_name=experiment,
        default_slo_s=resolver.default_slo_s,
        slo_dict=resolver.slo_dict,
        slo_dict_filename=resolver.slo_dict_filename,
        total_queries=total_queries,
        violating_queries=violating_queries,
        violation_rate=violation_rate,
        total_cost=total_cost,
        intervals=intervals,
    )


@router.get("/simulator/runs/{experiment}/{run_id}/log")
def get_run_log(experiment: str, run_id: str) -> list[dict]:
    """
    Return the raw solve log as a JSON array.  Each element is one log event.
    This endpoint is provided to enable a future browser-side scrubber without
    requiring any further backend changes.
    """
    log_path = os.path.join(_run_dir(experiment, run_id), "solve_log.parquet")
    _require_file(log_path, f"solve_log for run '{run_id}'")
    df = pd.read_parquet(log_path)
    return df.where(pd.notna(df), None).to_dict(orient="records")


@router.get("/simulator/runs/{experiment}/{run_id}/config")
def get_run_config(experiment: str, run_id: str) -> dict:
    """
    Return the run's config.yml as a plain dict.
    """
    config_path = os.path.join(_run_dir(experiment, run_id), "config.yml")
    _require_file(config_path, f"config for run '{run_id}'")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


@router.get(
    "/simulator/runs/{experiment}/{run_id}/template_stats",
    response_model=list[TemplateStats],
)
def get_run_template_stats(experiment: str, run_id: str) -> list[TemplateStats]:
    """
    Return per-template compliance statistics for the specified run.

    For each template ID seen in the completion events of the solve log, returns:
    - occurrences       : number of completed queries for that template
    - compliant         : queries whose latency ≤ per-template SLO
    - compliance_rate   : compliant / occurrences
    - avg/p50/p95 latency
    - avg_excess_s      : average(max(0, latency - slo_s)) across all queries
    """
    import numpy as np

    rdir = _run_dir(experiment, run_id)
    log_path = os.path.join(rdir, "solve_log.parquet")
    config_path = os.path.join(rdir, "config.yml")

    _require_file(log_path, f"solve_log for run '{run_id}'")
    _require_file(config_path, f"config for run '{run_id}'")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    resolver = SloResolver.from_dict(
        default_slo_s=config.get("slo_s", 0.0),
        slo_dict=config.get("slo_dict") or {},
        slo_dict_filename=config.get("slo_dict_filename"),
    )

    log = pd.read_parquet(log_path)
    completions = log[log["event_type"] == "completion"].copy()

    # Join with routing to get query_text_id (completions log lacks it)
    routing = (
        log[log["event_type"] == "routing"]
        .set_index("query_id")[["query_text_id", "latency_s"]]
    )
    # Completions have query_id in the index after set_index; merge on query_id
    completions = completions.set_index("query_id")
    # Use latency_s from completions when available, fall back to routing
    merged = completions[["latency_s"]].join(
        routing[["query_text_id"]], how="left"
    )
    merged["latency_s"] = merged["latency_s"].fillna(0.0)

    # Extract template IDs
    from autoslo.workload_definition.query import Query

    merged["template_id"] = merged["query_text_id"].map(
        lambda x: Query.template_id(x) if pd.notna(x) and x else -1
    )
    merged["slo_s_row"] = merged["query_text_id"].map(resolver.resolve)

    results: list[TemplateStats] = []
    for tid, group in merged.groupby("template_id"):
        if tid == -1:
            continue
        lats = group["latency_s"].to_numpy(dtype=float)
        slo_val = float(group["slo_s_row"].iloc[0])
        n = len(lats)
        compliant = int((lats <= slo_val).sum())
        excess = np.maximum(0.0, lats - slo_val)
        results.append(
            TemplateStats(
                template_id=int(tid),  # type: ignore[arg-type]
                slo_s=slo_val,
                occurrences=n,
                compliant=compliant,
                compliance_rate=round(compliant / n, 6) if n > 0 else 0.0,
                avg_latency_s=round(float(np.mean(lats)), 4),
                p50_latency_s=round(float(np.percentile(lats, 50)), 4),
                p95_latency_s=round(float(np.percentile(lats, 95)), 4),
                avg_excess_s=round(float(np.mean(excess)), 4),
            )
        )

    results.sort(key=lambda s: s.template_id)
    return results
