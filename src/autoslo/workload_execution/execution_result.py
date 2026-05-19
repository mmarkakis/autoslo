from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from autoslo.clusters.cluster import Cluster
from autoslo.config.component_configs import SloResolverConfig
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.filesystem.yaml_helpers import load_yaml
from autoslo.slo.slo_metric import LatencySlo, SloMetric
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_execution.trace import Trace


def _compute_cost_live_run(execution_dir: Path) -> float:
    """Sum billed cost across clusters from sys_serverless_usage+*.parquet files.

    Raises
    ------
    FileNotFoundError
        If no ``sys_serverless_usage+*.parquet`` files exist in *execution_dir*.
        This indicates that ``RunStatsCollector`` has not yet completed; callers
        should only invoke :meth:`ExecutionResult.load` after stats collection
        is done.
    """
    usage_files = list(execution_dir.glob("sys_serverless_usage+*.parquet"))
    if not usage_files:
        raise FileNotFoundError(
            f"No sys_serverless_usage+*.parquet files found in {execution_dir}. "
            "Run RunStatsCollector before loading an ExecutionResult for a live run."
        )
    total_cost = 0.0
    for path in usage_files:
        df = pd.read_parquet(
            path,
            columns=[
                "charged_seconds",
                "charged_extra_compute_for_automatic_optimization_seconds",
            ],
        )
        charged = df["charged_seconds"].sum()
        charged += df[
            "charged_extra_compute_for_automatic_optimization_seconds"
        ].sum()
        total_cost += charged / 3600 * Cluster.US_EAST_1_COST_PER_RPU_HOUR
    return total_cost


@dataclass(frozen=True)
class ExecutionResult:
    """Metrics from a single execution (simulation or live run)."""

    execution_dir: Path
    violation_rate: float
    violation_amount_s: float
    violation_relative_mean: float
    total_cost: float
    num_queries: int
    total_rel_time_s: float
    tail_fraction: float

    @staticmethod
    def load(
        execution_dir: str | Path,
        slo_resolver: Optional[SloResolver] = None,
        tail_fraction: float = 1.0,
    ) -> ExecutionResult:
        """
        Load an :class:`ExecutionResult` from a run output directory.

        Works for both simulation directories (``data/simulator_runs/<id>/``)
        and live run directories (``data/runs/<id>/``).

        Parameters
        ----------
        execution_dir :
            Path to the run output directory.
        slo_resolver :
            Optional pre-built resolver.  When omitted, one is constructed
            from ``execution_config.yml`` in *execution_dir*.
        tail_fraction :
            Violation metrics are computed only over the last
            ``tail_fraction`` fraction of queries (ordered by arrival time).
            Must be in (0, 1].  Default uses all queries.
        """
        execution_dir = Path(execution_dir)
        is_live = False

        # -- SLO resolver --
        if slo_resolver is None:
            config_path = execution_dir / "execution_config.yml"
            with open(config_path) as f:
                config: dict[str, Any] = yaml.safe_load(f)
            slo_resolver = SloResolver(SloResolverConfig.from_config(config))

        # -- cost --
        total_cost = 0.0
        billing_path = execution_dir / "billing_interval_analysis.yml"
        if billing_path.exists():
            # This is a simulation.
            billing: dict[str, Any] = load_yaml(billing_path)
            for cluster_data in billing.values():
                total_cost += cluster_data.get("total_billed_cost", 0.0)
        else:
            # This is a live run.
            is_live = True
            total_cost = _compute_cost_live_run(execution_dir)

        # -- violations --
        violation_rate = 0.0
        violation_amount_s = 0.0
        violation_relative_mean = 0.0
        num_queries = 0
        total_rel_time_s = 0

        log_path = execution_dir / "structured_log.parquet"
        if log_path.exists():
            latencies_df = StructuredLog.load(log_path).query_latencies(
                drop_incomplete=True
            )
            if tail_fraction < 1.0 and not latencies_df.empty:
                n = max(1, math.ceil(len(latencies_df) * tail_fraction))
                latencies_df = latencies_df.sort_values("arrival_s").iloc[-n:]
            if not latencies_df.empty:
                num_queries = len(latencies_df)
                per_row_slo = (
                    latencies_df["query_text_id"]
                    .map(slo_resolver.resolve)
                    .fillna(0.0)
                )
                lat_and_slos = [
                    LatencySlo(lat, slo)
                    for lat, slo in zip(latencies_df["latency_s"], per_row_slo)
                ]
                violation_rate = SloMetric.BINARY.aggregate_batch(lat_and_slos)
                violation_amount_s = SloMetric.ABSOLUTE_S.aggregate_batch(
                    lat_and_slos
                )
                violation_relative_mean = SloMetric.RELATIVE.aggregate_batch(
                    lat_and_slos
                )
            total_rel_time_s = latencies_df["completion_s"].max()

        # For live runs: print a warning if there were aborted queries.
        if is_live:
            num_aborted_queries = Trace(execution_dir.name).was_aborted().sum()
            if num_aborted_queries > 0:
                print(
                    f"Warning: detected {num_aborted_queries} aborted queries "
                    f"in live run {execution_dir}. These queries are not "
                    "included in the violation metrics, but may indicate "
                    "instability."
                )

        return ExecutionResult(
            execution_dir=execution_dir,
            violation_rate=violation_rate,
            violation_amount_s=violation_amount_s,
            violation_relative_mean=violation_relative_mean,
            total_cost=total_cost,
            num_queries=num_queries,
            total_rel_time_s=total_rel_time_s,
            tail_fraction=tail_fraction,
        )
