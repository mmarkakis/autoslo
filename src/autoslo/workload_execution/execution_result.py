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


def _compute_cost_live_run(
    execution_dir: Path,
    start_time_cutoff: Optional["pd.Timestamp"] = None,
) -> float:
    """Sum billed cost across clusters from sys_serverless_usage+*.parquet files.

    Parameters
    ----------
    execution_dir :
        Path to the run output directory.
    start_time_cutoff :
        If provided, only rows whose ``start_time`` is at or after this
        timestamp contribute to the cost.  Rows that predate the cutoff are
        assumed to belong to a warm-up period outside the measured window.

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
    read_cols = [
        "charged_seconds",
        "charged_extra_compute_for_automatic_optimization_seconds",
    ]
    if start_time_cutoff is not None:
        read_cols = ["start_time"] + read_cols
    total_cost = 0.0
    for path in usage_files:
        df = pd.read_parquet(path, columns=read_cols)
        if start_time_cutoff is not None:
            # Align timezone-awareness between the column and the cutoff so
            # the comparison is always valid.
            cutoff = start_time_cutoff
            if (
                pd.api.types.is_datetime64_any_dtype(df["start_time"])
                and df["start_time"].dt.tz is not None
            ):
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize("UTC")
            else:
                if cutoff.tzinfo is not None:
                    cutoff = cutoff.replace(tzinfo=None)
            df = df[df["start_time"] >= cutoff]
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
    min_cluster_index: Optional[int] = None

    def violation_for_metric(self, metric: SloMetric) -> float:
        """Return the violation for the given metric."""
        if metric == SloMetric.BINARY:
            return self.violation_rate
        elif metric == SloMetric.ABSOLUTE_S:
            return self.violation_amount_s
        elif metric in (SloMetric.RELATIVE, SloMetric.RELATIVE_UNCONSTRAINED):
            return self.violation_relative_mean
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @staticmethod
    def load(
        execution_dir: Path,
        slo_resolver: Optional[SloResolver] = None,
        tail_fraction: float = 1.0,
        min_cluster_index: Optional[int] = None,
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
            Mutually exclusive with *min_cluster_index*.
        min_cluster_index :
            If set, only include queries that arrived at or after the time
            the cluster with this counter index became ready.  Prints a warning
            if no matching cluster is found in the structured
            log.  Mutually exclusive with *tail_fraction*.
        """
        if tail_fraction != 1.0 and min_cluster_index is not None:
            raise ValueError(
                "At most one of 'tail_fraction' and 'min_cluster_index' may be "
                "specified at a time."
            )

        # -- SLO resolver --
        if slo_resolver is None:
            config_path = execution_dir / "execution_config.yml"
            with open(config_path) as f:
                config: dict[str, Any] = yaml.safe_load(f)
            slo_resolver = SloResolver(SloResolverConfig.from_config(config))

        # -- violations --
        violation_rate = 0.0
        violation_amount_s = 0.0
        violation_relative_mean = 0.0
        num_queries = 0
        total_rel_time_s = 0

        slog: Optional[StructuredLog] = None
        latencies_df = pd.DataFrame(
            columns=[
                "query_id",
                "query_text_id",
                "arrival_s",
                "completion_s",
                "latency_s",
            ]
        )

        log_path = execution_dir / "structured_log.parquet"
        if log_path.exists():
            slog = StructuredLog.load(log_path)
            latencies_df = slog.query_latencies(drop_incomplete=True)
            if min_cluster_index is not None and not latencies_df.empty:
                ready_times = slog.cluster_ready_times()
                matches = {
                    name: t
                    for name, t in ready_times.items()
                    if Cluster.counter_for_cluster_name(name)
                    == min_cluster_index
                }
                if not matches:
                    print(
                        f"No cluster with index {min_cluster_index} found in "
                        f"structured log at {log_path}."
                    )
                else:
                    cutoff_s = min(matches.values())
                    latencies_df = latencies_df[
                        latencies_df["arrival_s"] >= cutoff_s
                    ].reset_index(drop=True)
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

        # -- Detect run type and compute cost --
        billing_path = execution_dir / "billing_interval_analysis.yml"
        is_live = not billing_path.exists()
        total_cost = 0.0
        if not is_live:
            billing: dict[str, Any] = load_yaml(billing_path)
            for cluster_data in billing.values():
                total_cost += cluster_data.get("total_billed_cost", 0.0)
        else:
            start_time_cutoff: Optional[pd.Timestamp] = None
            if (
                slog is not None
                and not latencies_df.empty
                and (tail_fraction < 1.0 or min_cluster_index is not None)
            ):
                query_window_start_s = float(latencies_df["arrival_s"].min())
                # Derive absolute run-start epoch from any event's wall_clock_s
                # and rel_time_s, then shift by the relative window start.
                ref_df = slog.df[["wall_clock_s", "rel_time_s"]].dropna()
                if not ref_df.empty:
                    ref = ref_df.iloc[0]
                    run_epoch_start_s = float(ref["wall_clock_s"]) - float(
                        ref["rel_time_s"]
                    )
                    start_time_cutoff = pd.Timestamp(
                        run_epoch_start_s + query_window_start_s, unit="s"
                    )
            total_cost = _compute_cost_live_run(
                execution_dir, start_time_cutoff
            )

        # For live runs: print a warning if there were aborted queries.
        if is_live:
            num_aborted_queries = Trace(execution_dir.name).was_aborted().sum()
            if num_aborted_queries > 0:
                print(
                    f"Warning: detected {num_aborted_queries} aborted queries "
                    f"in live run {execution_dir}. These queries may indicate "
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
            min_cluster_index=min_cluster_index,
        )
