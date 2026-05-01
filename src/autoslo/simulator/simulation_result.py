from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from autoslo.config.component_configs import SloResolverConfig
from autoslo.slo.slo_metric import LatencySlo, SloMetric
from autoslo.slo.slo_resolver import SloResolver


@dataclass(frozen=True)
class SimulationResult:
    """Metrics from a single simulation."""

    simulation_dir: Path
    violation_rate: float
    violation_amount_s: float
    violation_relative_mean: float
    total_cost: float
    num_queries: int

    @staticmethod
    def load(
        simulation_dir: str | Path,
    ) -> SimulationResult:
        """Load a SimulationResult from the output directory of a single
        simulation.

        Reads ``billing_interval_analysis.yml`` for cost and
        ``structured_log.parquet`` for violation statistics — the same logic
        used by :meth:`WorkloadSimulator._write_experiment_meta`.
        """
        simulation_dir = Path(simulation_dir)

        # -- build slo resolver for this scenario from its execution_config.yml --
        config_path = simulation_dir / "execution_config.yml"
        config: dict[str, Any] = {}
        with open(config_path) as f:
            config = yaml.safe_load(f)
        slo_resolver = SloResolver(SloResolverConfig.from_config(config))

        # -- cost --
        total_cost = 0.0
        billing_path = simulation_dir / "billing_interval_analysis.yml"
        if billing_path.exists():
            with open(billing_path) as f:
                billing: dict[str, Any] = yaml.safe_load(f) or {}
            for cluster_data in billing.values():
                total_cost += cluster_data.get("total_billed_cost", 0.0)

        # -- violations --
        violation_rate = 0.0
        violation_amount_s = 0.0
        violation_relative_mean = 0.0
        num_queries = 0

        log_path = simulation_dir / "structured_log.parquet"
        if log_path.exists():
            log = pd.read_parquet(
                log_path,
                columns=[
                    "rel_time_s",
                    "event_type",
                    "query_id",
                    "query_text_id",
                ],
            )
            log = log[log["event_type"].isin({"arrival", "completion"})]
            if not log.empty:

                pivoted = log.pivot(
                    index=["query_id", "query_text_id"],
                    columns="event_type",
                    values="rel_time_s",
                )
                latencies = (
                    pivoted["completion"] - pivoted["arrival"]
                ).tolist()
                per_row_slo = (
                    pivoted.index.get_level_values("query_text_id")
                    .map(slo_resolver.resolve)
                    .fillna(0.0)
                )

                ## TODO: Deal with failed queries. Not super needed here because
                ## in the simulator all queries succeed, but needed in principle.
                lat_and_slos = [
                    LatencySlo(lat, slo)
                    for lat, slo in zip(latencies, per_row_slo)
                ]
                violation_rate = SloMetric.BINARY.aggregate_batch(lat_and_slos)
                violation_amount_s = SloMetric.ABSOLUTE_S.aggregate_batch(
                    lat_and_slos
                )
                violation_relative_mean = SloMetric.RELATIVE.aggregate_batch(
                    lat_and_slos
                )

        return SimulationResult(
            simulation_dir=simulation_dir,
            violation_rate=violation_rate,
            violation_amount_s=violation_amount_s,
            violation_relative_mean=violation_relative_mean,
            total_cost=total_cost,
            num_queries=num_queries,
        )

    @staticmethod
    def load_batch(
        batch_dir: str | Path,
    ) -> list[SimulationResult]:
        """Load all simulation results from the given directory."""
        batch_dir = Path(batch_dir)
        results: list[SimulationResult] = []
        for simulation_dir in batch_dir.iterdir():
            if simulation_dir.is_dir():
                result = SimulationResult.load(
                    simulation_dir=simulation_dir,
                )
                results.append(result)
        return results
