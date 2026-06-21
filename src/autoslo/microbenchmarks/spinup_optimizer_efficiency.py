from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np
import pandas as pd

from autoslo.config.component_configs import (
    SloObjectiveConfig,
    SloResolverConfig,
    SpinupOptimizerConfig,
)
from autoslo.microbenchmarks.microbenchmark_runner import MicrobenchmarkRunner
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.spinup_optimizer import find_next_spinup_time_df
from autoslo.visualizations.colors import Palette


class SpinupOptimizerEfficiencyBenchmark(MicrobenchmarkRunner):
    @classmethod
    def name(cls) -> str:
        return "spinup_optimizer_efficiency"

    @classmethod
    def required_keys(cls) -> list[str]:
        return [
            "workload_seed",
            "reps",
            "n_scenarios_values",
            "queries_per_scenario_values",
            "latency_multiplier",
            "slo_resolver_config",
            "slo_objective_config",
            "spinup_optimizer_config",
        ]

    @classmethod
    def run_from_manifest(cls, manifest: dict[str, Any]) -> None:

        # Setup.
        workload_seed = int(manifest["workload_seed"])
        reps = int(manifest["reps"])

        n_scenarios_values = [int(v) for v in manifest["n_scenarios_values"]]
        queries_per_scenario_values = [
            int(v) for v in manifest["queries_per_scenario_values"]
        ]

        latency_multiplier = float(manifest["latency_multiplier"])

        slo_resolver = SloResolver(SloResolverConfig.from_config(manifest))
        slo_objective = SloObjective(SloObjectiveConfig.from_config(manifest))
        spinup_optimizer_config = SpinupOptimizerConfig.from_config(manifest)

        # Run.
        rows: list[dict[str, Any]] = []
        with cls.make_progress() as progress:
            total_steps = (
                len(n_scenarios_values)
                * len(queries_per_scenario_values)
                * reps
            )
            task = progress.add_task(
                "Spinup optimizer benchmark", total=total_steps
            )
            for (
                n_scenarios,
                queries_per_scenario,
            ) in itertools.product(
                n_scenarios_values,
                queries_per_scenario_values,
            ):
                min_delinquent_workloads = max(
                    1,
                    math.ceil(
                        spinup_optimizer_config.min_delinquent_workload_fraction
                        * n_scenarios
                    ),
                )
                # Unique seed per (n_scenarios, queries)
                rng = np.random.default_rng(
                    workload_seed
                    + n_scenarios * 100000
                    + queries_per_scenario * 1000
                )

                # Generate synthetic completion logs for all scenarios.
                completion_logs = []
                for s in range(n_scenarios):
                    inter_arrivals = rng.exponential(
                        5, size=queries_per_scenario
                    )
                    arrival_s = np.cumsum(inter_arrivals)
                    latency_s = slo_resolver.default_slo_s * latency_multiplier
                    completion_s = arrival_s + latency_s
                    completion_logs.append(
                        pd.DataFrame(
                            {
                                "query_id": [
                                    f"s{s}_q{i}"
                                    for i in range(queries_per_scenario)
                                ],
                                "query_text_id": "ext_tpcds1000#001#001",
                                "arrival_s": arrival_s,
                                "completion_s": completion_s,
                                "latency_s": np.full(
                                    queries_per_scenario, latency_s
                                ),
                            }
                        )
                    )
                for rep in range(reps):

                    t0 = time.perf_counter()
                    candidates = find_next_spinup_time_df(
                        completion_structured_logs=completion_logs,
                        slo_resolver=slo_resolver,
                        slo_objective=slo_objective,
                        min_delinquent_workloads=min_delinquent_workloads,
                        lead_time_s=spinup_optimizer_config.lead_time_s,
                        min_candidate_spacing_s=spinup_optimizer_config.min_candidate_spacing_s,
                        verbose=False,
                    )
                    elapsed_s = time.perf_counter() - t0

                    rows.append(
                        {
                            "n_scenarios": n_scenarios,
                            "queries_per_scenario": queries_per_scenario,
                            "total_simulated_queries": n_scenarios
                            * queries_per_scenario,
                            "n_candidates_found": len(candidates),
                            "rep": rep,
                            "elapsed_s": elapsed_s,
                        }
                    )
                    progress.update(task, advance=1)

        df = pd.DataFrame(rows)
        df.to_csv(cls.csv_path(), index=False)

    @classmethod
    def plot(cls) -> None:

        cls.microbenchmark_scatter_plot(
            x_col="total_simulated_queries",
            y_col="elapsed_s",
            shape_col="n_scenarios",
            color_col="queries_per_scenario",
            shape_legend_title="Forecasts",
            colorbar_label="Queries / Forecast",
            cmap_colors=[
                Palette.light_gray,
                Palette.light_purple,
                Palette.light_purple_sat,
            ],
            log_x=True,
            log_color_base=2,
            log_y=True,
        )
