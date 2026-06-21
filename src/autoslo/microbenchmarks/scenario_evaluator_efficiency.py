from __future__ import annotations

import shutil
import time
from typing import Any

import pandas as pd
from rich.console import Console

from autoslo.config.component_configs import AutoscalerConfig, WorkloadConfig
from autoslo.microbenchmarks.microbenchmark_runner import MicrobenchmarkRunner
from autoslo.tuner.scenario_evaluator import ScenarioEvaluator
from autoslo.visualizations.colors import Palette
from autoslo.workload_definition.poisson_workload_creator import (
    PoissonWorkloadCreator,
)


class ScenarioEvaluatorEfficiencyBenchmark(MicrobenchmarkRunner):
    @classmethod
    def name(cls) -> str:
        return "scenario_evaluator_efficiency"

    @classmethod
    def required_keys(cls) -> list[str]:
        return [
            "workload_seed",
            "reps",
            "n_parallel_values",
            "n_queries_per_simulation_values",
            "initial_rpus",
            "slo_resolver_config",
            "slo_objective_config",
            "query_router_config",
        ]

    @classmethod
    def run_from_manifest(cls, manifest: dict[str, Any]) -> None:

        n_parallel_values = [int(v) for v in manifest["n_parallel_values"]]
        n_queries_values = [
            int(v) for v in manifest["n_queries_per_simulation_values"]
        ]
        initial_rpus = [int(v) for v in manifest["initial_rpus"]]
        reps = int(manifest["reps"])

        evaluator = ScenarioEvaluator()
        console = Console()
        rows: list[dict[str, Any]] = []
        total_steps = len(n_parallel_values) * len(n_queries_values) * reps
        step = 0

        for n_queries in n_queries_values:

            # Set up workload and configs.
            workload = (
                PoissonWorkloadCreator.create_poisson_workload_with_n_queries(
                    num_templates=99,
                    num_query_texts_per_template=1,
                    num_total_queries=n_queries,
                    poisson_lambda=0.2,
                    seed=int(manifest["workload_seed"]),
                    print_summary=False,
                )
            )
            workload_config = WorkloadConfig(
                workload_name=workload.workload_name
            )
            base_config = {
                "workload_config": workload_config.to_dict(),
                "slo_resolver_config": manifest["slo_resolver_config"],
                "slo_objective_config": manifest["slo_objective_config"],
                "provisioner_config": cls.DEFAULT_PROVISIONER_CONFIG_ARGS,
                "managed_cluster_pool_config": {
                    "initial_rpus": initial_rpus,
                },
                "scheduled_spinups": [],
                "query_router_config": manifest["query_router_config"],
                "autoscaler_config": AutoscalerConfig().to_dict(),
            }

            for n_parallel in n_parallel_values:
                grid_point_sim_dir = (
                    cls.scratch_dir() / "sims" / f"np{n_parallel}_nq{n_queries}"
                )

                for rep in range(reps):
                    step += 1
                    console.print(
                        f"[{step}/{total_steps}] "
                        f"n_parallel={n_parallel} n_queries={n_queries} rep={rep}"
                    )
                    trial_out_dir = grid_point_sim_dir / f"rep{rep}"
                    t0 = time.perf_counter()
                    evaluator.evaluate_batch_from_configs(
                        progress_bar_label=(
                            f"np{n_parallel}_nq{n_queries}_rep{rep}"
                        ),
                        out_dir=trial_out_dir,
                        workload_configs=[workload_config],
                        configs=[base_config] * n_parallel,
                        verbose_progress=False,
                    )
                    elapsed_s = time.perf_counter() - t0
                    rows.append(
                        {
                            "n_parallel": n_parallel,
                            "n_queries_per_simulation": n_queries,
                            "total_simulated_queries": n_parallel * n_queries,
                            "rep": rep,
                            "elapsed_s": elapsed_s,
                        }
                    )

                if grid_point_sim_dir.exists():
                    shutil.rmtree(grid_point_sim_dir)

        df = pd.DataFrame(rows)
        df.to_csv(cls.csv_path(), index=False)

    @classmethod
    def plot(cls) -> None:
        cls.microbenchmark_scatter_plot(
            x_col="total_simulated_queries",
            y_col="elapsed_s",
            shape_col="n_parallel",
            color_col="n_queries_per_simulation",
            shape_legend_title="Parallel\nSimulations",
            colorbar_label="Queries / Simulation",
            cmap_colors=[
                Palette.light_gray,
                Palette.light_red,
                Palette.light_red_sat,
            ],
            log_x=True,
            log_y=True,
            log_color_base=2,
        )
