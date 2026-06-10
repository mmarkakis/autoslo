from __future__ import annotations

import time
from dataclasses import replace

import pandas as pd

from autoslo.clusters.autoscaler import Autoscaler
from autoslo.config.component_configs import (
    AutoscalerConfig,
    ProvisionerConfig,
    QueryRouterConfig,
    SloObjectiveConfig,
    SloResolverConfig,
)
from autoslo.microbenchmarks.microbenchmark_runner import MicrobenchmarkRunner
from autoslo.routing.query_router import QueryRouter
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.visualizations.colors import Palette
from autoslo.workload_definition.poisson_workload_creator import (
    PoissonWorkloadCreator,
)


class AutoscalingEfficiencyBenchmark(MicrobenchmarkRunner):
    @classmethod
    def name(cls) -> str:
        return "autoscaling_efficiency"

    @classmethod
    def required_keys(cls) -> list[str]:
        return [
            "workload_seed",
            "reps",
            "candidate_rpu_values",
            "arrival_rate_qps_values",
            "initial_rpus",
            "slo_resolver_config",
            "slo_objective_config",
            "query_router_config",
            "autoscaler_config",
        ]

    @classmethod
    def run_from_manifest(cls, manifest: dict) -> None:

        # Parse manifest parameters.
        candidate_rpu_values = [
            int(v) for v in manifest["candidate_rpu_values"]
        ]
        arrival_rates = [float(v) for v in manifest["arrival_rate_qps_values"]]
        for rate in arrival_rates:
            if rate <= 0:
                raise ValueError(
                    "arrival_rate_qps_values must contain only "
                    "positive values."
                )
        reps = int(manifest["reps"])

        # Setup router and autoscaler.
        slo_resolver = SloResolver(SloResolverConfig.from_config(manifest))
        slo_objective = SloObjective(SloObjectiveConfig.from_config(manifest))
        query_router_config = QueryRouterConfig.from_config(manifest)
        base_autoscaler_config = AutoscalerConfig.from_config(manifest)
        router = QueryRouter(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            query_router_config=query_router_config,
            out_dir=cls.scratch_dir(),
        )
        cache_state_dim = router.iconq_model.iconq_query_featurizer.num_tables
        provisioner_config = ProvisionerConfig(
            **cls.DEFAULT_PROVISIONER_CONFIG_ARGS
            | {"cluster_cache_state_dim": cache_state_dim},
        )

        # Run.
        rows: list[dict[str, float | int]] = []
        with cls.make_progress() as progress:
            total_steps = len(candidate_rpu_values) * len(arrival_rates) * reps
            task = progress.add_task("Autoscaling benchmark", total=total_steps)

            for arrival_rate_qps in arrival_rates:

                # Set up workload with the given arrival rate.
                total_queries_needed = 2 * int(
                    arrival_rate_qps
                    * base_autoscaler_config.observation_window_s
                )  # Multiply by 2 as a safety buffer.
                workload = PoissonWorkloadCreator.create_poisson_workload_with_n_queries(
                    num_templates=99,
                    num_query_texts_per_template=1,
                    num_total_queries=total_queries_needed,
                    poisson_lambda=arrival_rate_qps,
                    seed=int(manifest["workload_seed"]),
                    print_summary=False,
                )
                workload.populate_featurizations_and_isolated_predictions(
                    iconq_model=router.iconq_model,
                    allowed_rpu_sizes=candidate_rpu_values,
                )
                idx_of_first_non_ingested_query = 0
                for query in workload.queries():
                    if (
                        query.rel_start_time_s
                        < base_autoscaler_config.observation_window_s
                    ):
                        idx_of_first_non_ingested_query += 1
                    else:
                        break
                next_time_s = workload.queries()[
                    idx_of_first_non_ingested_query
                ].rel_start_time_s

                for last_candidate_rpu_idx in range(len(candidate_rpu_values)):
                    candidate_rpu = candidate_rpu_values[
                        :last_candidate_rpu_idx + 1
                    ]
                    autoscaler_config = replace(
                        base_autoscaler_config,
                        allowed_rpu_sizes=candidate_rpu,
                    )
                    autoscaler = Autoscaler(
                        slo_resolver=slo_resolver,
                        slo_objective=slo_objective,
                        provisioner_config=provisioner_config,
                        query_router_config=query_router_config,
                        autoscaler_config=autoscaler_config,
                        out_dir=cls.scratch_dir(),
                    )
                    snapshot = cls.ingest_initial(
                        workload=workload,
                        n_to_ingest=idx_of_first_non_ingested_query,
                        initial_cluster_sizes=manifest["initial_rpus"],
                        query_router=router,
                        autoscaler=autoscaler,
                    )

                    for rep in range(reps):
                        t0 = time.perf_counter()
                        _, stats = autoscaler._select_rpu(
                            rel_time_s=next_time_s,
                            pool_snapshot_with_current_query=snapshot,
                        )
                        elapsed_s = time.perf_counter() - t0
                        total_simulated_queries = (
                            stats.pre_spinup_arrivals_processed
                            + sum(stats.post_spinup_arrivals_processed.values())
                        )
                        rows.append(
                            {
                                "candidate_rpu": f"[{','.join(map(str, sorted(candidate_rpu)))}]",
                                "arrival_rate_qps": arrival_rate_qps,
                                "simulated_queries": total_simulated_queries,
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
            x_col="simulated_queries",
            y_col="elapsed_s",
            shape_col="candidate_rpu",
            color_col="arrival_rate_qps",
            shape_legend_title="Candidate RPU",
            colorbar_label="Arrival Rate (Queries/s)",
            cmap_colors=[
                Palette.light_gray,
                Palette.light_green,
                Palette.light_green_sat,
            ],
            log_x=True,
            log_y=True,
        )
