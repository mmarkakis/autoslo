from __future__ import annotations

import itertools
import time

import pandas as pd

from autoslo.config.component_configs import (
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


class RoutingEfficiencyBenchmark(MicrobenchmarkRunner):
    @classmethod
    def name(cls) -> str:
        return "routing_efficiency"

    @classmethod
    def required_keys(cls) -> list[str]:
        return [
            "workload_seed",
            "rpu",
            "reps",
            "cluster_values",
            "active_queries_per_cluster_values",
            "slo_resolver_config",
            "slo_objective_config",
            "query_router_config",
        ]

    @classmethod
    def run_from_manifest(cls, manifest: dict) -> None:

        # Parse manifest parameters.
        rpu = int(manifest["rpu"])
        reps = int(manifest["reps"])
        cluster_values = [int(v) for v in manifest["cluster_values"]]
        active_values = [
            int(v) for v in manifest["active_queries_per_cluster_values"]
        ]

        # Setup router and workload.
        router = QueryRouter(
            slo_resolver=SloResolver(SloResolverConfig.from_config(manifest)),
            slo_objective=SloObjective(
                SloObjectiveConfig.from_config(manifest)
            ),
            query_router_config=QueryRouterConfig.from_config(manifest),
            out_dir=cls.scratch_dir(),
        )
        total_queries_needed = max(cluster_values) * max(active_values) + 1
        workload = (
            PoissonWorkloadCreator.create_poisson_workload_with_n_queries(
                num_templates=99,
                num_query_texts_per_template=1,
                num_total_queries=total_queries_needed,
                poisson_lambda=total_queries_needed / 0.010,  # All within 10ms.
                seed=int(manifest["workload_seed"]),
                print_summary=False,
            )
        )
        workload.populate_featurizations_and_isolated_predictions(
            iconq_model=router.iconq_model,
            allowed_rpu_sizes=[rpu],
        )

        # Run.
        rows: list[dict[str, float | int | str]] = []
        with cls.make_progress() as progress:
            total_steps = len(cluster_values) * len(active_values) * reps
            task = progress.add_task("Routing benchmark", total=total_steps)
            for n_clusters, active_per_cluster in itertools.product(
                cluster_values, active_values
            ):
                total_active = n_clusters * active_per_cluster
                snapshot = cls.ingest_initial(
                    workload=workload,
                    n_to_ingest=total_active,
                    initial_cluster_sizes=[rpu] * n_clusters,
                    query_router=router,
                )
                next_q = workload.queries()[total_active]
                for rep in range(reps):
                    t0 = time.perf_counter()
                    router.route_query(
                        query=next_q,
                        snapshot=snapshot,
                        rel_time_s=next_q.rel_start_time_s,
                    )
                    elapsed_s = time.perf_counter() - t0
                    rows.append(
                        {
                            "clusters": n_clusters,
                            "active_per_cluster": active_per_cluster,
                            "total_active_queries": total_active,
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
            x_col="total_active_queries",
            y_col="elapsed_s",
            shape_col="clusters",
            color_col="active_per_cluster",
            shape_legend_title="Clusters",
            colorbar_label="Active Queries / Cluster",
            cmap_colors=[
                Palette.light_gray,
                Palette.light_blue,
                Palette.light_blue_sat,
            ],
            log_color_base=2,
            log_x=True,
            log_y=True,
        )
