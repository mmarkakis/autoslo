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
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.visualizations.colors import Palette


class RoutingEfficiencyBenchmark(MicrobenchmarkRunner):
    @classmethod
    def name(cls) -> str:
        return "routing_efficiency"

    @classmethod
    def required_keys(cls) -> list[str]:
        return [
            "model_id",
            "template_selection_seed",
            "rpu",
            "reps",
            "cluster_values",
            "active_queries_per_cluster_values",
            "routing_policy_name",
            "slo_s",
            "slo_metric",
            "slo_threshold",
        ]

    @classmethod
    def run_from_manifest(cls, manifest: dict) -> None:

        # Parse manifest parameters.
        model_id = str(manifest["model_id"])
        seed = int(manifest["template_selection_seed"])
        rpu = int(manifest["rpu"])
        reps = int(manifest["reps"])
        cluster_values = [int(v) for v in manifest["cluster_values"]]
        active_values = [
            int(v) for v in manifest["active_queries_per_cluster_values"]
        ]

        # Setup.
        model = IconqModel.load(model_id)
        max_needed = (max(cluster_values) * max(active_values)) + 1
        query_pool = cls.load_uniform_tpcds_template_001_pool(
            model=model,
            allowed_rpu_sizes=[rpu],
            n_queries=max_needed,
            template_selection_seed=seed,
        )
        rows: list[dict[str, float | int | str]] = []
        total_steps = len(cluster_values) * len(active_values) * reps

        # Run.
        with cls.make_progress() as progress:
            task = progress.add_task("Routing benchmark", total=total_steps)
            for n_clusters, active_per_cluster in itertools.product(
                cluster_values, active_values
            ):
                needed_active = n_clusters * active_per_cluster
                initial_queries = query_pool[:needed_active]
                incoming = query_pool[needed_active]
                router = QueryRouter(
                    slo_resolver=SloResolver(
                        SloResolverConfig(slo_s=float(manifest["slo_s"]))
                    ),
                    slo_objective=SloObjective(
                        SloObjectiveConfig(
                            slo_metric=str(manifest["slo_metric"]),
                            slo_threshold=float(manifest["slo_threshold"]),
                        )
                    ),
                    query_router_config=QueryRouterConfig(
                        routing_policy_name=manifest["routing_policy_name"],
                        iconq_model_id=model_id,
                    ),
                    iconq_model=model,
                    out_dir=cls.scratch_dir(),
                )
                snapshot = cls.ingest_initial(
                    initial_queries=initial_queries,
                    n_clusters=n_clusters,
                    rpu=rpu,
                    cache_state_dim=model.iconq_query_featurizer.num_tables,
                    query_router=router,
                )

                for rep in range(reps):

                    t0 = time.perf_counter()
                    router.route_query(
                        query=incoming,
                        snapshot=snapshot,
                        rel_time_s=incoming.rel_start_time_s,
                    )
                    elapsed_s = time.perf_counter() - t0
                    rows.append(
                        {
                            "clusters": n_clusters,
                            "active_per_cluster": active_per_cluster,
                            "total_active_queries": needed_active,
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
