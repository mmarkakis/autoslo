import os
from dataclasses import dataclass
from typing import Optional

import autoslo.config.utils as cfgu
import autoslo.filesystem.path_utils as pu
from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.capacity_checkpoint import CapacityCheckpoint
from autoslo.clusters.cluster import Cluster
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.clusters.redshift_provisioner import RedshiftServerlessProvisioner
from autoslo.config.component_configs import (
    AutoscalerConfig,
    ForecasterConfig,
    ManagedClusterPoolConfig,
    ProvisionerConfig,
    QueryRouterConfig,
    ReservoirConfig,
    SloObjectiveConfig,
    SloResolverConfig,
    WorkloadConfig,
    WorkloadRunnerConfig,
)
from autoslo.filesystem.logging import StructuredLogHandler, setup_run_logging
from autoslo.filesystem.structured_events import wall_clock_utc
from autoslo.forecasting.forecaster import Forecaster
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.reservoir import QueryReservoir
from autoslo.workload_definition.workload import Workload


@dataclass(frozen=True)
class StructuredConfig:
    run_id: str
    out_dir: str
    write_text_log: bool
    iconq_model: IconqModel
    workload: Workload
    slo_objective: SloObjective
    slo_resolver: SloResolver
    pool: ManagedClusterPool
    capacity_checkpoints: list[CapacityCheckpoint]
    router: QueryRouter
    autoscaler: Autoscaler
    structured_log_handler: StructuredLogHandler
    workload_runner_config: WorkloadRunnerConfig

    @classmethod
    def build(
        cls,
        cfg: dict,
        out_dir: Optional[str] = None,
        write_text_log: bool = False,
        is_runner: bool = False,
    ) -> "StructuredConfig":
        """
        Build a ``StructuredConfig`` from the given configuration dictionary and
        other parameters.

        Parameters
        ----------
        cfg :
            The configuration dictionary, typically loaded from a YAML file.
        is_runner :
            Whether this is being built for a live runner (as opposed to a
            simulator).  This controls certain defaults and behaviors, such as
            the provisioner type and output directory structure.
        """

        # ── Determine run_id and set up logging ──────────────────────────────
        run_id = str(int(wall_clock_utc() * 1000))
        default_parent = "runs" if is_runner else "simulation_runs"
        out_dir = out_dir or os.path.join(
            pu.get_data_path(), default_parent, run_id
        )
        write_text_log = write_text_log
        structured_log_handler = setup_run_logging(
            out_dir=out_dir,
            write_text_log=write_text_log,
        )

        # ── basic ────────────────────────────────────────────────────────
        iconq_model_id: str = cfgu.getd(
            cfg, "basic_config.iconq_model_id", required=True
        )
        iconq_model = IconqModel.load(iconq_model_id)

        # ── workload ─────────────────────────────────────────────────────
        workload_config = WorkloadConfig.from_config(cfg)
        workload = Workload(workload_config=workload_config)
        workload.populate_featurizations_and_isolated_predictions(
            iconq_model=iconq_model,
            allowed_rpu_sizes=Cluster.ALL_ALLOWED_RPU_SIZES,
        )
        if is_runner:
            workload.print_summary()

        # ── SLO ──────────────────────────────────────────────────────────
        slo_resolver_config = SloResolverConfig.from_config(cfg)
        slo_objective_config = SloObjectiveConfig.from_config(cfg)
        slo_resolver = SloResolver(slo_resolver_config)
        slo_objective = SloObjective(slo_objective_config)

        # ── Runner config ──────────────────────────────────────────────────
        workload_runner_config = WorkloadRunnerConfig.from_config(cfg)

        # ── Provisioner ─────────────────────────────────────────────
        cluster_cache_state_dim = iconq_model.iconq_query_featurizer.num_tables
        provisioner_config = ProvisionerConfig.from_config(
            cfg, cluster_cache_state_dim=cluster_cache_state_dim
        )
        provisioner = (
            RedshiftServerlessProvisioner(provisioner_config)
            if is_runner
            else SimulatedProvisioner(provisioner_config)
        )

        # ── Capacity Checkpoints ─────────────────────────────────────────────
        capacity_checkpoints = [
            CapacityCheckpoint(
                rel_time_s=float(cp["rel_time_s"]),
                min_rpus=tuple(cp["min_rpus"]),
            )
            for cp in cfgu.getd(cfg, "capacity_checkpoints", [])
        ]

        # ── Managed Cluster Pool ─────────────────────────────────────────────
        num_reserved_clusters = CapacityCheckpoint.worst_case_total_spinups(
            capacity_checkpoints
        )
        managed_cluster_pool_config = ManagedClusterPoolConfig.from_config(
            cfg, num_reserved_clusters=num_reserved_clusters
        )
        pool: ManagedClusterPool = ManagedClusterPool(
            provisioner=provisioner,
            config=managed_cluster_pool_config,
        )

        # ── Forecasting ──────────────────────────────────────────────────────
        reservoir_config = ReservoirConfig.from_config(cfg)
        query_reservoir = QueryReservoir(reservoir_config=reservoir_config)
        forecaster_config = ForecasterConfig.from_config(cfg)
        forecaster = Forecaster(
            reservoir=query_reservoir,
            forecaster_config=forecaster_config,
        )
        target_date = workload.abs_start_time_range()[0].date()
        forecasted_workload, _ = forecaster.forecast(
            target_date=target_date,
            use_fixed_queries_per_hour=True,
            out_dir=out_dir,
            workload_name="forecasted_workload",
        )
        forecasted_workload = forecasted_workload.rescale_rel_times(
            workload_config.rescale_factor
        )
        rel_time_s_to_forecasted_table_vecs = (
            forecasted_workload.get_rel_time_s_to_table_vecs(
                iconq_query_featurizer=iconq_model.iconq_query_featurizer
            )
        )

        # ── QueryRouter ──────────────────────────────────────────────────────
        query_router_config = QueryRouterConfig.from_config(
            cfg,
            rel_time_s_to_forecasted_table_vecs=rel_time_s_to_forecasted_table_vecs,
        )
        router: QueryRouter = QueryRouter(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            query_router_config=query_router_config,
        )

        # ── Autoscaler ──────────────────────────────────────────────────────
        autoscaler_config = AutoscalerConfig.from_config(cfg)
        autoscaler = Autoscaler(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            iconq_model=iconq_model,
            query_router_config=query_router_config,
            autoscaler_config=autoscaler_config,
            cluster_cache_state_dim=cluster_cache_state_dim,
        )

        return StructuredConfig(
            run_id=run_id,
            out_dir=out_dir,
            write_text_log=write_text_log,
            iconq_model=iconq_model,
            workload=workload,
            slo_objective=slo_objective,
            slo_resolver=slo_resolver,
            pool=pool,
            capacity_checkpoints=capacity_checkpoints,
            router=router,
            autoscaler=autoscaler,
            structured_log_handler=structured_log_handler,
            workload_runner_config=workload_runner_config,
        )
