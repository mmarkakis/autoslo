import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.cluster import Cluster
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.clusters.redshift_provisioner import RedshiftServerlessProvisioner
from autoslo.clusters.scheduled_spinup import ScheduledSpinUp
from autoslo.config.component_configs import (
    AutoscalerConfig,
    ManagedClusterPoolConfig,
    ProvisionerConfig,
    QueryRouterConfig,
    SloObjectiveConfig,
    SloResolverConfig,
    WorkloadConfig,
    WorkloadRunnerConfig,
)
from autoslo.filesystem.logging import StructuredLogHandler, setup_run_logging
from autoslo.filesystem.structured_events import wall_clock_utc
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.workload import Workload


@dataclass(frozen=True)
class ExecutionConfig:
    run_id: str
    out_dir: str | Path
    workload: Workload
    pool: ManagedClusterPool
    scheduled_spinups: list[ScheduledSpinUp]
    router: QueryRouter
    autoscaler: Autoscaler
    structured_log_handler: StructuredLogHandler
    workload_runner_config: WorkloadRunnerConfig

    @classmethod
    def build(
        cls,
        cfg: dict,
        out_dir: Optional[str | Path] = None,
        write_text_log: bool = False,
        is_runner: bool = False,
    ) -> "ExecutionConfig":
        """
        Build a ``ExecutionConfig`` from the given configuration dictionary and
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
        default_parent = "runs" if is_runner else "simulator_runs"
        out_dir = out_dir or os.path.join(
            pu.get_data_path(), default_parent, run_id
        )
        structured_log_handler = setup_run_logging(
            out_dir=out_dir,
            write_text_log=write_text_log,
        )

        # Parse direct configs.
        workload_config = WorkloadConfig.from_config(cfg)
        query_router_config = QueryRouterConfig.from_config(cfg)
        slo_resolver_config = SloResolverConfig.from_config(cfg)
        slo_objective_config = SloObjectiveConfig.from_config(cfg)
        autoscaler_config = AutoscalerConfig.from_config(cfg)
        scheduled_spinups = ScheduledSpinUp.from_config(cfg)
        if is_runner:
            workload_runner_config = WorkloadRunnerConfig.from_config(cfg)
        else:
            # Not needed for the simulator, so use defaults to let it be absent.
            workload_runner_config = WorkloadRunnerConfig()

        # Necessary compute
        iconq_model = IconqModel.load(query_router_config.iconq_model_id)
        cluster_cache_state_dim = iconq_model.iconq_query_featurizer.num_tables
        num_reserved_clusters = ScheduledSpinUp.total_spinups(scheduled_spinups)

        # Parse remaining configs
        provisioner_config = ProvisionerConfig.from_config(
            cfg, cluster_cache_state_dim=cluster_cache_state_dim
        )
        managed_cluster_pool_config = ManagedClusterPoolConfig.from_config(
            cfg, num_reserved_clusters=num_reserved_clusters
        )

        # Initialize.
        workload = Workload(workload_config=workload_config)
        workload.populate_featurizations_and_isolated_predictions(
            iconq_model=iconq_model,
            allowed_rpu_sizes=Cluster.ALL_ALLOWED_RPU_SIZES,
        )
        if is_runner:
            workload.print_summary()
        slo_resolver = SloResolver(slo_resolver_config)
        slo_objective = SloObjective(slo_objective_config)
        provisioner = (
            RedshiftServerlessProvisioner(provisioner_config)
            if is_runner
            else SimulatedProvisioner(provisioner_config)
        )
        pool = ManagedClusterPool(
            provisioner=provisioner,
            config=managed_cluster_pool_config,
        )
        router = QueryRouter(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            query_router_config=query_router_config,
            iconq_model=iconq_model,
            out_dir=out_dir,
        )
        autoscaler = Autoscaler(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            iconq_model=iconq_model,
            query_router_config=query_router_config,
            autoscaler_config=autoscaler_config,
            cluster_cache_state_dim=cluster_cache_state_dim,
            out_dir=out_dir,
        )

        return ExecutionConfig(
            run_id=run_id,
            out_dir=out_dir,
            workload=workload,
            pool=pool,
            scheduled_spinups=scheduled_spinups,
            router=router,
            autoscaler=autoscaler,
            structured_log_handler=structured_log_handler,
            workload_runner_config=workload_runner_config,
        )
