import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import autoslo.utils.config as cfgu
import autoslo.utils.paths as pu
from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.capacity_checkpoint import CapacityCheckpoint
from autoslo.clusters.cluster import Cluster
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.clusters.redshift_provisioner import RedshiftServerlessProvisioner
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter, QueryRouterPolicy
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.logging import StructuredLogHandler, setup_run_logging
from autoslo.workload_definition.query_text_registry import QueryTextRegistry
from autoslo.workload_definition.schema import Schema
from autoslo.workload_definition.workload import Workload


def _make_out_dir(
    run_id: str,
    out_dir_override: str | None = None,
    experiment_name: str | None = None,
    overwrite_experiment: bool = False,
    is_runner: bool = False,
) -> str:
    """
    Determine the output directory for a run based on the given parameters.
    """
    default_out_dir_parent = "runs" if is_runner else "simulator_runs"

    if out_dir_override is not None:
        out_dir = os.path.join(str(out_dir_override), run_id)
    elif experiment_name:
        experiment_dir = os.path.join(
            pu.get_data_path(), default_out_dir_parent, experiment_name
        )
        if os.path.exists(experiment_dir) and overwrite_experiment:
            shutil.rmtree(experiment_dir)
        out_dir = os.path.join(experiment_dir, run_id)
    else:
        out_dir = os.path.join(
            pu.get_data_path(), default_out_dir_parent, run_id
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


@dataclass(frozen=True)
class StructuredConfig:
    query_text_registry: QueryTextRegistry
    iconq_model: IconqModel
    closed_loop: bool
    workload: Workload
    slo_objective: SloObjective
    slo_resolver: SloResolver
    thread_pool_executor: ThreadPoolExecutor
    pool: ManagedClusterPool
    capacity_checkpoints: list[CapacityCheckpoint]
    router: QueryRouter
    autoscaler: Autoscaler
    out_dir: str
    experiment_name: Optional[str]
    write_text_log: bool
    structured_log_handler: StructuredLogHandler

    @classmethod
    def build(
        cls,
        cfg: dict,
        run_id: str,
        workload: Optional[Workload] = None,
        is_runner: bool = False,
    ) -> "StructuredConfig":
        """
        Build a ``StructuredConfig`` from the given configuration dictionary and
        other parameters.

        Parameters
        ----------
        cfg :
            The configuration dictionary, typically loaded from a YAML file.
        run_id :
            A unique identifier for this run, used for output organization.
        workload :
            An optional pre-constructed workload.  If not provided, it will be
            constructed from the configuration.
        is_runner :
            Whether this is being built for a live runner (as opposed to a
            simulator).  This controls certain defaults and behaviors, such as
            the provisioner type and output directory structure.
        """

        # ── basic ────────────────────────────────────────────────────────
        schema_name: str = cfgu.getd(
            cfg, "basic_config.schema_name", required=True
        )
        schema = Schema.load(schema_name)

        query_text_registry = QueryTextRegistry(schema_name)
        iconq_model_id: str = cfgu.getd(
            cfg, "basic_config.iconq_model_id", required=True
        )
        iconq_model = IconqModel.load(iconq_model_id)

        # ── workload ─────────────────────────────────────────────────────
        closed_loop: bool = bool(
            cfgu.getd(cfg, "workload_config.closed_loop", False)
        )
        if workload is None:
            workload_name = cfgu.getd(
                cfg, "workload_config.workload_name", required=True
            )
            abs_start = cfgu.getd(cfg, "workload_config.abs_start_time_start")
            abs_end = cfgu.getd(cfg, "workload_config.abs_start_time_end")
            rescale = cfgu.getd(cfg, "workload_config.rescale_factor", None)
            workload = Workload(
                workload_name=workload_name, schema_name=schema_name
            )
            workload.prepare(
                abs_start=abs_start,
                abs_end=abs_end,
                rescale_factor=rescale,
            )
            workload.populate_featurizations_and_isolated_predictions(
                iconq_model=iconq_model,
                allowed_rpu_sizes=Cluster.ALL_ALLOWED_RPU_SIZES,
            )
        if is_runner:
            workload.print_summary()

        # ── SLO ──────────────────────────────────────────────────────────
        slo_s: float = cfgu.getd(cfg, "slo_config.slo_s", 10.0)
        slo_metric = SloMetric(
            cfgu.getd(cfg, "slo_config.slo_metric", "relative")
        )
        slo_threshold: float = float(
            cfgu.getd(cfg, "slo_config.slo_threshold", 0.0)
        )
        slo_dict_filename: Optional[str] = cfgu.getd(
            cfg, "slo_config.slo_dict_filename"
        )
        slo_resolver = SloResolver(slo_s, slo_dict_filename)
        slo_objective = SloObjective(
            slo_metric=slo_metric,
            slo_threshold=slo_threshold,
        )

        # ── Runner config ──────────────────────────────────────────────────
        max_threads = cfgu.getd(cfg, "runner_config.max_threads", 10)
        thread_pool_executor = ThreadPoolExecutor(max_workers=max_threads)

        # ── Provisioner ─────────────────────────────────────────────
        absolute_aws_config_path = os.path.join(
            pu.AUTOSLO_ROOT,
            cfgu.getd(
                cfg,
                "provisioner_config.relative_aws_config_path",
                "data/__run_configs/aws.yml",
            ),
        )
        spin_up_delay_s = cfgu.getd(
            cfg, "provisioner_config.spin_up_delay_s", 300.0
        )
        provisioner = (
            RedshiftServerlessProvisioner(
                aws_config_path=absolute_aws_config_path
            )
            if is_runner
            else SimulatedProvisioner(spin_up_delay_s=spin_up_delay_s)
        )

        # ── Output ───────────────────────────────────────────────────────────
        experiment_name: Optional[str] = cfgu.getd(
            cfg, "output_config.experiment_name"
        )
        overwrite_experiment: bool = cfgu.getd(
            cfg, "output_config.overwrite_experiment", False
        )
        write_text_log: bool = cfgu.getd(
            cfg, "output_config.write_text_log", False
        )
        if is_runner:
            write_text_log = True
        out_dir_override = cfgu.getd(cfg, "output_config.out_dir", None)
        out_dir = _make_out_dir(
            run_id=run_id,
            out_dir_override=out_dir_override,
            experiment_name=experiment_name,
            overwrite_experiment=overwrite_experiment,
            is_runner=is_runner,
        )

        # ── Logging (before pool so provisioner events are captured) ──────
        structured_log_handler = setup_run_logging(
            out_dir=out_dir,
            write_text_log=write_text_log,
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
        initial_rpus = cfgu.getd(
            cfg,
            "managed_cluster_pool_config.initial_rpus",
            [8],
        )
        maxconns = cfgu.getd(
            cfg,
            "managed_cluster_pool_config.maxconns",
            1000,
        )
        pool: ManagedClusterPool = ManagedClusterPool(
            provisioner=provisioner,
            initial_rpus=initial_rpus,
            maxconns=maxconns,
            search_path=schema.search_path,
            collect_cluster_stats=True,
            run_id=run_id,
            out_dir=out_dir,
            background_executor=thread_pool_executor if is_runner else None,
        )

        # ── QueryRouter ──────────────────────────────────────────────────────
        routing_policy_str: str = cfgu.getd(
            cfg, "routing_config.routing_policy", "use_iconq_model"
        )
        routing_policy = QueryRouterPolicy(routing_policy_str)
        router: QueryRouter = QueryRouter(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            routing_policy=routing_policy,
        )

        # ── Autoscaler ──────────────────────────────────────────────────────
        autoscaler = Autoscaler(
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            iconq_model=iconq_model,
            routing_policy=routing_policy,
            allowed_rpu_sizes=cfgu.getd(
                cfg,
                "autoscaling_config.allowed_rpu_sizes",
                Cluster.ALL_ALLOWED_RPU_SIZES,
            ),
            min_cluster_lifetime_s=cfgu.getd(
                cfg,
                "autoscaling_config.min_cluster_lifetime_s",
                1200.0,
            ),
            idle_time_before_tear_down_s=cfgu.getd(
                cfg,
                "autoscaling_config.idle_time_before_tear_down_s",
                600.0,
            ),
            observation_window_s=cfgu.getd(
                cfg,
                "autoscaling_config.observation_window_s",
                300.0,
            ),
            min_observations_to_act=cfgu.getd(
                cfg,
                "autoscaling_config.min_observations_to_act",
                5,
            ),
            slo_tightening_factor=cfgu.getd(
                cfg,
                "autoscaling_config.slo_tightening_factor",
                1.0,
            ),
        )

        return StructuredConfig(
            query_text_registry=query_text_registry,
            iconq_model=iconq_model,
            closed_loop=closed_loop,
            workload=workload,
            slo_objective=slo_objective,
            slo_resolver=slo_resolver,
            thread_pool_executor=thread_pool_executor,
            pool=pool,
            capacity_checkpoints=capacity_checkpoints,
            router=router,
            autoscaler=autoscaler,
            out_dir=out_dir,
            experiment_name=experiment_name,
            write_text_log=write_text_log,
            structured_log_handler=structured_log_handler,
        )
