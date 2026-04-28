from dataclasses import dataclass
from typing import Optional, Self

import numpy as np

from autoslo.routing.query_router import QueryRouterPolicy
from autoslo.slo.slo_metric import SloMetric


@dataclass(frozen=True)
class _PartialConfig:
    """
    A parent class for partial configs that implements parsing from a config
    file.
    """

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> Self:
        """
        Parse the relevant fields from the given config dict and return an
        instance of the PartialConfig subclass.
        """

        # Go from camel case to snake case to find the relevant section.
        sub_config_name = cls.__name__  # e.g. "AutoscalerConfig"
        sub_config_key = "".join(
            ["_" + c.lower() if c.isupper() else c for c in sub_config_name]
        ).lstrip(
            "_"
        )  # e.g. "autoscaler_config"

        if sub_config_key in cfg:
            cfg = cfg[sub_config_key]
        return cls(**cfg, **kwargs)


@dataclass(frozen=True)
class WorkloadConfig(_PartialConfig):
    """
    Configuration for a workload, including the name and schema.
    """

    workload_name: str
    workload_dir: Optional[str] = None  # Defaults to data/workloads/
    start_date_inclusive: Optional[str] = None  # YYYY-MM-DD
    end_date_inclusive: Optional[str] = None  # YYYY-MM-DD
    rescale_factor: float = 1.0


@dataclass(frozen=True)
class ReservoirConfig(WorkloadConfig):
    """
    Configuration for the query reservoir, including the schema and parameters
    for loading the reservoir. This is a subclass of WorkloadConfig since the
    reservoir is built from a workload.
    """
    
    def as_workload_config(self) -> WorkloadConfig:
        """
        Return a WorkloadConfig with the same fields as this ReservoirConfig.
        """
        return WorkloadConfig(
            workload_name=self.workload_name,
            workload_dir=self.workload_dir,
            start_date_inclusive=self.start_date_inclusive,
            end_date_inclusive=self.end_date_inclusive,
            rescale_factor=self.rescale_factor,
        )


@dataclass(frozen=True)
class AutoscalerConfig(_PartialConfig):
    """
    Configuration for the autoscaler, including the target SLO and scaling
    parameters.
    """

    allowed_rpu_sizes: list[int]
    min_cluster_lifetime_s: float = 1200.0
    idle_time_before_tear_down_s: float = 300.0
    observation_window_s: float = 120.0
    min_observations_to_act: int = 5
    slo_tightening_factor: float = 1.0


@dataclass(frozen=True)
class QueryRouterConfig(_PartialConfig):
    """
    Configuration for the QueryRouter.
    """

    rel_time_s_to_forecasted_table_vecs: dict[float, np.ndarray]

    routing_policy: QueryRouterPolicy = QueryRouterPolicy.USE_ICONQ_MODEL
    cluster_cache_state_update_alpha: float = 0.7

    cache_risk_cost_multiplier: float = 0.0
    # adj_cost = real_cost * (1 + multiplier * risk)
    cache_risk_coverage: float = 0.9
    cache_risk_epsilon: float = 1e-6

    def __post_init__(self):
        if not (0 <= self.cache_risk_cost_multiplier):
            raise ValueError(
                f"cache_risk_cost_multiplier must be non-negative, "
                f"got {self.cache_risk_cost_multiplier}"
            )
        if not (0 <= self.cache_risk_coverage <= 1):
            raise ValueError(
                f"cache_risk_coverage must be in [0, 1], "
                f"got {self.cache_risk_coverage}"
            )
        if not (0 < self.cache_risk_epsilon):
            raise ValueError(
                f"cache_risk_epsilon must be positive, "
                f"got {self.cache_risk_epsilon}"
            )


@dataclass(frozen=True)
class SloResolverConfig(_PartialConfig):
    """
    Configuration for the SloResolver, including the default SLO and any
    per-template overrides.
    """

    slo_s: float
    slo_dict_filename: Optional[str] = None


@dataclass(frozen=True)
class SloObjectiveConfig(_PartialConfig):
    """
    Configuration for the SLO objective, including the metric and threshold.
    """

    slo_metric: str | SloMetric
    slo_threshold: float

@dataclass(frozen=True)
class WorkloadRunnerConfig(_PartialConfig):
    """
    Configuration for the workload runner, including max threads and whether to
    run in closed loop.
    """

    max_threads: int
    closed_loop: bool
    schema_name: str

@dataclass(frozen=True)
class ProvisionerConfig(_PartialConfig):
    """
    Configuration for the provisioner, including AWS credentials and cluster
    parameters.
    """

    aws_config_path: str
    cluster_cache_state_dim: int
    spin_up_delay_s: float = 300.0

@dataclass(frozen=True)
class ManagedClusterPoolConfig(_PartialConfig):
    """
    Configuration for the ManagedClusterPool.
    """

    initial_rpus: list[int]
    num_reserved_clusters: int = 0
    max_clusters: int = 20
    maxconns: int = 1000

@dataclass(frozen=True)
class ForecastPolicyConfig(_PartialConfig):
    """
    Configuration for the forecast policy, including the reservoir config and
    forecast horizon.
    """

    forecast_policy_name: str
    decay_factor: float = 0.5
    fixed_queries_per_hour: int = 100


@dataclass(frozen=True)
class OutputConfig(_PartialConfig):
    """
    Configuration for output, including the output directory and whether to
    overwrite existing files.
    """

    out_dir: str
    overwrite: bool = False
