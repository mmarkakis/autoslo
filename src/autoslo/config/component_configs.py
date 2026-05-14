from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Self

import autoslo.filesystem.path_utils as pu
from autoslo.slo.slo_metric import SloMetric

# An "execution config" is the YAML config file that a user provides to run
# the workload runner or workload simulator. It contains:
# - workload_config
# - slo_resolver_config
# - slo_objective_config
# - provisioner_config
# - [workload_runner_config]
# - managed_cluster_pool_config
# - scheduled_spinups
# - query_router_config
# - autoscaler_config


# A "tuner config" is the YAML config file that a user provides to run the
# policy tuner, alongisde an initial execution config. It contains:
# - sampling_config
# - spinup_optimizer_config
# - [autoscaling_param_sweep.param_sweep_config]
# - [query_routing_param_sweep.param_sweep_config]


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

        if sub_config_key not in cfg:
            raise ValueError(f"{sub_config_key} not found in config.")

        # Treat kwargs as overrides that take precedence over the config file.
        d = {**cfg[sub_config_key], **kwargs}
        return cls(**d)

    def to_dict(self) -> dict:
        """
        Convert the dataclass to a dictionary for dumping to YAML.
        """
        return asdict(self)


@dataclass(frozen=True)
class WorkloadConfig(_PartialConfig):
    """
    Configuration for a workload, including the name and parent directory.
    """

    workload_name: str
    workload_dir: Optional[str | Path] = None  # Defaults to data/workloads/
    target_date: Optional[str] = None  # YYYY-MM-DD
    rescale_factor: float = 1.0

    def __post_init__(self):
        if self.workload_dir is None:
            object.__setattr__(self, "workload_dir", pu.get_workloads_dir())

    def id(self) -> str:
        return "__".join(
            [
                self.workload_name,
                self.target_date or "target",
                f"rf{self.rescale_factor:.3f}",
            ]
        )

    @classmethod
    def from_id(cls, wid: str) -> "WorkloadConfig":
        """Reconstruct a WorkloadConfig from its id() string."""
        parts = wid.split("__")
        name = "__".join(parts[:-3])
        target, rf = parts[-3], parts[-1]
        return cls(
            workload_name=name,
            target_date=None if target == "target" else target,
            rescale_factor=float(rf[2:]),
        )


@dataclass(frozen=True)
class ReservoirConfig(_PartialConfig):
    """
    Configuration for the query reservoir, including the schema and parameters
    for loading the reservoir. This is a subclass of WorkloadConfig since the
    reservoir is built from a workload.
    """

    workload_name: str
    last_day_date_inclusive: str  # YYYY-MM-DD
    num_days: int = 1
    workload_dir: Optional[str | Path] = None  # Defaults to data/workloads/

    def __post_init__(self):
        if self.workload_dir is None:
            object.__setattr__(self, "workload_dir", pu.get_workloads_dir())

    def to_workload_config(self) -> WorkloadConfig:
        """
        Convert this ReservoirConfig to a WorkloadConfig that covers at least
        the same time period.
        """
        return WorkloadConfig(
            workload_name=self.workload_name,
            workload_dir=self.workload_dir,
            target_date=None,
            rescale_factor=1.0,
        )


@dataclass(frozen=True)
class ForecasterConfig(_PartialConfig):
    """
    Configuration for the forecaster.
    """

    forecast_policy_name: str
    decay_factor: float = 0.5
    fixed_queries_per_hour: int = 100
    rescale_factor: float = 1.0
    reservoir_config: Optional[ReservoirConfig] = None

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> Self:
        """
        Override to also parse the nested reservoir_config if present.
        """
        try:
            reservoir_config = ReservoirConfig.from_config(
                cfg["forecaster_config"]
            )
        except (KeyError, ValueError):
            reservoir_config = None
        return super().from_config(cfg, reservoir_config=reservoir_config)

    def __post_init__(self):
        if self.decay_factor < 0 or self.decay_factor > 1:
            raise ValueError(
                f"decay_factor must be in [0, 1], got {self.decay_factor}"
            )

        if self.fixed_queries_per_hour <= 0:
            raise ValueError(
                f"fixed_queries_per_hour must be positive, got "
                f"{self.fixed_queries_per_hour}"
            )


@dataclass(frozen=True)
class SamplingConfig(_PartialConfig):
    """
    Configuration for sampling, including the random seed and sampling method.
    """

    num_scenarios: int
    seed: int = 42
    train_fraction: float = 0.6
    aggregation_method: str = "mean"
    forecaster_config: Optional[ForecasterConfig] = None

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> Self:
        """
        Override to also parse the nested forecaster_config if present.
        """
        try:
            forecaster_config = ForecasterConfig.from_config(
                cfg["sampling_config"]
            )
        except (KeyError, ValueError):
            forecaster_config = None
        return super().from_config(cfg, forecaster_config=forecaster_config)


@dataclass(frozen=True)
class AutoscalerConfig(_PartialConfig):
    """
    Configuration for the autoscaler, including the target SLO and scaling
    parameters.
    """

    allowed_rpu_sizes: list[int]
    autoscaling_policy: str = "add_single_best"
    min_cluster_lifetime_s: float = 1200.0
    idle_time_before_tear_down_s: float = 300.0
    observation_window_s: float = 120.0
    min_observations_to_act: int = 5
    slo_tightening_factor: float = 1.0
    force_one_decision_after_query_fraction: Optional[float] = None  # When set, 
    # disables reactive autoscaling and fires a single forced decision after 
    # this fraction of the workload's queries have been routed.


@dataclass(frozen=True)
class QueryRouterConfig(_PartialConfig):
    """
    Configuration for the QueryRouter.
    """

    routing_policy_name: str = "use_iconq_model"
    iconq_model_id: str = "1771539369"
    cluster_cache_state_update_alpha: float = 0.7

    cache_risk_cost_multiplier: float = 0.0
    # adj_cost = real_cost * (1 + multiplier * risk)
    cache_risk_coverage: float = 0.9
    cache_risk_epsilon: float = 1e-6

    forecaster_config: Optional[ForecasterConfig] = None

    @classmethod
    def from_config(cls, cfg: dict, **kwargs) -> Self:
        """
        Override to also parse the nested forecaster_config if present.
        """
        try:
            forecaster_config = ForecasterConfig.from_config(
                cfg["query_router_config"]
            )
        except (KeyError, ValueError):
            forecaster_config = None
        return super().from_config(cfg, forecaster_config=forecaster_config)

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

    max_threads: int = 512
    closed_loop: bool = False
    schema_name: str = "ext_tpcds1000"


@dataclass(frozen=True)
class ProvisionerConfig(_PartialConfig):
    """
    Configuration for the provisioner, including AWS credentials and cluster
    parameters.
    """

    aws_config_path: str
    cluster_cache_state_dim: int
    run_id: str
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
class SpinupOptimizerConfig(_PartialConfig):
    """
    Configuration for the spin-up optimizer, including the criteria
    for accepting spin-ups and the initial RPU candidates.
    """

    max_spinups: int = 5
    min_delinquent_workload_fraction: float = 0.5
    lead_time_s: float = 360.0
    initial_rpu_candidates: list[list[int]] = field(
        default_factory=lambda: [[16], [8], [32]]
    )
    allowed_rpu_sizes: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    max_attempts_per_round: int = 10
    # Minimum distance (seconds) between any two retained placement-time
    # candidates.  After scoring, candidates are greedily selected in
    # descending-score order; a candidate is dropped if it falls within
    # this distance of an already-selected candidate.
    #
    # None (the default) means use lead_time_s.  Two placement times closer
    # than lead_time_s apart produce a spin-up arrival difference smaller than
    # the lead window itself, so they are effectively targeting the same point
    # in the workload and should count as a single attempt.
    # Set to 0.0 to disable spacing enforcement entirely.
    min_candidate_spacing_s: Optional[float] = None


@dataclass(frozen=True)
class ParamSweepConfig(_PartialConfig):
    """
    Configuration for a parameter sweep, including the target component and
    the parameters to sweep.
    """

    params: dict[str, list[Any]]
    strategy: str = "grid"
    val_top_k: int = 10

    # For random strategy
    budget: int = 20
    seed: int = 42

    # For coordinate descent strategy
    max_cycles: int = 3
    starting_point: Optional[dict[str, Any]] = None
