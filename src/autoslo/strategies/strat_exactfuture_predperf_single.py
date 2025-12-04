import os

import numpy as np
import xgboost as xgb
import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.f_minimal import FMinimal
from autoslo.routing.query_router import QueryRouter
from autoslo.routing.r_fixed import RFixed
from autoslo.strategies.slo_strategy import SLOStrategy
from autoslo.strategies.slo_strategy_performance import (
    E2ESLOMetrics,
    SLOStrategyPerformance,
)
from autoslo.workload_definition.composite import Composite


class StratExactFuturePredPerfSingle(SLOStrategy):
    """
    SLO strategy that assumes access to tomorrow's exact workload features, but
    then uses a model to predict performance on each single-cluster blueprint.
    Among the blueprints predicted to meet the SLO, selects the one with the
    lowest RPU.
    """

    def __init__(
        self,
        model_training_run_id: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the StratExactFuturePredPerfSingle strategy.

        Parameters:
            model_training_run_id: The run ID of the model training run.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        super().__init__(*args, **kwargs)

        # Initialize and load the model.
        self.model = xgb.XGBRegressor()
        model_dir = os.path.join(pu.get_models_dir(), model_training_run_id)
        model_path = os.path.join(model_dir, "model.json")
        self.model.load_model(model_path)

        # Read the model config to find the summary metrics.
        training_params_path = os.path.join(model_dir, f"training_params.yml")
        with open(training_params_path, "r") as f:
            tp = yaml.safe_load(f)
        if "feature_set" not in tp or "label" not in tp:
            raise ValueError(
                "Model training parameters file is missing "
                "'feature_set' or 'label' entries."
            )
        self.input_summary_metric = tp["feature_set"].split("_")[-1]
        self.output_summary_metric = tp["label"].split("_")[-1]

        # Initialize the featurizer.
        self.featurizer = FMinimal(summary_metric=self.input_summary_metric)

    def suggest(
        self,
        workload: Composite,
        day_idx: int,
        latency_slo_s: float,
        *args,
        **kwargs,
    ) -> tuple[Blueprint, QueryRouter]:
        options: list[tuple[Blueprint, QueryRouter]] = []
        option_perfs: list[E2ESLOMetrics] = []

        diffs_from_slo: list[float] = []

        # For each candidate single-cluster blueprint, evaluate its past
        # performance over the specified window size.
        for rpu in sorted(Cluster.all_allowed_rpu_sizes()):
            blueprint = Blueprint.one_cluster_with(rpu)
            query_router = RFixed(blueprint, blueprint.cluster_names[0])
            options.append((blueprint, query_router))

            workload_trace = workload.get_most_recent_trace_on(
                blueprint.name, query_router.name, day_idx=day_idx
            )
            features = self.featurizer.featurize_trace(workload_trace)

            predicted_tail_latency = np.expm1(
                self.model.predict(np.array(features).reshape(1, -1))
            )

            diff = predicted_tail_latency - latency_slo_s
            diffs_from_slo.append(diff)

        # If any blueprint is predicted to meet the SLO, select the one with
        # the lowest RPU. The are in increasing RPU order.
        for i, diff in enumerate(diffs_from_slo):
            if diff <= 0:
                return options[i]

        # If no blueprint is predicted to meet the SLO, return the one with
        # the smallest violation.
        min_diff_idx = diffs_from_slo.index(min(diffs_from_slo))
        return options[min_diff_idx]
