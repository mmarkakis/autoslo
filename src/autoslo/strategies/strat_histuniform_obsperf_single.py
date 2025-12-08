import os

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.blueprint_timeseries import BlueprintTimeseries
from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.featurizer import Featurizer
from autoslo.routing.query_router import QueryRouter
from autoslo.routing.r_fixed import RFixed
from autoslo.strategies.slo_strategy import SLOStrategy
from autoslo.strategies.slo_strategy_performance import (
    E2ESLOMetrics,
    SLOStrategyPerformance,
)
from autoslo.workload_definition.composite import Composite


class StratHistUniformObsPerfSingle(SLOStrategy):
    """
    SLO strategy that uses past data from the most recent periods to
    predict and select the best single-cluster blueprint for the next period.

    For each day of past data, we assume access to the observed performance
    metrics on one single-cluster blueprint, and use a model to predict
    performance on the remaining single-cluster blueprints.
    """

    def __init__(
        self,
        window_size: int,
        slo_violation_rate_threshold: float,
        model_training_run_id: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the StratHistUniformObsPerfSingle strategy.

        Parameters:
            window_size: The number of past periods to consider for prediction.
            slo_violation_rate_threshold: The acceptable SLO violation rate
                threshold. SLO violation rates at or below this threshold are
                considered acceptable.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).
        """
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.violation_rate_threshold = slo_violation_rate_threshold

        # Initialize and load the model.
        self.model = xgb.XGBRegressor()
        model_dir = os.path.join(pu.get_models_dir(), model_training_run_id)
        model_path = os.path.join(model_dir, "model.json")
        self.model.load_model(model_path)

        # Read the model config and initialize the featurizer.
        training_params_path = os.path.join(model_dir, f"training_params.yml")
        with open(training_params_path, "r") as f:
            tp = yaml.safe_load(f)
        featurization_params = tp.get("featurization_params", None)
        if featurization_params is None:
            raise ValueError(
                "Model training parameters file is missing "
                "'featurization_params' entry."
            )
        self.featurizer = Featurizer.from_name(
            featurization_params["featurizer_name"], **featurization_params
        )

    def suggest(
        self,
        workload: Composite,
        day_idx: int,
        latency_slo_s: float,
        past_blueprints: BlueprintTimeseries,
        *args,
        **kwargs,
    ) -> tuple[Blueprint, QueryRouter]:
        """
        Base suggestions on how each single-cluster blueprint would have
        performed over the past `window_size` periods.

        Parameters:
            workload: The composite workload to predict for.
            day_idx: The index of the day to make suggestions for.
            latency_slo_s: The latency SLO in seconds.
            past_blueprints: The blueprints used in the past periods.
            args: Positional arguments (as needed by specific strategies).
            kwargs: Keyword arguments (as needed by specific strategies).

        Returns:
            A tuple containing the selected Blueprint and its associated
                QueryRouter.

        Raises:
            ValueError: If past_blueprints is missing entries for any of the
                required past days.
        """
        options: list[tuple[Blueprint, QueryRouter]] = []
        diffs_from_slo: list[dict[str, float]] = []

        # Make sure that `past_blueprints` has entries for required past days.
        for past_day_idx in range(max(0, day_idx - self.window_size), day_idx):
            try:
                _ = past_blueprints.blueprint_for_period(past_day_idx)
            except KeyError as e:
                raise ValueError(
                    f"Past blueprints for day index {day_idx} are missing an"
                    f"entry for day index {past_day_idx}, but it is required "
                    f"by window size {self.window_size}."
                ) from e

        # For each candidate single-cluster blueprint, evaluate its past
        # performance over each day in the specified window size.
        for rpu in Cluster.all_allowed_rpu_sizes():
            blueprint = Blueprint.one_cluster_with(rpu)
            query_router = RFixed(blueprint, blueprint.cluster_names[0])
            options.append((blueprint, query_router))

            for past_day_idx in range(
                max(0, day_idx - self.window_size), day_idx
            ):
                # Is this was the observed blueprint for this period?
                if (
                    past_blueprints.blueprint_for_period(past_day_idx)
                    == blueprint
                ):
                    # If yes, evaluate how it would have performed by co-opting
                    # evaluate_suggestion.
                    past_perf = self.evaluate_suggestion(
                        workload,
                        past_day_idx,
                        latency_slo_s,
                        blueprint,
                        query_router,
                    )
                    predicted_tail_latency = past_perf.latency_s_at_quantile(
                        1 - self.violation_rate_threshold
                    )
                else:
                    # If not, use the model to predict its performance.
                    workload_trace = workload.get_most_recent_trace_on(
                        blueprint.name, query_router.name, day_idx=past_day_idx
                    )
                    features, _ = self.featurizer.featurize_trace(workload_trace)

                    predicted_tail_latency = np.expm1(
                        self.model.predict(np.array(features).reshape(1, -1))
                    )

                # Compute the difference from the SLO.
                diff = predicted_tail_latency - latency_slo_s
                d = {
                    "rpu": rpu,
                    "options_idx": len(options) - 1,
                    "day_idx": past_day_idx,
                    "predicted_tail_latency": float(predicted_tail_latency),
                    "diff_from_slo": float(diff),
                }
                diffs_from_slo.append(d)

        # Coalesce performance metrics
        diffs_from_slo_df = pd.DataFrame(diffs_from_slo)
        grouped = diffs_from_slo_df.groupby("rpu").agg(
            options_idx=("options_idx", "first"),
            diff_from_slo_flag=(
                "diff_from_slo",
                lambda x: all(v <= 0 for v in x),
            ),
            diff_from_slo_mean=("diff_from_slo", "mean"),
        )

        # If any blueprint is predicted to meet the SLO every day, select the
        # one with the lowest RPU.
        meeting_rpu = grouped[grouped["diff_from_slo_flag"]].index.tolist()
        if meeting_rpu:
            best_rpu = min(meeting_rpu)
            best_options_idx = grouped[grouped.index == best_rpu][
                "options_idx"
            ].values[0]
            return options[best_options_idx]

        # If no blueprint is predicted to meet the SLO, return the one with
        # the smallest violation.
        min_diff_rpu = grouped["diff_from_slo_mean"].idxmin()
        best_options_idx = grouped.loc[min_diff_rpu, "options_idx"]
        return options[best_options_idx]


class StratHistUniformObsPerfSingle1(StratHistUniformObsPerfSingle):
    """
    A StratHistUniformObsPerfSingle strategy that uses a past window size of 1.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        model_training_run_id: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            window_size=1,
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            model_training_run_id=model_training_run_id,
        )


class StratHistUniformObsPerfSingle7(StratHistUniformObsPerfSingle):
    """
    A StratHistUniformObsPerfSingle strategy that uses a past window size of 7.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        model_training_run_id: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            window_size=7,
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            model_training_run_id=model_training_run_id,
        )


class StratHistUniformObsPerfSingle14(StratHistUniformObsPerfSingle):
    """
    A StratHistUniformObsPerfSingle strategy that uses a past window size of 14.
    """

    def __init__(
        self,
        slo_violation_rate_threshold: float,
        model_training_run_id: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            window_size=14,
            slo_violation_rate_threshold=slo_violation_rate_threshold,
            model_training_run_id=model_training_run_id,
        )
