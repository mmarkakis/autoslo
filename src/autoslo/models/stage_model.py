import os
from datetime import datetime
from typing import Any, Optional

import yaml

import autoslo.utils.paths as pu
from autoslo.models.cache_model import CacheModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.models.xgboost_model import XGBoostModel
from autoslo.workload_definition.query import QueryTextId
from autoslo.workload_execution.trace import Trace


class StageModel:
    """
    A Stage model for predicting query runtime.
    """

    def __init__(
        self,
        cache_model_id: Optional[str] = None,
        cache_model_init_params: Optional[dict[str, Any]] = None,
        cache_model_train_params: Optional[dict[str, Any]] = None,
        xgboost_model_id: Optional[str] = None,
        xgboost_model_init_params: Optional[dict[str, Any]] = None,
        xgboost_model_train_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the model.

        Parameters:
            cache_model_id: The ID of a pre-trained CacheModel to load.
            cache_model_init_params: The initialization parameters for the
                CacheModel, if cache_model_id is not provided.
            cache_model_train_params: The training parameters for the
                CacheModel, if cache_model_id is not provided.
            xgboost_model_id: The ID of a pre-trained XGBoostModel to load.
            xgboost_model_init_params: The initialization parameters for the
                XGBoostModel, if xgboost_model_id is not provided.
            xgboost_model_train_params: The training parameters for the
                XGBoostModel, if xgboost_model_id is not provided.

        Raises:
            ValueError: If, for either sub-model, neither a model ID nor
                initialization and training parameters are provided.
        """

        # Initialize CacheModel
        if cache_model_id is not None:
            self._cache_model_id = cache_model_id
            self._cache_model = CacheModel.load(cache_model_id)
        elif (
            cache_model_init_params is not None
            and cache_model_train_params is not None
        ):
            self._cache_model = CacheModel(**cache_model_init_params)
            self._cache_model.train(**cache_model_train_params)
            self._cache_model_id = self._cache_model.save()
        else:
            raise ValueError(
                "Either cache_model_id or both "
                "cache_model_init_params and "
                "cache_model_train_params must be provided."
            )

        # Initialize XGBoostModel
        if xgboost_model_id is not None:
            self._xgboost_model_id = xgboost_model_id
            self._xgboost_model = XGBoostModel.load(xgboost_model_id)
        elif (
            xgboost_model_init_params is not None
            and xgboost_model_train_params is not None
        ):
            self._xgboost_model = XGBoostModel(**xgboost_model_init_params)
            self._xgboost_model.train(**xgboost_model_train_params)
            self._xgboost_model_id = self._xgboost_model.save()
        else:
            raise ValueError(
                "Either xgboost_model_id or both "
                "xgboost_model_init_params and "
                "xgboost_model_train_params must be provided."
            )

        # Memoization cache: (query_text_id, cluster_name) -> ModelPrediction.
        # Both sub-models are deterministic functions of these two inputs, so
        # any repeated call for the same pair is a free dict lookup. The
        # vocabulary is finite (~800 entries), so the cache saturates quickly.
        self._prediction_cache: dict[
            tuple[QueryTextId, str], ModelPrediction
        ] = {}

    def predict(
        self,
        query_texts: dict[str, str],
        cluster_name: str,
        schema_name: str,
    ) -> dict[str, ModelPrediction]:
        """
        Predicts the runtime of the given query texts.

        Parameters:
            query_texts: The query texts to predict the runtime of, as a
                dictionary mapping query ids to query texts.
            cluster_name: The name of the cluster where the queries will be run.
            schema_name: The name of the schema the queries belong to.

        Returns:
            A dictionary mapping query ids to ModelPrediction instances,
                where each element is in seconds.
        """
        query_text_ids = {
            query_id: Trace.extract_query_text_id(query_text, schema_name)
            for query_id, query_text in query_texts.items()
        }
        return self.predict_from_query_text_id(query_text_ids, cluster_name)

    def predict_from_query_text_id(
        self, query_text_ids: dict[str, QueryTextId], cluster_name: str
    ) -> dict[str, ModelPrediction]:
        """
        Predicts the runtime of the given queries, based on their
        :class:`~autoslo.workload_definition.query.QueryTextId`.

        Parameters:
            query_text_ids: A dictionary mapping query ids to
                :class:`~autoslo.workload_definition.query.QueryTextId` objects.
            cluster_name: The name of the cluster where the queries will be run.

        Returns:
            A dictionary mapping query ids to ModelPrediction instances,
                where each element is in seconds.
        """
        overall_predictions: dict[str, ModelPrediction] = {}

        # Check memoization cache first; compute only for unseen pairs.
        remaining_query_text_ids: dict[str, QueryTextId] = {}
        for query_id, query_text_id in query_text_ids.items():
            cached = self._prediction_cache.get((query_text_id, cluster_name))
            if cached is not None:
                overall_predictions[query_id] = cached
            else:
                remaining_query_text_ids[query_id] = query_text_id

        if not remaining_query_text_ids:
            return overall_predictions

        # Process cache model first
        cache_predictions = self._cache_model.predict_from_query_text_id(
            remaining_query_text_ids,
            cluster_name=cluster_name,
        )
        xgboost_remaining: dict[str, QueryTextId] = {}
        for query_id, prediction in cache_predictions.items():
            if prediction is not None:
                overall_predictions[query_id] = prediction
                self._prediction_cache[
                    (remaining_query_text_ids[query_id], cluster_name)
                ] = prediction
            else:
                xgboost_remaining[query_id] = remaining_query_text_ids[
                    query_id
                ]

        # Use XGBoost only for the queries that were not cache hits.
        xgboost_predictions = (
            self._xgboost_model.predict_from_query_text_id(
                xgboost_remaining,
                cluster_name=cluster_name,
            )
        )
        for query_id, prediction in xgboost_predictions.items():
            overall_predictions[query_id] = prediction
            self._prediction_cache[
                (xgboost_remaining[query_id], cluster_name)
            ] = prediction

        return overall_predictions

    def save(self, parent_save_dir: Optional[str] = None) -> str:
        """
        Saves the StageModel.

        Parameters:
            parent_save_dir: The parent directory where stage models are stored.
                If None, defaults to `data/stage_models/`.
        Returns:
            The identifier of the saved StageModel. This is a subdirectory under
                the parent_save_dir named after the current timestamp.
        """
        # Create directory.
        if parent_save_dir is None:
            parent_save_dir = os.path.join(pu.get_data_path(), "stage_models")
        timestamp = str(int(datetime.now().timestamp()))
        save_dir = os.path.join(
            parent_save_dir,
            timestamp,
        )
        os.makedirs(save_dir, exist_ok=False)

        # Save stage model parameters
        param_path = os.path.join(save_dir, "params.yml")
        with open(param_path, "w") as f:
            yaml.safe_dump(
                {
                    "cache_model_id": self._cache_model_id,
                    "xgboost_model_id": self._xgboost_model_id,
                },
                f,
            )

        return save_dir

    @staticmethod
    def load(
        timestamp: str, parent_load_dir: Optional[str] = None
    ) -> "StageModel":
        """
        Loads the model from the given directory.

        Parameters:
            timestamp: The identifier of the saved StageModel to load.
            parent_load_dir: The parent directory where stage models are stored.
                If None, defaults to `data/stage_models/`.

        Returns:
            The loaded StageModel.

        Raises:
            ValueError: If the specified directory does not exist.
        """
        if parent_load_dir is None:
            parent_load_dir = os.path.join(pu.get_data_path(), "stage_models")
        load_dir = os.path.join(parent_load_dir, timestamp)
        if not os.path.exists(load_dir):
            raise ValueError(f"StageModel directory {load_dir} does not exist.")

        # Load model parameters
        param_path = os.path.join(load_dir, "params.yml")
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)

        return StageModel(
            cache_model_id=params["cache_model_id"],
            xgboost_model_id=params["xgboost_model_id"],
        )
