from typing import Any, Optional

from autoslo.models.cache_model import CacheModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.models.xgboost_model import XGBoostModel


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

    def predict(
        self, query_texts: dict[str, str]
    ) -> dict[str, ModelPrediction]:
        """
        Predicts the runtime of the given query texts.

        Parameters:
            query_texts: The query texts to predict the runtime of, as a
                dictionary mapping query ids to query texts.

        Returns:
            A dictionary mapping query ids to ModelPrediction instances,
                where each element is in seconds.
        """
        overall_predictions: dict[str, ModelPrediction] = {}

        # Process cache model first
        cache_predictions = self._cache_model.predict(query_texts)
        remaining_query_texts = {}
        for query_id, prediction in cache_predictions.items():
            if prediction is not None:
                overall_predictions[query_id] = prediction
            else:
                remaining_query_texts[query_id] = query_texts[query_id]

        # Use XGBoost only for the queries that were not cache hits.
        xgboost_predictions = self._xgboost_model.predict(remaining_query_texts)
        overall_predictions |= xgboost_predictions

        return overall_predictions
