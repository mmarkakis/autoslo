import os
from datetime import datetime
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

import autoslo.utils.paths as pu
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_definition.query import QueryTextId
from autoslo.workload_execution.trace import Trace


class XGBoostModel:
    """
    A query runtime model that uses XGBoost to predict the runtime of a query
    based on its features.
    """

    def __init__(
        self,
        train_on_log_runtime: bool = False,
        n_estimators: int = 1000,
        max_depth: int = 8,
        eta: float = 0.2,
        eval_metric: str = "mae",
        early_stopping_rounds: int = 100,
        random_seed: int = 42,
        iconq_query_featurizer_id: Optional[tuple[str, str]] = None,
        iconq_query_featurizer_init_params: Optional[dict[str, Any]] = None,
        ignore_cluster_size: bool = False,
    ):
        """
        Initializes the XGBoostModel.

        Parameters:
            train_on_log_runtime: Whether to train on the log of the runtime, as
                opposed to the runtime itself.
            n_estimators: Number of boosting rounds (number of gradient boosted
                trees).
            max_depth: The maximum depth of the trees.
            eta: The learning rate.
            eval_metric: The evaluation metric to use.
            early_stopping_rounds: The number of rounds with no improvement to
                wait before stopping.
            random_seed: The random seed to use for training.
            iconq_query_featurizer_id: The identifier of the
                IconqQueryFeaturizer to use for featurizing queries. If not
                provided, must provide iconq_query_featurizer_init_params, with
                appropriate keys, to initialize a new IconqQueryFeaturizer.
            iconq_query_featurizer_init_params: The initialization parameters
                for the IconqQueryFeaturizer, if iconq_query_featurizer_id is
                not provided. Must include a key for each required parameter of
                the constructor of IconqQueryFeaturizer.
            ignore_cluster_size: Whether to ignore the cluster size when
                featurizing queries. If True, the cluster size feature will be
                zeroed out for all queries.
        Raises:
            ValueError: If neither iconq_query_featurizer_id nor
                iconq_query_featurizer_init_params is provided.
        """

        if iconq_query_featurizer_id is None:
            if iconq_query_featurizer_init_params is None:
                raise ValueError(
                    "Must provide either iconq_query_featurizer_id or "
                    "iconq_query_featurizer_init_params."
                )
            self._iconq_query_featurizer = IconqQueryFeaturizer(
                **iconq_query_featurizer_init_params
            )
            self._iconq_query_featurizer_id = (
                self._iconq_query_featurizer.save()
            )
        else:
            self._iconq_query_featurizer_id = iconq_query_featurizer_id
            self._iconq_query_featurizer = IconqQueryFeaturizer.load(
                schema_name="ext_tpcds1000",  # TODO: pass schema_name in as a parameter instead of hardcoding
                timestamp=iconq_query_featurizer_id,
            )

        self._train_on_log_runtime = train_on_log_runtime
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._eta = eta
        self._eval_metric = eval_metric
        self._early_stopping_rounds = early_stopping_rounds
        self._model = XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            eta=self._eta,
            subsample=1.0,
            eval_metric=self._eval_metric,
            early_stopping_rounds=self._early_stopping_rounds,
        )
        self._random_seed = random_seed

        self._only_non_overlapping_queries = False
        self._ignore_cluster_size = ignore_cluster_size

    def predict(
        self,
        query_texts: dict[str, str],
        cluster_rpu: int,
        schema_name: str,
    ) -> dict[str, ModelPrediction]:
        """
        Predicts the runtime of the given query texts.

        Parameters:
            query_texts: The query texts to predict the runtime of, as a
                dictionary mapping query ids to query texts.
            cluster_rpu: The RPU size of the target cluster.
            schema_name: The name of the schema the queries belong to.

        Returns:
            A dictionary mapping query ids to ModelPrediction instances,
                where each element is in seconds.
        """
        query_text_ids = {
            query_id: Trace.extract_query_text_id(query_text, schema_name)
            for query_id, query_text in query_texts.items()
        }
        return self.predict_from_query_text_id(query_text_ids, cluster_rpu)

    def predict_from_query_text_id(
        self,
        query_text_ids: dict[str, QueryTextId],
        cluster_rpu: int,
    ) -> dict[str, ModelPrediction]:
        """
        Predicts the runtime of the given queries, based on their
        :class:`~autoslo.workload_definition.query.QueryTextId`.

        Parameters:
            query_text_ids: A dictionary mapping query ids to
                :class:`~autoslo.workload_definition.query.QueryTextId` objects.
            cluster_rpu: The RPU size of the target cluster.

        Returns:
            A dictionary mapping query ids to ModelPrediction instances,
                where each element is in seconds.
        """
        predictions: dict[str, ModelPrediction] = {}
        effective_rpu = 0 if self._ignore_cluster_size else cluster_rpu

        for query_id, query_text_id in query_text_ids.items():
            featurization = (
                self._iconq_query_featurizer.featurize_from_query_text_id(
                    query_text_id
                ).copy()
            )
            if featurization is None:
                raise ValueError(
                    f"Query {query_id} could not be featurized using "
                    f"IconqQueryFeaturizer "
                    f"{self._iconq_query_featurizer_id}."
                )
            featurization.append(effective_rpu)
            featurization_array = np.array(featurization).reshape(1, -1)
            raw_prediction = self._model.predict(featurization_array)[0]
            if self._train_on_log_runtime:
                raw_prediction = np.exp(raw_prediction)
            predictions[query_id] = ModelPrediction(mean_s=[raw_prediction])

        return predictions

    def train(
        self,
        run_ids: list[str],
        parent_save_dir: Optional[str] = None,
        only_non_overlapping_queries: bool = False,
    ) -> tuple[float, float]:
        """
        Trains the model on the given run IDs.

        Parameters:
            run_ids: The run IDs to train the model on.
            parent_save_dir: The parent directory where xgboost models are
                stored. If None, defaults to `data/xgboost_models/`.
            only_non_overlapping_queries: Whether to only use train on queries
                that do not overlap with any other queries in the trace.

        Returns:
            A tuple containing the final training and validation loss.
        """

        # Create directory.
        if parent_save_dir is None:
            parent_save_dir = os.path.join(pu.get_data_path(), "xgboost_models")
        self._run_id = str(int(datetime.now().timestamp()))
        self._save_dir = os.path.join(
            parent_save_dir,
            self._run_id,
        )
        os.makedirs(self._save_dir, exist_ok=True)

        self._only_non_overlapping_queries = only_non_overlapping_queries

        # Save the XGBoostModel parameters.
        params_path = os.path.join(self._save_dir, "params.yml")
        with open(params_path, "w") as f:
            yaml.dump(
                {
                    "iconq_query_featurizer_id": (
                        self._iconq_query_featurizer_id
                    ),
                    "train_on_log_runtime": self._train_on_log_runtime,
                    "n_estimators": self._n_estimators,
                    "max_depth": self._max_depth,
                    "eta": self._eta,
                    "eval_metric": self._eval_metric,
                    "early_stopping_rounds": self._early_stopping_rounds,
                    "random_seed": self._random_seed,
                    "only_non_overlapping_queries": (
                        self._only_non_overlapping_queries
                    ),
                    "ignore_cluster_size": self._ignore_cluster_size,
                },
                f,
            )

        # Load featurizations and apply log transform if specified.
        l: list[dict[str, Any]] = []
        for run_id in run_ids:
            trace = Trace(run_id)
            featurizations = self._iconq_query_featurizer.featurize_trace(trace)
            latencies = trace.latencies_s
            query_is_non_overlapping = trace.query_is_non_overlapping()
            new_items = []

            for query_id in featurizations.keys():
                if (
                    only_non_overlapping_queries
                    and not query_is_non_overlapping[query_id]  # type: ignore
                ):
                    continue

                featurization = featurizations[query_id].copy()
                latency = latencies[query_id]
                if featurization is None or len(featurization) == 0:
                    continue
                cluster_name = trace.cluster_name_from_query_id(query_id)
                from autoslo.clusters.cluster import Cluster

                cluster_rpu = (
                    0
                    if self._ignore_cluster_size
                    else Cluster.rpu_for_cluster_name(cluster_name)
                )
                featurization.append(cluster_rpu)
                new_items.append(
                    {
                        "query_id": query_id,
                        "query_featurization": featurization,
                        "runtime_s": latency,
                    }
                )

            l.extend(new_items)

        featurization_df = pd.DataFrame(l)
        label_column_name = "runtime_s"
        if self._train_on_log_runtime:
            featurization_df["log_runtime_s"] = np.log(
                featurization_df["runtime_s"]
            )
            label_column_name = "log_runtime_s"

        # Reproducibly shuffle and split into training and validation sets.
        featurization_df = (
            featurization_df.sort_index()
            .sample(frac=1.0, random_state=self._random_seed)
            .reset_index(drop=True)
        )
        split_idx = int(0.8 * len(featurization_df))
        train_df = featurization_df.iloc[:split_idx]
        val_df = featurization_df.iloc[split_idx:]

        # Save them out.
        train_df_path = os.path.join(
            self._save_dir, "train_featurizations.parquet"
        )
        train_df.to_parquet(train_df_path)
        val_df_path = os.path.join(self._save_dir, "val_featurizations.parquet")
        val_df.to_parquet(val_df_path)

        # Train the model.
        X_train = np.stack(train_df["query_featurization"].to_list()).astype(
            np.float32
        )
        y_train = train_df[label_column_name].to_numpy()

        X_val = np.stack(val_df["query_featurization"].to_list()).astype(
            np.float32
        )
        y_val = val_df[label_column_name].to_numpy()
        self._model.fit(
            X_train,
            y_train,
            # We include the training set to `eval_set` in order to get
            # final_train_loss out. XGBoost only uses the last entry in this
            # list for early stopping so it doesn't affect the training.
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        # Get final training and validation losses.
        losses = self._model.evals_result()
        final_train_loss = losses["validation_0"][self._eval_metric][-1]
        final_val_loss = losses["validation_1"][self._eval_metric][-1]

        return final_train_loss, final_val_loss

    def save(self) -> str:
        """
        Saves the XGBoostModel.

        Returns:
            The identifier of the saved XGBoostModel.
        """

        # Save the model.
        model_json_path = os.path.join(self._save_dir, "model.json")
        self._model.save_model(model_json_path)

        # Also save the loss trajectories of the model as a plot.
        losses = self._model.evals_result()
        loss_plot_path = os.path.join(self._save_dir, "loss_plot.png")
        plt.figure()
        plt.plot(losses["validation_0"][self._eval_metric], label="Train Loss")
        plt.plot(
            losses["validation_1"][self._eval_metric], label="Validation Loss"
        )
        plt.xlabel("Iteration")
        plt.ylabel(self._eval_metric)
        plt.title("XGBoost Training and Validation Loss")
        plt.legend()
        plt.savefig(loss_plot_path)

        return self._run_id

    @staticmethod
    def load(
        timestamp: str, parent_load_dir: Optional[str] = None
    ) -> "XGBoostModel":
        """
        Loads the model from the given directory.

        Parameters:
            timestamp: The identifier of the saved XGBoostModel to load.
            parent_load_dir: The parent directory where xgboost models are
                stored. If None, defaults to `data/xgboost_models/`.
        """
        if parent_load_dir is None:
            parent_load_dir = os.path.join(pu.get_data_path(), "xgboost_models")
        load_dir = os.path.join(
            parent_load_dir,
            timestamp,
        )
        if not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"XGBoostModel directory {load_dir} does not exist."
            )

        # Load model parameters.
        params_path = os.path.join(load_dir, "params.yml")
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        model = XGBoostModel(
            iconq_query_featurizer_id=params["iconq_query_featurizer_id"],
            train_on_log_runtime=params["train_on_log_runtime"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            eta=params["eta"],
            eval_metric=params["eval_metric"],
            early_stopping_rounds=params["early_stopping_rounds"],
            random_seed=params["random_seed"],
            ignore_cluster_size=params.get("ignore_cluster_size", False),
        )

        # Load the model.
        model_json_path = os.path.join(load_dir, "model.json")
        model._model.load_model(model_json_path)

        return model
