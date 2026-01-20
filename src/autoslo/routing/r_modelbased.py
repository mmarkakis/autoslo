import os

import pandas as pd
import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.routing.query_router import QueryRouter

import numpy as np

import xgboost as xgb


class RModelBased(QueryRouter):
    """
    A QueryRouter implementation that routes queries based on a model-based
    approach, trained from a specific selector run.
    """

    def __init__(
        self,
        selector_run_id: str,
        iconq_query_featurizer_id: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize an RModelBased instance.

        Parameters:
            selector_run_id: The identifier for the selector run used to
                determine the sequence number to cluster mapping.
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Raises:
            FileNotFoundError: If the mapping file for the given selector_run_id
                does not exist.
        """
        # Retrieve the exact mapping from the selector run, as well as the
        # workload to which it refers.
        self._selector_run_id = selector_run_id
        mapping_path = os.path.join(
            pu.get_data_path(), "selector_runs", selector_run_id, "mapping.yml"
        )
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"Mapping file not found for selector_run_id "
                f"{selector_run_id} at path {mapping_path}."
            )
        with open(mapping_path, "r") as f:
            self._mapping: dict[int, str] = yaml.safe_load(f)
        config_path = os.path.join(
            pu.get_data_path(), "selector_runs", selector_run_id, "config.yml"
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self._workload_name: str = config["workload_name"]

        # Create a blueprint that includes all clusters in the mapping
        self._cluster_names = sorted(list(set(self._mapping.values())))
        self._blueprint = Blueprint(
            clusters=[Cluster.from_config(name) for name in self._cluster_names]
        )

        # Initialize the featurizer and then get the featurization of each query
        # from the workload.
        self._iconq_query_featurizer_id = iconq_query_featurizer_id
        self._featurizer = IconqQueryFeaturizer.load(
            self._iconq_query_featurizer_id
        )
        workload_path = os.path.join(
            pu.get_data_path(),
            "chunks",
            self._workload_name,
            "chunk_workload.parquet",
        )
        workload_df = pd.read_parquet(workload_path)
        workload_df["tpcds_temp_and_q_idx"] = workload_df.apply(
            lambda row: f"{row['query_template']:03d}_{row['query_num_within_template']:03d}",
            axis=1,
        )
        workload_df["featurization"] = workload_df[
            "tpcds_temp_and_q_idx"
        ].apply(self._featurizer.featurize_from_tpcds_temp_and_q_idx)
        workload_df["mapped_cluster"] = workload_df["query_id"].apply(
            lambda qid: self._mapping[qid]
        )
        self._featurizations = {
            row["query_id"]: row["featurization"]
            for _, row in workload_df.iterrows()
        }

        # For each query, compute an exponential moving average of the past queries that
        # the mapping assigned to its cluster. Weigh by the difference in rel
        # start times.
        self._query_to_prev_state = {}
        self._alpha = 0.7

        prev_states = {
            cluster_name: np.zeros(self._featurizer.num_dims)
            for cluster_name in self._cluster_names
        }

        for idx, row in workload_df.iterrows():

            query_id = row["query_id"]
            mapped_cluster_name = row["mapped_cluster"]

            self._query_to_prev_state[query_id] = np.copy(
                np.concatenate(
                    [
                        prev_states[cluster_name]
                        for cluster_name in self._cluster_names
                    ]
                )
            )

            prev_states[mapped_cluster_name] = (
                self._alpha * np.array(row["featurization"])
                + (1 - self._alpha) * prev_states[mapped_cluster_name]
            )

        # Now learn an xgboost ranker to correctly rank the clusters for each query,
        # so that the mapped cluster is ranked highest.
        training_rows = []

        for idx, row in workload_df.iterrows():
            query_id = row["query_id"]
            featurization = np.array(row["featurization"])

            prev_state = self._query_to_prev_state[query_id]

            combined_features = np.concatenate([featurization, prev_state])

            label = self._cluster_names.index(row["mapped_cluster"])

            training_rows.append(
                {
                    "query_id": query_id,
                    "features": combined_features,
                    "label": label,
                }
            )

        training_df = pd.DataFrame(training_rows)

        dtrain = xgb.DMatrix(
            np.vstack(training_df["features"].to_numpy()),
            label=training_df["label"].to_numpy(),
        )

        params = {
            "objective": "multi:softmax",
            "num_class": len(self._cluster_names),
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "max_depth": 5,
            "seed": 42,
        }
        self._model = xgb.train(
            params,
            dtrain,
            num_boost_round=5,
        )

        # Print accuracy on training set
        preds = self._model.predict(dtrain)
        workload_df["predicted_cluster"] = [
            self._cluster_names[int(pred)] for pred in preds
        ]
        accuracy = np.mean(preds == training_df["label"].to_numpy())
        print(f"RModelBased: Training accuracy: {accuracy:.4f}")

        self._workload_df = workload_df

        # Create variables to keep the running state during routing.
        self._prev_state_per_cluster = {
            cluster_name: np.zeros(self._featurizer.num_dims)
            for cluster_name in self._cluster_names
        }

    @property
    def workload_name(self) -> str:
        """
        Get the workload name associated with this RModelBased router.

        Returns:
            The workload name.
        """
        return self._workload_name

    @property
    def name(self) -> str:
        """
        Get the name of the RModelBased instance.
        """
        return (
            f"RModelBased(selector_run_id={repr(self._selector_run_id)}, "
            f"iconq_query_featurizer_id="
            f"{repr(self._iconq_query_featurizer_id)})"
        )

    @property
    def blueprint(self) -> Blueprint:
        """
        Get the Blueprint instance associated with this RModelBased router.

        Returns:
            The Blueprint instance.
        """
        return self._blueprint

    def route_query(self, tpcds_temp_and_q_idx, *args, **kwargs) -> str:
        """
        Route the query based on its featurization.

        Parameters:
            *args: Additional positional arguments, as needed.
            **kwargs: Additional keyword arguments, as needed.

        Returns:
            The cluster name to which the query should be routed.
        """

        # Find the cluster that ranks highest according to the model.
        featurization = self._featurizer.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx
        )

        # Prepare the input features for the model.
        X = np.concatenate(
            [
                np.array(featurization),
                np.concatenate(
                    [
                        self._prev_state_per_cluster[cluster_name]
                        for cluster_name in self._cluster_names
                    ]
                ),
            ]
        ).reshape(1, -1)

        dtest = xgb.DMatrix(X)
        pred_idx = self._model.predict(dtest)
        best_cluster = self._cluster_names[int(pred_idx)]

        # Update the previous state for the selected cluster.
        prev_state = self._prev_state_per_cluster[best_cluster]
        new_prev_state = (
            self._alpha * np.array(featurization)
            + (1 - self._alpha) * prev_state
        )
        self._prev_state_per_cluster[best_cluster] = new_prev_state

        return best_cluster
