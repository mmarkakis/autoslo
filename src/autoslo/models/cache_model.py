import os
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

import autoslo.utils.paths as pu
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_execution.trace import Trace

from autoslo.blueprints.cluster import Cluster

from collections import defaultdict


class CacheModel:
    """
    A query runtime model that simply caches the past runtimes of each query,
    based on their TPC-DS template and query index. For cache hits, the model
    returns the mean and standard deviation of the cached runtimes for the query
    at hand. For cache misses, the model generally returns None.

    If template caching is enabled, unseen queries from seen templates will be
    predicted as the mean and standard deviation of the seen queries of the
    template.

    If best effort is enabled, cache misses will return the overall mean and
    standard deviation of all cached runtimes.
    """

    def __init__(
        self,
        enable_template_cache: bool = False,
        best_effort: bool = False,
    ) -> None:
        """
        Initializes the CacheModel.

        Parameters:
            enable_template_cache: Whether to enable the template cache.
                If enabled, unseen queries from seen templates will be predicted
                as the mean and standard deviation of the seen queries of the
                template.
            best_effort: Whether to enable best-effort predictions for cache
                misses. If enabled, cache misses will return the overall mean
                and standard deviation of all cached runtimes on a cluster of
                the same size.
        """
        # key: cluster RPU
        # value: dictionary where
        #   key: tpcds template index
        #   value: dictionary where
        #       key: index of query with that template
        #       value: (list of runtimes, mean runtime, std runtime)
        self._cache: dict[
            int,
            dict[
                int,
                dict[int, tuple[list[float], float, float]],
            ],
        ] = {}
        self._enable_template_cache = enable_template_cache
        self._best_effort = best_effort
        self._run_ids: list[str] = []
        self._mean_runtime_s_for_rpu: dict[int, float] = defaultdict(float)
        self._std_runtime_s_for_rpu: dict[int, float] = defaultdict(float)

        self._only_non_overlapping_queries = False

    def predict(
        self, query_texts: dict[str, str], cluster_name: str
    ) -> dict[str, Optional[ModelPrediction]]:
        """
        Predicts the runtime of the given query texts.

        Parameters:
            query_texts: The query texts to predict the runtime of, as a
                dictionary mapping query ids to query texts.
            cluster_name: The name of the cluster where the queries will be run.

        Returns:
            A dictionary mapping query ids to ModelPrediction instances,
                where each element is in seconds.
        """
        query_temp_and_q_idxs = {
            query_id: Trace.extract_temp_and_q_idxs(query_text)
            for query_id, query_text in query_texts.items()
        }
        return self.predict_from_tpcds_temp_and_q_idx(
            query_temp_and_q_idxs, cluster_name
        )

    def predict_from_tpcds_temp_and_q_idx(
        self,
        query_temp_and_q_idxs: dict[str, Trace.TPCDSTempAndQIdx],
        cluster_name: str,
    ) -> dict[str, Optional[ModelPrediction]]:
        """
        Predicts the runtime of the given queries, based on their TPC-DS
        template and query indices.

        Parameters:
            query_temp_and_q_idxs: The TPC-DS template and query indices of
                the queries to predict the runtime of, as a dictionary mapping
                query ids to TPC-DS template and query indices.
            cluster_name: The name of the cluster where the queries will be run.

        Returns:
            A dictionary mapping query ids to ModelPrediction instances,
                where each element is in seconds.
        """
        predictions: dict[str, Optional[ModelPrediction]] = {}

        cluster_rpu = Cluster.rpu_for_cluster_name(cluster_name)
        cache_for_rpu = self._cache.get(cluster_rpu, {})

        for query_id, temp_and_q_idx in query_temp_and_q_idxs.items():
            template_id = Trace.extract_temp(temp_and_q_idx)
            query_within_template_id = Trace.extract_q_idx(temp_and_q_idx)

            if (template_id not in cache_for_rpu) or (
                (query_within_template_id not in cache_for_rpu[template_id])
                and not self._enable_template_cache
            ):
                # Cache miss
                if self._best_effort:
                    predictions[query_id] = ModelPrediction(
                        mean_s=[self._mean_runtime_s_for_rpu[cluster_rpu]],
                        std_dev_s=[self._std_runtime_s_for_rpu[cluster_rpu]],
                    )
                else:
                    predictions[query_id] = None
            elif (
                query_within_template_id not in cache_for_rpu[template_id]
            ) and self._enable_template_cache:
                # Template cache hit
                runtimes = []
                for _, (local_runtimes, _, _) in cache_for_rpu[
                    template_id
                ].items():
                    runtimes.extend(local_runtimes)
                predictions[query_id] = ModelPrediction(
                    mean_s=[float(np.mean(runtimes))],
                    std_dev_s=[float(np.std(runtimes, ddof=0))],
                )
            elif query_within_template_id in cache_for_rpu[template_id]:
                # Cache hit
                _, mean_runtime, std_runtime = cache_for_rpu[template_id][
                    query_within_template_id
                ]
                predictions[query_id] = ModelPrediction(
                    mean_s=[mean_runtime],
                    std_dev_s=[std_runtime],
                )

        return predictions

    def train(
        self,
        run_ids: list[str],
        from_scratch: bool = False,
        only_non_overlapping_queries: bool = False,
    ) -> None:
        """
        Trains the model on the given run IDs.

        Parameters:
            run_ids: The run IDs to train the model on.
            from_scratch: Whether to train the model from scratch, or
                continue training from the existing model.
            only_non_overlapping_queries: Whether to only use train on queries
                that do not overlap with any other queries in the trace.
        """

        # If retraining from scratch, reset the cache.
        if from_scratch:
            self._cache = {}
            self._overall_mean_runtime_s = 0.0
            self._overall_std_runtime_s = 0.0
            self._run_ids = []

        self._only_non_overlapping_queries = only_non_overlapping_queries

        for run_id in run_ids:
            trace = Trace(run_id)
            latencies = trace.latencies_s
            temp_and_q_idxs = trace.tpcds_temp_and_q_idxs
            query_is_non_overlapping = trace.query_is_non_overlapping()

            for (query_id, latency), temp_and_q_idx in zip(
                latencies.items(), temp_and_q_idxs
            ):
                if (
                    only_non_overlapping_queries
                    and not query_is_non_overlapping[query_id]  # type: ignore
                ):
                    continue

                cluster_name = trace.cluster_name_from_query_id(
                    query_id  # type: ignore
                )
                cluster_rpu = Cluster.rpu_for_cluster_name(cluster_name)
                template_id = Trace.extract_temp(temp_and_q_idx)
                query_within_template_id = Trace.extract_q_idx(temp_and_q_idx)

                if cluster_rpu not in self._cache:
                    self._cache[cluster_rpu] = {}
                if template_id not in self._cache[cluster_rpu]:
                    self._cache[cluster_rpu][template_id] = {}
                if (
                    query_within_template_id
                    not in self._cache[cluster_rpu][template_id]
                ):
                    self._cache[cluster_rpu][template_id][
                        query_within_template_id
                    ] = (
                        [],
                        0.0,
                        0.0,
                    )

                runtimes, _, _ = self._cache[cluster_rpu][template_id][
                    query_within_template_id
                ]
                runtimes.append(latency)
                self._cache[cluster_rpu][template_id][
                    query_within_template_id
                ] = (
                    runtimes,
                    float(np.mean(runtimes)),
                    float(np.std(runtimes, ddof=0)),
                )

        # Update overall mean and standard deviation
        for cluster_rpu, template_dict in self._cache.items():
            runtimes_for_rpu = []
            for query_dict in template_dict.values():
                for runtimes, _, _ in query_dict.values():
                    runtimes_for_rpu.extend(runtimes)
            self._mean_runtime_s_for_rpu[cluster_rpu] = float(
                np.mean(runtimes_for_rpu)
            )
            self._std_runtime_s_for_rpu[cluster_rpu] = float(
                np.std(runtimes_for_rpu, ddof=0)
            )

        self._run_ids.extend(run_ids)
        self._run_ids = list(set(self._run_ids))

    def save(self, parent_save_dir: Optional[str] = None) -> str:
        """
        Saves the CacheModel.

        Parameters:
            parent_save_dir: The parent directory where cache models are stored.
                If None, defaults to `data/cache_models/`.

        Returns:
            The identifier of the saved CacheModel. This is a subdirectory under
                the parent_save_dir named after the current timestamp.
        """
        # Create directory.
        if parent_save_dir is None:
            parent_save_dir = os.path.join(pu.get_data_path(), "cache_models")
        timestamp = str(int(datetime.now().timestamp()))
        save_dir = os.path.join(
            parent_save_dir,
            timestamp,
        )
        os.makedirs(save_dir, exist_ok=False)

        # Save cache model parameters
        param_path = os.path.join(save_dir, "params.yml")
        with open(param_path, "w") as f:
            yaml.safe_dump(
                {
                    "enable_template_cache": self._enable_template_cache,
                    "best_effort": self._best_effort,
                    "only_non_overlapping_queries": self._only_non_overlapping_queries,
                    "run_ids": self._run_ids,
                    "mean_runtime_s_for_rpu": dict(
                        self._mean_runtime_s_for_rpu
                    ),
                    "std_runtime_s_for_rpu": dict(self._std_runtime_s_for_rpu),
                },
                f,
            )

        # Save the model itself
        cache_pkl_path = os.path.join(save_dir, "model.pkl")
        with open(cache_pkl_path, "wb") as f:
            pickle.dump(self._cache, f)
        cache_yml_path = os.path.join(save_dir, "model.yml")
        with open(cache_yml_path, "w") as f:
            yaml.safe_dump(self._cache, f)

        return timestamp

    @staticmethod
    def load(
        timestamp: str, parent_load_dir: Optional[str] = None
    ) -> "CacheModel":
        """
        Loads the model from the given directory.

        Parameters:
            timestamp: The identifier of the saved CacheModel to load.
            parent_load_dir: The parent directory where cache models are stored.
                If None, defaults to `data/cache_models/`.

        Returns:
            The loaded CacheModel.

        Raises:
            ValueError: If the specified directory does not exist.
        """
        if parent_load_dir is None:
            parent_load_dir = os.path.join(pu.get_data_path(), "cache_models")
        load_dir = os.path.join(
            parent_load_dir,
            timestamp,
        )
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"CacheModel {timestamp} not found")

        # Load model parameters
        param_path = os.path.join(load_dir, "params.yml")
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)
        model = CacheModel(
            enable_template_cache=params["enable_template_cache"],
            best_effort=params["best_effort"],
        )
        model._run_ids = params["run_ids"]
        model._mean_runtime_s_for_rpu = params["mean_runtime_s_for_rpu"]
        model._std_runtime_s_for_rpu = params["std_runtime_s_for_rpu"]

        # Load the model itself
        cache_pkl_path = os.path.join(load_dir, "model.pkl")
        with open(cache_pkl_path, "rb") as f:
            model._cache = pickle.load(f)

        return model
