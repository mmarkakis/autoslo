import os
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

import autoslo.filesystem.path_utils as pu
from autoslo.clusters.cluster import Cluster
from autoslo.models.model_prediction import ModelPrediction
from autoslo.workload_definition.query import ClusterAwareQueryId, QueryTextId
from autoslo.workload_execution.trace import Trace


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
        ignore_cluster_size: bool = False,
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
            ignore_cluster_size: Whether to ignore the cluster RPU associated
                with each query, both in training and at inference time.
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
        self._ignore_cluster_size = ignore_cluster_size

    def predict_from_query_text_id(
        self,
        query_text_ids: dict[str, QueryTextId],
        cluster_rpu: int,
    ) -> dict[str, Optional[ModelPrediction]]:
        """
        Predicts the runtime of the given queries, based on their
        :class:`~autoslo.workload_definition.query.QueryTextId`.

        Parameters:
            query_text_ids: A dictionary mapping query ids to
                :class:`~autoslo.workload_definition.query.QueryTextId` objects.
            cluster_rpu: The RPU size of the target cluster.

        Returns:
            A dictionary mapping cluster aware query ids to ModelPrediction
                instances, where each element is in seconds.
        """
        predictions: dict[str, Optional[ModelPrediction]] = {}

        effective_rpu = 0 if self._ignore_cluster_size else cluster_rpu
        cache_for_rpu = self._cache.get(effective_rpu, {})

        for query_id, query_text_id in query_text_ids.items():
            template_id = int(query_text_id.template_id)
            query_within_template_id = int(query_text_id.query_index)

            if (template_id not in cache_for_rpu) or (
                (query_within_template_id not in cache_for_rpu[template_id])
                and not self._enable_template_cache
            ):
                # Cache miss
                if self._best_effort:
                    predictions[query_id] = ModelPrediction(
                        mean_s=[self._mean_runtime_s_for_rpu[effective_rpu]],
                        std_dev_s=[self._std_runtime_s_for_rpu[effective_rpu]],
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
        use_client_side_latencies: bool = False,
    ) -> None:
        """
        Trains the model on the given run IDs.

        Parameters:
            run_ids: The run IDs to train the model on.
            from_scratch: Whether to train the model from scratch, or
                continue training from the existing model.
            only_non_overlapping_queries: Whether to only use train on queries
                that do not overlap with any other queries in the trace.
            use_client_side_latencies: Use client-side latencies from the
                structured log instead of Redshift server-side elapsed_time.
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
            latencies = (
                trace.client_side_latencies_s
                if use_client_side_latencies
                else trace.server_side_latencies_s
            )
            query_text_ids = trace.query_text_ids
            query_is_non_overlapping = trace.query_is_non_overlapping()

            for (cluster_aware_query_id, latency), query_text_id in zip(
                latencies.items(), query_text_ids
            ):
                if (
                    only_non_overlapping_queries
                    and not query_is_non_overlapping[cluster_aware_query_id]
                ):
                    continue

                cluster_name = ClusterAwareQueryId(
                    cluster_aware_query_id
                ).cluster_name

                cluster_rpu = (
                    0
                    if self._ignore_cluster_size
                    else Cluster.rpu_for_cluster_name(cluster_name)
                )
                template_id = int(query_text_id.template_id)
                query_within_template_id = int(query_text_id.query_index)

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
                    "ignore_cluster_size": self._ignore_cluster_size,
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
            ignore_cluster_size=params.get("ignore_cluster_size", False),
        )
        model._run_ids = params["run_ids"]
        model._mean_runtime_s_for_rpu = params["mean_runtime_s_for_rpu"]
        model._std_runtime_s_for_rpu = params["std_runtime_s_for_rpu"]

        # Load the model itself
        cache_yml_path = os.path.join(load_dir, "model.yml")
        with open(cache_yml_path, "r") as f:
            model._cache = yaml.safe_load(f)

        return model
