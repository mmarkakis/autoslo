import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

    MEAN_S_WITH_NO_INFO = 0.001

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
        #       value: bucket dict with exact and censored (lower-bound) data.
        self._cache: dict[
            int,
            dict[
                int,
                dict[int, dict[str, Any]],
            ],
        ] = {}
        self._enable_template_cache = enable_template_cache
        self._best_effort = best_effort
        self._run_ids: list[str] = []
        self._overall_mean_runtime_s: float = 0.0
        self._overall_std_runtime_s: float = 0.0
        self._mean_runtime_s_for_rpu: dict[int, float] = defaultdict(float)
        self._std_runtime_s_for_rpu: dict[int, float] = defaultdict(float)

        self._only_non_overlapping_queries = False
        self._ignore_cluster_size = ignore_cluster_size

    @staticmethod
    def _restricted_mean_std_from_km(
        exact_runtimes: list[float],
        lower_bounds: list[float],
    ) -> tuple[float, float]:
        """Estimate restricted mean/std from right-censored samples.

        This computes a Kaplan-Meier survival estimate and integrates it up to
        the largest observed time (restricted mean).
        """
        if not exact_runtimes and not lower_bounds:
            return CacheModel.MEAN_S_WITH_NO_INFO, 0.0

        exact = np.asarray(exact_runtimes, dtype=float)
        cens = np.asarray(lower_bounds, dtype=float)

        if exact.size == 0:
            lb = (
                float(np.max(cens))
                if cens.size
                else CacheModel.MEAN_S_WITH_NO_INFO
            )
            return lb, 0.0  # TODO: should have a nonzero std in this case.

        if cens.size == 0:
            return float(np.mean(exact)), float(np.std(exact, ddof=0))

        times = np.unique(np.concatenate([exact, cens]))
        times.sort()

        s_prev = 1.0
        prev_t = 0.0
        mean = 0.0
        second_moment = 0.0

        for t in times:
            dt = float(t - prev_t)
            if dt > 0:
                mean += s_prev * dt
                second_moment += s_prev * (t**2 - prev_t**2)

            n_at_risk = int((exact >= t).sum() + (cens >= t).sum())
            d_t = int((exact == t).sum())
            if n_at_risk > 0 and d_t > 0:
                s_prev *= max(0.0, 1.0 - (d_t / n_at_risk))
            prev_t = float(t)

        var = max(0.0, second_moment - mean**2)
        return float(mean), float(np.sqrt(var))

    def _init_bucket(self) -> dict[str, Any]:
        return {
            "exact_runtimes": [],
            "lower_bounds": [],
            "mean_runtime": 0.0,
            "std_runtime": 0.0,
        }

    def _normalize_bucket(self, bucket: Any) -> dict[str, Any]:
        """Normalize legacy tuple buckets and new dict buckets to one schema."""
        if isinstance(bucket, tuple) and len(bucket) == 3:
            runtimes, mean_runtime, std_runtime = bucket
            return {
                "exact_runtimes": list(runtimes),
                "lower_bounds": [],
                "mean_runtime": float(mean_runtime),
                "std_runtime": float(std_runtime),
            }

        if isinstance(bucket, dict):
            exact = list(bucket.get("exact_runtimes", []))
            lb = list(bucket.get("lower_bounds", []))
            if "mean_runtime" in bucket and "std_runtime" in bucket:
                mean_runtime = float(bucket["mean_runtime"])
                std_runtime = float(bucket["std_runtime"])
            else:
                mean_runtime, std_runtime = self._restricted_mean_std_from_km(
                    exact, lb
                )
            return {
                "exact_runtimes": exact,
                "lower_bounds": lb,
                "mean_runtime": mean_runtime,
                "std_runtime": std_runtime,
            }

        return self._init_bucket()

    def _bucket_stats(self, bucket: dict[str, Any]) -> tuple[float, float]:
        exact = [float(x) for x in bucket.get("exact_runtimes", [])]
        lb = [float(x) for x in bucket.get("lower_bounds", [])]
        return self._restricted_mean_std_from_km(exact, lb)

    def _refresh_bucket_summary(self, bucket: dict[str, Any]) -> dict[str, Any]:
        mean_runtime, std_runtime = self._bucket_stats(bucket)
        bucket["mean_runtime"] = mean_runtime
        bucket["std_runtime"] = std_runtime
        return bucket

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
                        metadata={
                            "origin": "CacheModel_miss",
                            "censored_support_used": True,
                        },
                    )
                else:
                    predictions[query_id] = None
            elif (
                query_within_template_id not in cache_for_rpu[template_id]
            ) and self._enable_template_cache:
                # Template cache hit
                runtimes: list[float] = []
                lower_bounds: list[float] = []
                for raw_bucket in cache_for_rpu[template_id].values():
                    bucket = self._normalize_bucket(raw_bucket)
                    runtimes.extend(bucket["exact_runtimes"])
                    lower_bounds.extend(bucket["lower_bounds"])
                mean_runtime, std_runtime = self._restricted_mean_std_from_km(
                    runtimes, lower_bounds
                )
                predictions[query_id] = ModelPrediction(
                    mean_s=[mean_runtime],
                    std_dev_s=[std_runtime],
                    metadata={
                        "origin": "CacheModel_template_hit",
                        "censored_support_used": True,
                    },
                )
            elif query_within_template_id in cache_for_rpu[template_id]:
                # Cache hit
                bucket = self._normalize_bucket(
                    cache_for_rpu[template_id][query_within_template_id]
                )
                mean_runtime = float(bucket["mean_runtime"])
                std_runtime = float(bucket["std_runtime"])
                predictions[query_id] = ModelPrediction(
                    mean_s=[mean_runtime],
                    std_dev_s=[std_runtime],
                    metadata={
                        "origin": "CacheModel_hit",
                        "censored_support_used": True,
                    },
                )

        return predictions

    def train(
        self,
        run_ids: list[str],
        from_scratch: bool = False,
        only_non_overlapping_queries: bool = False,
        use_client_side_latencies: bool = False,
        ignore_aborted_queries: bool = False,
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
            ignore_aborted_queries: Whether to exclude aborted queries from
                training.
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
            was_aborted = trace.was_aborted()

            for cluster_aware_query_id, query_text_id in query_text_ids.items():
                key = str(cluster_aware_query_id)
                latency = float(latencies[key])
                if (
                    only_non_overlapping_queries
                    and not query_is_non_overlapping[key]
                ):
                    continue
                if ignore_aborted_queries and was_aborted.get(key, False):
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
                    ] = self._init_bucket()

                bucket = self._normalize_bucket(
                    self._cache[cluster_rpu][template_id][
                        query_within_template_id
                    ]
                )
                if bool(was_aborted.get(key, False)):
                    bucket["lower_bounds"].append(latency)
                else:
                    bucket["exact_runtimes"].append(latency)

                bucket = self._refresh_bucket_summary(bucket)
                self._cache[cluster_rpu][template_id][
                    query_within_template_id
                ] = bucket

        # Update overall mean and standard deviation
        for cluster_rpu, template_dict in self._cache.items():
            exacts: list[float] = []
            lower_bounds: list[float] = []
            for query_dict in template_dict.values():
                for raw_bucket in query_dict.values():
                    bucket = self._normalize_bucket(raw_bucket)
                    exacts.extend(bucket["exact_runtimes"])
                    lower_bounds.extend(bucket["lower_bounds"])

            mean_runtime, std_runtime = self._restricted_mean_std_from_km(
                exacts,
                lower_bounds,
            )
            self._mean_runtime_s_for_rpu[cluster_rpu] = mean_runtime
            self._std_runtime_s_for_rpu[cluster_rpu] = std_runtime

        if self._mean_runtime_s_for_rpu:
            self._overall_mean_runtime_s = float(
                np.mean(list(self._mean_runtime_s_for_rpu.values()))
            )
            self._overall_std_runtime_s = float(
                np.mean(list(self._std_runtime_s_for_rpu.values()))
            )

        self._run_ids.extend(run_ids)
        self._run_ids = list(set(self._run_ids))

    def save(self, parent_save_dir: Optional[Path] = None) -> str:
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
        parent_save_dir = parent_save_dir or pu.get_data_dir() / "cache_models"
        timestamp = str(int(datetime.now().timestamp()))
        save_dir = parent_save_dir / timestamp
        save_dir.mkdir(parents=True, exist_ok=False)

        # Save cache model parameters
        param_path = save_dir / "params.yml"
        with open(param_path, "w") as f:
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "enable_template_cache": self._enable_template_cache,
                    "best_effort": self._best_effort,
                    "ignore_cluster_size": self._ignore_cluster_size,
                    "only_non_overlapping_queries": self._only_non_overlapping_queries,
                    "run_ids": self._run_ids,
                    "mean_runtime_s_for_rpu": dict(
                        self._mean_runtime_s_for_rpu
                    ),
                    "std_runtime_s_for_rpu": dict(self._std_runtime_s_for_rpu),
                    "overall_mean_runtime_s": self._overall_mean_runtime_s,
                    "overall_std_runtime_s": self._overall_std_runtime_s,
                },
                f,
            )

        # Save the model itself
        cache_pkl_path = save_dir / "model.pkl"
        with open(cache_pkl_path, "wb") as f:
            pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        return timestamp

    @staticmethod
    def load(
        timestamp: str, parent_load_dir: Optional[Path] = None
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
        parent_load_dir = parent_load_dir or pu.get_data_dir() / "cache_models"
        load_dir = Path(parent_load_dir) / timestamp
        if not load_dir.exists():
            raise FileNotFoundError(f"CacheModel {timestamp} not found")

        # Load model parameters
        param_path = load_dir / "params.yml"
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)
        model = CacheModel(
            enable_template_cache=params["enable_template_cache"],
            best_effort=params["best_effort"],
            ignore_cluster_size=params.get("ignore_cluster_size", False),
        )
        model._run_ids = params["run_ids"]
        model._mean_runtime_s_for_rpu = defaultdict(
            float,
            {
                int(k): float(v)
                for k, v in params["mean_runtime_s_for_rpu"].items()
            },
        )
        model._std_runtime_s_for_rpu = defaultdict(
            float,
            {
                int(k): float(v)
                for k, v in params["std_runtime_s_for_rpu"].items()
            },
        )
        model._overall_mean_runtime_s = float(
            params.get(
                "overall_mean_runtime_s",
                (
                    np.mean(list(model._mean_runtime_s_for_rpu.values()))
                    if model._mean_runtime_s_for_rpu
                    else 0.0
                ),
            )
        )
        model._overall_std_runtime_s = float(
            params.get(
                "overall_std_runtime_s",
                (
                    np.mean(list(model._std_runtime_s_for_rpu.values()))
                    if model._std_runtime_s_for_rpu
                    else 0.0
                ),
            )
        )

        # Load the model itself.  Prefer the binary pickle format; fall back
        # to the legacy YAML and auto-migrate on the way out.
        cache_pkl_path = load_dir / "model.pkl"
        cache_yml_path = load_dir / "model.yml"
        if cache_pkl_path.exists():
            with open(cache_pkl_path, "rb") as f:
                model._cache = pickle.load(f)
        else:
            with open(cache_yml_path, "r") as f:
                raw_cache = yaml.safe_load(f)

            if raw_cache is None:
                raw_cache = {}

            normalized_cache: dict[int, dict[int, dict[int, dict[str, Any]]]] = {}
            for rpu, template_dict in raw_cache.items():
                irpu = int(rpu)
                normalized_cache[irpu] = {}
                for template_id, query_dict in template_dict.items():
                    itemplate_id = int(template_id)
                    normalized_cache[irpu][itemplate_id] = {}
                    for qidx, bucket in query_dict.items():
                        iqidx = int(qidx)
                        nb = model._normalize_bucket(bucket)
                        nb = model._refresh_bucket_summary(nb)
                        normalized_cache[irpu][itemplate_id][iqidx] = nb

            model._cache = normalized_cache
            # Auto-migrate to pickle so future loads skip YAML parsing.
            with open(cache_pkl_path, "wb") as f:
                pickle.dump(model._cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        return model
