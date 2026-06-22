import functools
from typing import Any, Optional

import numpy as np

from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_definition.query import QueryTextId


class IconqInteractionFeaturizer:
    """
    A class for featurizing interactions between queries, as presented in the
    IconQ paper.
    """

    IconqInteractionFeaturization = list[float]
    SUPPORTED_FEATURE_VERSIONS = {"v1", "v2"}
    """
    Represents the vectorized features of a query interaction.
    
    Format per featurization:
        [
        qa_query_features..., qa_latency_prediction,
        qb_query_features..., qb_latency_prediction,
        abs(arrival_time_diff), arrival_time_sign, rpu
        ]
    """

    def __init__(
        self,
        schema_name: str,
        iconq_query_featurizer: Optional["IconqQueryFeaturizer"] = None,
        iconq_query_featurizer_id: Optional[str] = None,
        iconq_query_featurizer_init_params: Optional[dict[str, Any]] = None,
        ignore_cluster_size: bool = False,
        interaction_feature_version: str = "v1",
    ):
        """
        Initializes the IconqInteractionFeaturizer.

        Parameters:
            schema_name: The schema name.
            iconq_query_featurizer: A pre-loaded :class:`IconqQueryFeaturizer`
                instance.  When provided, neither *iconq_query_featurizer_id*
                nor *iconq_query_featurizer_init_params* is used, avoiding a
                redundant disk read.
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
            interaction_feature_version: The interaction feature schema
                version. "v1" reproduces legacy behavior (single raw RPU
                feature). "v2" appends additional RPU-derived features.
        """
        if iconq_query_featurizer is not None:
            self._iconq_query_featurizer = iconq_query_featurizer
        elif iconq_query_featurizer_id is not None:
            self._iconq_query_featurizer = IconqQueryFeaturizer.load(
                schema_name, iconq_query_featurizer_id
            )
        elif iconq_query_featurizer_init_params is not None:
            self._iconq_query_featurizer = IconqQueryFeaturizer(
                **iconq_query_featurizer_init_params
            )
        else:
            raise ValueError(
                "Must provide one of: iconq_query_featurizer, "
                "iconq_query_featurizer_id, or iconq_query_featurizer_init_params."
            )
        self._ignore_cluster_size = ignore_cluster_size
        if interaction_feature_version not in self.SUPPORTED_FEATURE_VERSIONS:
            raise ValueError(
                "Unsupported interaction_feature_version "
                f"'{interaction_feature_version}'. "
                f"Expected one of {sorted(self.SUPPORTED_FEATURE_VERSIONS)}."
            )
        self._interaction_feature_version = interaction_feature_version

    @property
    def interaction_feature_version(self) -> str:
        """Returns the interaction feature schema version in use."""
        return self._interaction_feature_version

    @functools.cached_property
    def _base_dim(self) -> int:
        """Number of dimensions before interaction-specific scalar features."""
        return 2 * self._iconq_query_featurizer.num_dims

    @functools.cached_property
    def _rpu_block_num_dims(self) -> int:
        """Number of dimensions allocated to RPU-related interaction features."""
        return 1 if self._interaction_feature_version == "v1" else 6

    @functools.cached_property
    def num_dims(self) -> int:
        """
        Returns the number of dimensions in the interaction feature vector,
        if the given query featurizer is used to produce the query features.

        Returns:
            The number of dimensions in the interaction feature vector.
        """
        return self._base_dim + 4 + self._rpu_block_num_dims

    @functools.cached_property
    def arrival_time_diff_dim_idx(self) -> int:
        """
        Returns the index of the arrival time difference feature in the
        interaction feature vector.

        Returns:
            The index of the arrival time difference feature.
        """
        return self._base_dim + 2

    @functools.cached_property
    def arrival_time_sign_dim_idx(self) -> int:
        """
        Returns the index of the arrival time sign feature in the interaction
        feature vector.

        Returns:
            The index of the arrival time sign feature.
        """
        return self._base_dim + 3

    @functools.cached_property
    def rpu_dim_idx(self) -> int:
        """
        Returns the index of the RPU feature in the interaction feature vector.

        Returns:
            The index of the RPU feature.
        """
        return self._base_dim + 4

    def featurize_one_vs_many_to_numpy(
        self,
        rpu: int,
        qa_query_text_id: QueryTextId,
        qa_start_time_s: float,
        qa_latency_prediction: float,
        qb_entries: list[tuple[float, QueryTextId, float, bool]],
    ) -> tuple[np.ndarray, int]:
        """
        Featurize a single base query (qa) against an ordered collection of
        neighbor queries (qb), writing all rows into one pre-allocated float32
        numpy array.



        Parameters:
            rpu: The RPU of the cluster on which the queries ran.
            qa_query_text_id: Query text ID of the base query.
            qa_start_time_s: Start time of the base query (seconds).
            qa_latency_prediction: Stage-model latency prediction for qa.
            qb_entries: One entry per row to produce. Each entry is a 4-tuple
                (qb_start_time_s, qb_query_text_id, qb_latency_prediction,
                is_self). Entries need not be sorted; this method sorts them by
                start time internally. The self-entry (is_self=True) identifies
                the pinch point.

        Returns:
            arr: Float32 numpy array of shape (N, num_dims), ready for
                torch.from_numpy.
            pinch_idx: Row index of the self-entry in the sorted output.
        """
        q_dim = self._iconq_query_featurizer.num_dims
        feat_dim = self.num_dims
        rpu_value = np.float32(0.0 if self._ignore_cluster_size else float(rpu))
        rpu_dim = self.rpu_dim_idx
        qa_np = (
            self._iconq_query_featurizer.featurize_from_query_text_id_as_numpy(
                qa_query_text_id
            )
        )

        # ── Fast path for the single-neighbor case (N == 1) ──────────────────
        # Most calls come from predict_incremental_batch,
        # where qb_entries always contains exactly one entry (the incoming
        # query). 
        if len(qb_entries) == 1:
            qb_t, qb_idx, qb_lat, is_self = qb_entries[0]
            arr = np.empty((1, feat_dim), dtype=np.float32)
            arr[0, :q_dim] = qa_np
            arr[0, q_dim] = qa_latency_prediction
            arr[0, q_dim + 1 : 2 * q_dim + 1] = (
                self._iconq_query_featurizer.featurize_from_query_text_id_as_numpy(
                    qb_idx
                )
            )
            arr[0, 2 * q_dim + 1] = qb_lat
            arr[0, 2 * q_dim + 2] = abs(qb_t - qa_start_time_s)
            arr[0, 2 * q_dim + 3] = float(qa_start_time_s < qb_t)
            arr[0, rpu_dim] = rpu_value
            if self._interaction_feature_version == "v2":
                if self._ignore_cluster_size:
                    arr[0, rpu_dim + 1 : rpu_dim + 6] = 0.0
                else:
                    log2_rpu = np.float32(np.log2(rpu_value))
                    inv_rpu = np.float32(1.0 / rpu_value)
                    arr[0, rpu_dim + 1] = log2_rpu
                    arr[0, rpu_dim + 2] = inv_rpu
                    arr[0, rpu_dim + 3] = (
                        np.float32(qa_latency_prediction) * rpu_value
                    )
                    arr[0, rpu_dim + 4] = np.float32(qb_lat) * rpu_value
                    arr[0, rpu_dim + 5] = (
                        np.float32(qa_latency_prediction) + np.float32(qb_lat)
                    ) * inv_rpu
            # pinch_idx is 0 in both cases:
            #   - initial isolated query  (is_self=True):  only row is the self row
            #   - incremental call        (is_self=False):  caller discards pinch_idx
            return arr, 0

        # ── General path for N > 1 ────────────────────────────────────────────
        # Sort by qb start time once.
        qb_entries_sorted = sorted(qb_entries, key=lambda e: e[0])
        N = len(qb_entries_sorted)

        arr = np.empty((N, feat_dim), dtype=np.float32)

        # qa columns are identical for every row — broadcast-assign once.
        arr[:, :q_dim] = qa_np
        arr[:, q_dim] = qa_latency_prediction

        # Fill qb columns row-by-row, writing feature vectors, latency,
        # time-difference, and time-sign directly into arr.
        pinch_idx = 0
        for j, (qb_t, qb_idx, qb_lat, is_self) in enumerate(qb_entries_sorted):
            arr[j, q_dim + 1 : 2 * q_dim + 1] = (
                self._iconq_query_featurizer.featurize_from_query_text_id_as_numpy(
                    qb_idx
                )
            )
            arr[j, 2 * q_dim + 1] = qb_lat
            arr[j, 2 * q_dim + 2] = abs(qb_t - qa_start_time_s)
            arr[j, 2 * q_dim + 3] = float(qa_start_time_s < qb_t)
            if is_self:
                pinch_idx = j

        arr[:, rpu_dim] = rpu_value

        if self._interaction_feature_version == "v2":
            if self._ignore_cluster_size:
                arr[:, rpu_dim + 1 : rpu_dim + 6] = 0.0
            else:
                # arr[:, 2 * q_dim + 1] already holds qb_lats — reuse as a view
                # to avoid a separate array allocation.
                qb_lats_view = arr[:, 2 * q_dim + 1]
                log2_rpu = np.float32(np.log2(rpu_value))
                inv_rpu = np.float32(1.0 / rpu_value)
                qa_work_proxy = np.float32(qa_latency_prediction) * rpu_value
                arr[:, rpu_dim + 1] = log2_rpu
                arr[:, rpu_dim + 2] = inv_rpu
                arr[:, rpu_dim + 3] = qa_work_proxy
                arr[:, rpu_dim + 4] = qb_lats_view * rpu_value
                arr[:, rpu_dim + 5] = (
                    np.float32(qa_latency_prediction) + qb_lats_view
                ) * inv_rpu

        return arr, pinch_idx
