from typing import Any, Optional

import numpy as np

from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_execution.trace import Trace


class IconqInteractionFeaturizer:
    """
    A class for featurizing interactions between queries, as presented in the
    IconQ paper.
    """

    IconqInteractionFeaturization = list[float]
    """Represents the vectorized features of a query interaction."""

    def __init__(
        self,
        iconq_query_featurizer_id: Optional[str] = None,
        iconq_query_featurizer_init_params: Optional[dict[str, Any]] = None,
        ignore_cluster_size: bool = False,
    ):
        """
        Initializes the IconqInteractionFeaturizer.

        Parameters:
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
                iconq_query_featurizer_id
            )
        self._ignore_cluster_size = ignore_cluster_size

    @property
    def num_dims(self) -> int:
        """
        Returns the number of dimensions in the interaction feature vector,
        if the given query featurizer is used to produce the query features.

        Returns:
            The number of dimensions in the interaction feature vector.
        """
        return 2 * self._iconq_query_featurizer.num_dims + 5

    @property
    def arrival_time_diff_dim_idx(self) -> int:
        """
        Returns the index of the arrival time difference feature in the
        interaction feature vector.

        Returns:
            The index of the arrival time difference feature.
        """
        return self.num_dims - 3

    @property
    def arrival_time_sign_dim_idx(self) -> int:
        """
        Returns the index of the arrival time sign feature in the interaction
        feature vector.

        Returns:
            The index of the arrival time sign feature.
        """
        return self.num_dims - 2

    @property
    def rpu_dim_idx(self) -> int:
        """
        Returns the index of the RPU feature in the interaction feature vector.

        Returns:
            The index of the RPU feature.
        """
        return self.num_dims - 1

    @property
    def ignore_cluster_size(self) -> bool:
        """
        Returns whether the cluster size is ignored when featurizing queries.

        Returns:
            True if the cluster size is ignored, False otherwise.
        """
        return self._ignore_cluster_size

    def featurize(
        self,
        cluster_name: str,
        qa_query_text: str,
        qa_start_time_s: float,
        qa_latency_prediction: float,
        qb_query_text: str,
        qb_start_time_s: float,
        qb_latency_prediction: float,
    ) -> IconqInteractionFeaturization:
        """
        Featurizes the interaction between two queries.

        Parameters:
            cluster_name: The name of the cluster on which the queries are
                executed.
            qa_query_text: The text of the first query.
            qa_features: The features of the first query.
            qa_start_time_s: The first query start time (Unix timestamp).
            qb_query_text: The text of the second query.
            qb_features: The features of the second query.
            qb_start_time_s: The second query start time (Unix timestamp).

        Returns:
            The features of the interaction between the two queries.
        """
        qa_tpcds_temp_and_q_idx = Trace.extract_temp_and_q_idxs(qa_query_text)
        qb_tpcds_temp_and_q_idx = Trace.extract_temp_and_q_idxs(qb_query_text)

        return self.featurize_from_tpcds_temp_and_q_idx(
            cluster_name=cluster_name,
            qa_tpcds_temp_and_q_idx=qa_tpcds_temp_and_q_idx,
            qa_start_time_s=qa_start_time_s,
            qa_latency_prediction=qa_latency_prediction,
            qb_tpcds_temp_and_q_idx=qb_tpcds_temp_and_q_idx,
            qb_start_time_s=qb_start_time_s,
            qb_latency_prediction=qb_latency_prediction,
        )

    def featurize_from_tpcds_temp_and_q_idx(
        self,
        cluster_name: str,
        qa_tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx,
        qa_start_time_s: float,
        qa_latency_prediction: float,
        qb_tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx,
        qb_start_time_s: float,
        qb_latency_prediction: float,
    ) -> IconqInteractionFeaturization:
        """
        Featurizes the interaction between two queries, given their TPC-DS
        template and query indices.

        Parameters:
            cluster_name: The name of the cluster on which the queries are
                executed.
            qa_tpcds_temp_and_q_idx: The TPC-DS template and query index of the
                first query.
            qa_start_time_s: The first query start time (Unix timestamp).
            qa_latency_prediction: The latency prediction of the first query.
            qb_tpcds_temp_and_q_idx: The TPC-DS template and query index of the
                second query.
            qb_start_time_s: The second query start time (Unix timestamp).
            qb_latency_prediction: The latency prediction of the second query.
        """

        return self.featurize_from_vectors(
            cluster_name=cluster_name,
            qa_features=self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                qa_tpcds_temp_and_q_idx
            ),
            qa_start_time_s=qa_start_time_s,
            qa_latency_prediction=qa_latency_prediction,
            qb_features=self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                qb_tpcds_temp_and_q_idx
            ),
            qb_start_time_s=qb_start_time_s,
            qb_latency_prediction=qb_latency_prediction,
        )

    def featurize_from_vectors(
        self,
        cluster_name: str,
        qa_features: list[float],
        qa_start_time_s: float,
        qa_latency_prediction: float,
        qb_features: list[float],
        qb_start_time_s: float,
        qb_latency_prediction: float,
    ) -> IconqInteractionFeaturization:
        """
        Featurizes the interaction between two queries, given their feature
        vectors.

        Parameters:
            cluster_name: The name of the cluster on which the queries are
                executed.
            qa_features: The features of the first query.
            qa_start_time_s: The first query start time (Unix timestamp).
            qa_latency_prediction: The latency prediction of the first query.
            qb_features: The features of the second query.
            qb_start_time_s: The second query start time (Unix timestamp).
            qb_latency_prediction: The latency prediction of the second query.
        """
        rpu = 0.0
        if not self.ignore_cluster_size:
            rpu = Cluster.rpu_for_cluster_name(cluster_name)
        return (
            qa_features
            + [qa_latency_prediction]
            + qb_features
            + [qb_latency_prediction]
            + [
                abs(qb_start_time_s - qa_start_time_s),
                float(qa_start_time_s < qb_start_time_s),
                rpu,
            ]
        )

    def featurize_one_vs_many_to_numpy(
        self,
        cluster_name: str,
        qa_tpcds_temp_and_q_idx: Trace.TPCDSTempAndQIdx,
        qa_start_time_s: float,
        qa_latency_prediction: float,
        qb_entries: list[tuple[float, Trace.TPCDSTempAndQIdx, float, bool]],
    ) -> tuple[np.ndarray, int]:
        """
        Featurize a single base query (qa) against an ordered collection of
        neighbor queries (qb), writing all rows into one pre-allocated float32
        numpy array.

        This is the batch equivalent of calling featurize_from_tpcds_temp_and_q_idx
        N times. It avoids per-row list allocations, repeated list concatenation,
        repeated rpu lookups, and the O(N) pinch-point search.

        Parameters:
            cluster_name: The cluster on which all queries execute.
            qa_tpcds_temp_and_q_idx: Template/query index of the base query.
            qa_start_time_s: Start time of the base query (seconds).
            qa_latency_prediction: Stage-model latency prediction for qa.
            qb_entries: One entry per row to produce. Each entry is a 4-tuple
                (qb_start_time_s, qb_tpcds_temp_and_q_idx, qb_latency_prediction,
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

        rpu = (
            0.0
            if self._ignore_cluster_size
            else Cluster.rpu_for_cluster_name(cluster_name)
        )

        # Sort by qb start time once.
        qb_entries_sorted = sorted(qb_entries, key=lambda e: e[0])

        arr = np.empty((len(qb_entries_sorted), feat_dim), dtype=np.float32)

        # qa columns are identical for every row — broadcast-assign once.
        qa_np = self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx_as_numpy(
            qa_tpcds_temp_and_q_idx
        )
        arr[:, :q_dim] = qa_np
        arr[:, q_dim] = qa_latency_prediction

        pinch_idx = 0
        for j, (qb_t, qb_idx, qb_lat, is_self) in enumerate(qb_entries_sorted):
            arr[j, q_dim + 1 : 2 * q_dim + 1] = (
                self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx_as_numpy(
                    qb_idx
                )
            )
            arr[j, 2 * q_dim + 1] = qb_lat
            arr[j, 2 * q_dim + 2] = abs(qb_t - qa_start_time_s)
            arr[j, 2 * q_dim + 3] = float(qa_start_time_s < qb_t)
            arr[j, 2 * q_dim + 4] = rpu
            if is_self:
                pinch_idx = j

        return arr, pinch_idx

    def featurize_all_vs_all_to_numpy(
        self,
        cluster_name: str,
        entries: list[tuple[Trace.TPCDSTempAndQIdx, float, float]],
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        Batch version of featurize_one_vs_many_to_numpy for the all-vs-all
        case: every entry acts as a base query (qa) in turn, with the full
        set of entries as neighbors (qb).

        Compared to calling featurize_one_vs_many_to_numpy N times:
        - The entries are sorted only once.
        - The shared qb columns (feat matrix, latencies, timestamps) are
          pre-computed as numpy arrays and reused for every base query.
        - Per-base-query work is reduced to two numpy broadcasts (qa feat &
          qa lat) plus two vectorised numpy operations (|Δt| and sign), with
          no Python scalar loop over neighbors.

        Parameters:
            cluster_name: The cluster on which all queries execute.
            entries: One entry per query: (tpcds_temp_and_q_idx, start_time_s,
                latency_prediction). The order need not be sorted.

        Returns:
            arrays: One float32 numpy array of shape (N, num_dims) per entry,
                where entry i is treated as the base query. Ready for
                torch.from_numpy.
            pinch_indices: For each array, the row index of the base query
                within the sorted output.
        """
        q_dim = self._iconq_query_featurizer.num_dims
        feat_dim = self.num_dims  # 2 * q_dim + 5
        n = len(entries)

        rpu = (
            0.0
            if self._ignore_cluster_size
            else Cluster.rpu_for_cluster_name(cluster_name)
        )

        # Sort by start time once, track original indices for pinch lookup.
        sorted_entries = sorted(
            enumerate(entries), key=lambda ie: ie[1][1]
        )  # List of (original_idx, (tpcds, t, lat))

        # Map original index → sorted position (for pinch_idx per base query).
        orig_to_sorted: list[int] = [0] * n
        for sorted_pos, (orig_idx, _) in enumerate(sorted_entries):
            orig_to_sorted[orig_idx] = sorted_pos

        # Pre-compute qb columns shared across all base queries.
        qb_feat = np.empty((n, q_dim), dtype=np.float32)
        qb_lat = np.empty(n, dtype=np.float32)
        qb_t = np.empty(n, dtype=np.float32)
        for sorted_pos, (_, (idx, t, lat)) in enumerate(sorted_entries):
            qb_feat[sorted_pos] = (
                self._iconq_query_featurizer
                .featurize_from_tpcds_temp_and_q_idx_as_numpy(idx)
            )
            qb_lat[sorted_pos] = lat
            qb_t[sorted_pos] = t

        # Build one interaction matrix per base query.
        arrays: list[np.ndarray] = []
        for orig_idx, (qa_idx, qa_t, qa_lat) in enumerate(entries):
            arr = np.empty((n, feat_dim), dtype=np.float32)

            # qa columns: constant across all rows — broadcast once.
            arr[:, :q_dim] = (
                self._iconq_query_featurizer
                .featurize_from_tpcds_temp_and_q_idx_as_numpy(qa_idx)
            )
            arr[:, q_dim] = qa_lat

            # qb columns: pre-computed, copy in one shot.
            arr[:, q_dim + 1 : 2 * q_dim + 1] = qb_feat
            arr[:, 2 * q_dim + 1] = qb_lat

            # Time-difference columns: vectorised.
            arr[:, 2 * q_dim + 2] = np.abs(qb_t - qa_t)
            arr[:, 2 * q_dim + 3] = (qa_t < qb_t).astype(np.float32)
            arr[:, 2 * q_dim + 4] = rpu

            arrays.append(arr)

        pinch_indices = orig_to_sorted
        return arrays, pinch_indices
