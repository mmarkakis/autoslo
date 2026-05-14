from typing import Any, Optional

import numpy as np

from autoslo.clusters.cluster import Cluster
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_definition.query import QueryTextId


class IconqInteractionFeaturizer:
    """
    A class for featurizing interactions between queries, as presented in the
    IconQ paper.
    """

    IconqInteractionFeaturization = list[float]
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
        iconq_query_featurizer_id: Optional[str] = None,
        iconq_query_featurizer_init_params: Optional[dict[str, Any]] = None,
        ignore_cluster_size: bool = False,
    ):
        """
        Initializes the IconqInteractionFeaturizer.

        Parameters:
            schema_name: The schema name.
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
        else:
            self._iconq_query_featurizer = IconqQueryFeaturizer.load(
                schema_name, iconq_query_featurizer_id
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

    def featurize_one_vs_many_to_numpy(
        self,
        cluster_name: str,
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
            cluster_name: The cluster on which all queries execute.
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

        rpu = (
            0.0
            if self._ignore_cluster_size
            else Cluster.rpu_for_cluster_name(cluster_name)
        )

        # Sort by qb start time once.
        qb_entries_sorted = sorted(qb_entries, key=lambda e: e[0])

        arr = np.empty((len(qb_entries_sorted), feat_dim), dtype=np.float32)

        # qa columns are identical for every row — broadcast-assign once.
        qa_np = (
            self._iconq_query_featurizer.featurize_from_query_text_id_as_numpy(
                qa_query_text_id
            )
        )
        arr[:, :q_dim] = qa_np
        arr[:, q_dim] = qa_latency_prediction

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
            arr[j, 2 * q_dim + 4] = rpu
            if is_self:
                pinch_idx = j

        return arr, pinch_idx
