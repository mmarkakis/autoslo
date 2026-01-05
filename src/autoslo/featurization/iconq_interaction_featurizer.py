from typing import Any, Optional

import autoslo.utils.paths as pu
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

        self._cluster_dicts = pu.get_cluster_dicts_from_config()

    @property
    def num_dims(self) -> int:
        """
        Returns the number of dimensions in the interaction feature vector,
        if the given query featurizer is used to produce the query features.

        Returns:
            The number of dimensions in the interaction feature vector.
        """
        return 2 * self._iconq_query_featurizer.num_dims + 5

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
        return (
            qa_features
            + [qa_latency_prediction]
            + qb_features
            + [qb_latency_prediction]
            + [
                abs(qb_start_time_s - qa_start_time_s),
                float(qa_start_time_s < qb_start_time_s),
                self._cluster_dicts[cluster_name]["rpu"],
            ]
        )
