from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_execution.trace import Trace


class IconqInteractionFeaturizer:
    """
    A class for featurizing interactions between queries, as presented in the
    IconQ paper.
    """

    IconqInteractionFeaturization = list[float]
    """Represents the vectorized features of a query interaction."""

    def __init__(self, iconq_query_featurizer: IconqQueryFeaturizer):
        """
        Initializes the IconqInteractionFeaturizer.

        Parameters:
            iconq_query_featurizer: The query featurizer used to produce the
                query features.
        """
        self._iconq_query_featurizer = iconq_query_featurizer

    def num_dims(self) -> int:
        """
        Returns the number of dimensions in the interaction feature vector,
        if the given query featurizer is used to produce the query features.

        Returns:
            The number of dimensions in the interaction feature vector.
        """
        return 2 * self._iconq_query_featurizer.num_dims + 4

    def featurize(
        self,
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
            qa_tpcds_temp_and_q_idx=qa_tpcds_temp_and_q_idx,
            qa_start_time_s=qa_start_time_s,
            qa_latency_prediction=qa_latency_prediction,
            qb_tpcds_temp_and_q_idx=qb_tpcds_temp_and_q_idx,
            qb_start_time_s=qb_start_time_s,
            qb_latency_prediction=qb_latency_prediction,
        )

    def featurize_from_tpcds_temp_and_q_idx(
        self,
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
            qa_tpcds_temp_and_q_idx: The TPC-DS template and query index of the
                first query.
            qa_start_time_s: The first query start time (Unix timestamp).
            qa_latency_prediction: The latency prediction of the first query.
            qb_tpcds_temp_and_q_idx: The TPC-DS template and query index of the
                second query.
            qb_start_time_s: The second query start time (Unix timestamp).
            qb_latency_prediction: The latency prediction of the second query.
        """

        return (
            self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                qa_tpcds_temp_and_q_idx
            )
            + [qa_latency_prediction]
            + self._iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
                qb_tpcds_temp_and_q_idx
            )
            + [qb_latency_prediction]
            + [
                abs(qb_start_time_s - qa_start_time_s),
                float(qa_start_time_s < qb_start_time_s),
            ]
        )
