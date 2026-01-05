from autoslo.blueprint_selection.query_timeline import QueryTimeline
from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.iconq_model import IconqModel
from autoslo.models.stage_model import StageModel
from autoslo.workload_definition.workload import Workload
from autoslo.workload_execution.trace import Trace


def bootstrap_query_timeline_from_workload(
    workload: Workload,
    cluster_name: str,
    iconq_query_featurizer: IconqQueryFeaturizer,
    iconq_interaction_featurizer: IconqInteractionFeaturizer,
    stage_model: StageModel,
    iconq_model: IconqModel,
) -> QueryTimeline:
    """
    Bootstraps a QueryTimeline from a workload. That is, iteratively
    updates the latency of each query in the workload using the provided stage
    model and Iconq model.

    Parameters:
        workload: The workload to bootstrap from.
        cluster_name: The name of the cluster on which the workload will be
            assumed to execute during bootstrapping.
        iconq_query_featurizer: The IconqQueryFeaturizer to use for featurizing
            queries.
        iconq_interaction_featurizer: The IconqInteractionFeaturizer to use for
            featurizing query interactions.
        stage_model: The stage model to use for estimating query stages.
        iconq_model: The Iconq model to use for estimating query interactions.

    Returns:
        A bootstrapped QueryTimeline instance.
    """

    timeline = QueryTimeline(
        iconq_query_featurizer=iconq_query_featurizer,
        iconq_interaction_featurizer=iconq_interaction_featurizer,
    )

    # Add the queries one by one.
    for query in workload.queries():
        query_id = query.query_id
        start_time_s = query.start_time_s
        temp_and_q_idx = query.tpcds_temp_and_q_idx

        stage_prediction_overall_mean = (
            stage_model.predict_from_tpcds_temp_and_q_idx(
                {query_id: temp_and_q_idx}
            )[query_id].overall_mean_s()
        )
        timeline.add_query(
            cluster_name=cluster_name,
            start_time_s=start_time_s,
            end_time_s=(start_time_s + stage_prediction_overall_mean),
            query_id=query_id,
            tpcds_temp_and_q_idx=temp_and_q_idx,
            stage_model_prediction=stage_prediction_overall_mean,
        )

    for i in range(10):

        # Get a dataset of overlapping queries.
        dataset = timeline.get_dataset(
            use_log_runtime=iconq_model._trained_on_log_runtime
        )

        predictions = iconq_model.predict_from_dataset(
            dataset=dataset,
        )
        num_updated = 0
        for query_id, prediction in predictions.items():
            updated = timeline.update_latency(
                query_id=query_id,
                latency_s=prediction.overall_mean_s(),
            )
            if updated:
                num_updated += 1

        print(
            f"Bootstrap iteration {i + 1}: "
            f"updated latencies for {num_updated} queries."
        )

    return timeline
