from dataclasses import dataclass


QueryFeaturization = list[float]


@dataclass
class Query:
    """Class representing a single query in the workload."""

    query_id: str
    start_time_s: float
    tpcds_temp_and_q_idx: str


    featurization: QueryFeaturization = []
    cluster_name: str = ""
    stage_latency_prediction_s: float = -1

    latency_s: float = -1
    latency_is_lower_bound: bool = False

