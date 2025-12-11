from fastapi import APIRouter

from autoslo.strategies.strat_exactfuture_fullperf_single import (
    StratExactFutureFullPerfSingle,
)
from autoslo.workload_definition.composite import Composite

router = APIRouter()


@router.post("/strat/cheapest_adherent_cluster", response_model=list)
def cheapest_adherent_cluster_post(
    workload_name: str, tail_slo_s: float, percentile: float = 95.0
):
    """
    Get the ground truth cheapest adherent cluster for each day in the workload.

    Parameters:
        workload_name: The name of the workload.
        tail_slo_s: The tail SLO in seconds.
        percentile: The percentile to consider (default is 95.0).

    Returns:
       For each day in the workload, the smallest endpoint RPU that meets the
       tail SLO, or None if no suitable RPU is found.
    """

    strategy = StratExactFutureFullPerfSingle(1 - percentile / 100.0)
    workload = Composite.load(workload_name)
    l = []

    for day_idx in range(workload.num_days()):
        blueprint, _ = strategy.suggest(workload, day_idx, tail_slo_s)
        rpu = blueprint.clusters[0].rpu
        l.append(rpu)

    return l
