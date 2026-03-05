from datetime import datetime

from autoslo.blueprint_selection.selector import BlueprintSelector
from autoslo.models.iconq_model import IconqModel
from autoslo.workload_definition.chunk import Chunk


def test_blueprint_selector_initialization() -> None:
    """Test the initialization of BlueprintSelector with IconqModel."""
    workload = Chunk.load('tpcds_99templates_00pctheavy_10meaninterarrivals')
    
    slo_s = 1.0
    slo_violation_rate_threshold = 0.05
    iconq_model_id = "1767629626"

    selector = BlueprintSelector(
        workload=workload,
        slo_s=slo_s,
        slo_violation_rate_threshold=slo_violation_rate_threshold,
        iconq_model_id=iconq_model_id,
        default_cluster_name="cluster_4",
    )

    assert selector._workload == workload
    assert selector._slo_s == slo_s
    assert (
        selector._slo_violation_rate_threshold == slo_violation_rate_threshold
    )
    assert selector._iconq_model_id == iconq_model_id
    assert isinstance(selector._iconq_model, IconqModel)
