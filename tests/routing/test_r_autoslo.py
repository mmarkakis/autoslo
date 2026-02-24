"""
Tests for :mod:`autoslo.routing.r_autoslo`.

Since ``RAutoSLO.__init__`` loads an ``IconqModel`` from disk, these tests
construct the router with a monkeypatched model to keep them fast and
isolated.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.models.model_prediction import ModelPrediction
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.routing.r_autoslo import RAutoSLO
from autoslo.routing.routing_core import ClusterSnapshot
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(mean: float) -> ModelPrediction:
    return ModelPrediction(mean_s=[mean])


class _FakeDataset:
    """Minimal stand-in for ConcurrentQueryDataset that records the
    run_to_base_to_neighbors it was built from, so the mocked
    predict_from_dataset can iterate over it."""

    def __init__(self, items: list[dict[str, str]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self._items[idx]


def _make_router(
    cluster_names: list[str] | None = None,
    default_slo_s: float = 10.0,
    on_capacity_pressure: Any = None,
) -> RAutoSLO:
    """Build an RAutoSLO with a mocked IconqModel and Cluster/Blueprint,
    sidestepping all I/O."""
    if cluster_names is None:
        cluster_names = ["c0", "c1"]

    # --- Mock IconqModel ---
    mock_model = MagicMock()

    # Featurizer: returns a dummy vector.
    mock_model.iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx.return_value = [
        0.0,
        1.0,
    ]

    # Stage model: returns a constant prediction.
    def _stage_predict(id_map, cluster_name):
        return {qid: _pred(5.0) for qid in id_map}

    mock_model.stage_model.predict_from_tpcds_temp_and_q_idx.side_effect = (
        _stage_predict
    )

    # Interaction featurizer: pass-through.
    mock_model.iconq_interaction_featurizer = MagicMock()

    # predict_from_dataset: returns predictions keyed by cluster_name → qid.
    # Default: predict 5s for every query.
    def _predict_dataset(dataset, **kw):
        result: dict[str, dict[str, ModelPrediction]] = {}
        for i in range(len(dataset)):
            item = dataset[i]
            run_id = item["run_id"]
            qid = item["query_id"]
            if run_id not in result:
                result[run_id] = {}
            result[run_id][qid] = _pred(5.0)
        return result

    mock_model.predict_from_dataset.side_effect = _predict_dataset

    # --- Patch ConcurrentQueryDataset.build_from_query_groups ---
    # Instead of building a real dataset (which needs a real featurizer),
    # return a _FakeDataset that maps (run_id, query_id) for each entry.
    def _fake_build(
        iconq_interaction_featurizer=None,
        run_to_base_to_neighbors=None,
        **kw,
    ):
        items: list[dict[str, str]] = []
        if run_to_base_to_neighbors is not None:
            for run_id, base_to_neigh in run_to_base_to_neighbors.items():
                for q in base_to_neigh.keys():
                    items.append({"run_id": run_id, "query_id": q.query_id})
        return _FakeDataset(items)

    build_patcher = patch.object(
        ConcurrentQueryDataset,
        "build_from_query_groups",
        staticmethod(_fake_build),
    )
    build_patcher.start()

    # --- Build the router with patched internals ---
    with patch.object(RAutoSLO, "__init__", lambda self, *a, **kw: None):
        router = RAutoSLO.__new__(RAutoSLO)

    # Manually set the attributes that __init__ would set.
    router._iconq_model_id = "mock"
    router._iconq_model = mock_model
    router._eligible_cluster_names = list(cluster_names)
    router._cost_per_second = {cn: 1.0 for cn in cluster_names}
    router._slo_resolver = SloResolver.from_dict(
        default_slo_s=default_slo_s,
        slo_dict={},
    )
    router._optimize_by_amount = True
    router._lock = __import__("threading").Lock()
    router._active_queries = {cn: {} for cn in cluster_names}
    router._neighbors_per_active_query = {}
    router._billing_window_start_s = {cn: None for cn in cluster_names}
    router._recent_tables = {cn: set() for cn in cluster_names}
    router._on_capacity_pressure = on_capacity_pressure
    router._name = "RAutoSLO(mock)"
    router.TOLERANCE_S = 1e-4
    router._build_patcher = build_patcher  # for cleanup

    # Fake Blueprint (just needs .cluster_names and .name)
    bp = MagicMock()
    bp.cluster_names = list(cluster_names)
    bp.name = "mock_blueprint"
    router._blueprint = bp

    return router


@pytest.fixture(autouse=True)
def _cleanup_patchers():
    """Stop any build_from_query_groups patchers after each test."""
    yield
    # After the test, stop any patcher that was started by _make_router.
    # We patch at the class level so we must restore it.
    patch.stopall()


# ---------------------------------------------------------------------------
# Tests: properties & basics
# ---------------------------------------------------------------------------


class TestRAutoSLOProperties:

    def test_name(self):
        r = _make_router()
        assert "RAutoSLO" in r.name

    def test_blueprint(self):
        r = _make_router(["c0", "c1"])
        assert "c0" in r.blueprint.cluster_names

    def test_slo_resolver(self):
        r = _make_router(default_slo_s=42.0)
        assert r.slo_resolver.default_slo_s == 42.0


# ---------------------------------------------------------------------------
# Tests: on_query_start / on_query_finish
# ---------------------------------------------------------------------------


class TestLifecycleHooks:

    def test_start_and_finish(self):
        r = _make_router()
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        assert len(r.get_active_queries("c0")) == 1
        r.on_query_finish("q1", "c0", current_time_s=5.0)
        assert len(r.get_active_queries("c0")) == 0

    def test_finish_unknown_raises(self):
        r = _make_router()
        with pytest.raises(KeyError):
            r.on_query_finish("nonexistent", "c0", current_time_s=5.0)

    def test_billing_window_lifecycle(self):
        """Billing window opens on first query, closes when cluster empties."""
        r = _make_router()

        min_bill_window = Billing.REDSHIFT_BILLING_THRESHOLD_S
        assert r._billing_window_start_s["c0"] is None
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        assert r._billing_window_start_s["c0"] == 0.0
        r.on_query_finish("q1", "c0", current_time_s=min_bill_window / 2)
        assert r._billing_window_start_s["c0"] == 0.0  # still open

    def test_multiple_queries_keep_window_open(self):
        r = _make_router()
        start_time_s = 3.0
        min_bill_window = Billing.REDSHIFT_BILLING_THRESHOLD_S
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=start_time_s
        )

        assert r._billing_window_start_s["c0"] == start_time_s
        r.on_query_finish(
            "q1", "c0", current_time_s=start_time_s + min_bill_window / 4
        )

        r.on_query_start(
            "q2",
            "c0",
            tpcds_temp_and_q_idx="2_0",
            current_time_s=start_time_s + min_bill_window / 2,
        )
        assert r._billing_window_start_s["c0"] == start_time_s
        r.on_query_finish(
            "q2", "c0", current_time_s=start_time_s + min_bill_window
        )
        assert r._billing_window_start_s["c0"] is None


# ---------------------------------------------------------------------------
# Tests: headroom helpers
# ---------------------------------------------------------------------------


class TestHeadroomHelpers:

    def test_empty_headroom(self):
        r = _make_router()
        assert r.get_slo_headroom() == 1.0

    def test_headroom_with_queries(self):
        r = _make_router(default_slo_s=10.0)
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        # stage prediction = 5s, SLO = 10s → headroom = 0.5
        h = r.get_slo_headroom()
        assert h == pytest.approx(0.5)

    def test_per_cluster_headroom(self):
        r = _make_router(default_slo_s=10.0)
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        assert r.get_cluster_headroom("c0") == pytest.approx(0.5)
        assert r.get_cluster_headroom("c1") == 1.0


# ---------------------------------------------------------------------------
# Tests: capacity pressure
# ---------------------------------------------------------------------------


class TestCapacityPressure:

    def test_pressure_callback_fires_when_all_violate(self):
        """When all clusters have positive marginal violation,
        on_capacity_pressure is called."""
        fired: list[bool] = []
        # Use a high slo_s so we can manipulate predictions.
        r = _make_router(
            default_slo_s=1.0,  # very tight SLO
            on_capacity_pressure=lambda: fired.append(True),
        )

        # Override predict_from_dataset to return latencies > SLO for all qids.
        def _all_violate(dataset, **kw):
            result: dict[str, dict[str, ModelPrediction]] = {}
            for i in range(len(dataset)):
                item = dataset[i]
                run_id = item["run_id"]
                qid = item["query_id"]
                if run_id not in result:
                    result[run_id] = {}
                result[run_id][qid] = _pred(50.0)  # >> SLO of 1s
            return result

        r._iconq_model.predict_from_dataset.side_effect = _all_violate

        r.route_query(
            query_id="q1",
            tpcds_temp_and_q_idx="1_0",
            start_time_s=0.0,
        )
        assert fired


# ---------------------------------------------------------------------------
# Tests: route_query returns a cluster name
# ---------------------------------------------------------------------------


class TestRouteQuery:

    def test_returns_valid_cluster(self):
        r = _make_router(["c0", "c1"])
        result = r.route_query(
            query_id="q1",
            tpcds_temp_and_q_idx="1_0",
            start_time_s=0.0,
        )
        assert result in ["c0", "c1"]

    def test_kwargs_compatibility(self):
        """QueryRunner passes seq_num and tpcds_temp_and_q_idx as kwargs."""
        r = _make_router(["c0"])
        result = r.route_query(
            seq_num="q1",
            tpcds_temp_and_q_idx="1_0",
            start_time_s=0.0,
        )
        assert result == "c0"


# ---------------------------------------------------------------------------
# Tests: neighbor history tracking
# ---------------------------------------------------------------------------


class TestNeighborHistory:

    def test_neighbors_initialized_on_start(self):
        """When a query starts, its neighbor list = current active queries."""
        r = _make_router(["c0"])
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        # q1 started alone → empty neighbor list.
        assert r._neighbors_per_active_query["q1"] == []

        r.on_query_start(
            "q2", "c0", tpcds_temp_and_q_idx="2_0", current_time_s=1.0
        )
        # q2 started while q1 was active → q2 neighbors = [q1].
        assert len(r._neighbors_per_active_query["q2"]) == 1
        assert r._neighbors_per_active_query["q2"][0].query_id == "q1"
        # q1's neighbor list should now include q2.
        assert len(r._neighbors_per_active_query["q1"]) == 1
        assert r._neighbors_per_active_query["q1"][0].query_id == "q2"

    def test_finished_query_stays_in_neighbors(self):
        """After q1 finishes, it remains in q2's neighbor list."""
        r = _make_router(["c0"])
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        r.on_query_start(
            "q2", "c0", tpcds_temp_and_q_idx="2_0", current_time_s=1.0
        )
        r.on_query_finish("q1", "c0", current_time_s=5.0)

        # q1's own entry is gone.
        assert "q1" not in r._neighbors_per_active_query
        # But q2 still has q1 in its neighbor list.
        neighbor_ids = [q.query_id for q in r._neighbors_per_active_query["q2"]]
        assert "q1" in neighbor_ids

    def test_three_queries_accumulate_neighbors(self):
        """q1, q2, q3 all start on the same cluster — neighbor lists grow."""
        r = _make_router(["c0"])
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        r.on_query_start(
            "q2", "c0", tpcds_temp_and_q_idx="2_0", current_time_s=1.0
        )
        r.on_query_start(
            "q3", "c0", tpcds_temp_and_q_idx="3_0", current_time_s=2.0
        )

        # q1: neighbors = [q2, q3]  (q2 added when q2 started, q3 added when q3 started)
        q1_neighbor_ids = {
            q.query_id for q in r._neighbors_per_active_query["q1"]
        }
        assert q1_neighbor_ids == {"q2", "q3"}

        # q2: neighbors = [q1, q3]  (q1 added when q2 started, q3 added when q3 started)
        q2_neighbor_ids = {
            q.query_id for q in r._neighbors_per_active_query["q2"]
        }
        assert q2_neighbor_ids == {"q1", "q3"}

        # q3: neighbors = [q1, q2]  (both were active when q3 started)
        q3_neighbor_ids = {
            q.query_id for q in r._neighbors_per_active_query["q3"]
        }
        assert q3_neighbor_ids == {"q1", "q2"}

    def test_neighbor_history_persists_across_finish(self):
        """q1 starts, q2 starts, q1 finishes, q3 starts.
        q2 should have neighbors [q1, q3] — q1 persists even after finishing."""
        r = _make_router(["c0"])
        r.on_query_start(
            "q1", "c0", tpcds_temp_and_q_idx="1_0", current_time_s=0.0
        )
        r.on_query_start(
            "q2", "c0", tpcds_temp_and_q_idx="2_0", current_time_s=1.0
        )
        r.on_query_finish("q1", "c0", current_time_s=5.0)
        r.on_query_start(
            "q3", "c0", tpcds_temp_and_q_idx="3_0", current_time_s=2.0
        )

        q2_neighbor_ids = {
            q.query_id for q in r._neighbors_per_active_query["q2"]
        }
        assert q2_neighbor_ids == {"q1", "q3"}

        # q3 only sees q2 (q1 was already finished when q3 started).
        q3_neighbor_ids = {
            q.query_id for q in r._neighbors_per_active_query["q3"]
        }
        assert q3_neighbor_ids == {"q2"}
