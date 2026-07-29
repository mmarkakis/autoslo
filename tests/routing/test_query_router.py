"""Tests for QueryRouter — Groups A–G.

Unit tests (Groups A–F) mock IconqModel and run with no data files.
Integration tests (Group G) load a real model from disk and are marked
@pytest.mark.integration.

Test group overview:
  A — _collect_cluster_pairs
  B — _updated_cluster_state
  C — _score_cache_risk
  D — select_best  (replaces test_select_best.py)
  E — route_query  (mocked IconqModel)
  F — QueryRouter constructor behaviour
  G — end-to-end integration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autoslo.clusters.cluster import Cluster, ClusterState, ClusterView
from autoslo.config.component_configs import (
    QueryRouterConfig,
    SloObjectiveConfig,
    SloResolverConfig,
)
from autoslo.filesystem.structured_events import EventType
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.routing.query_router import QueryRouter
from autoslo.slo.slo_objective import SloObjective, ViolationCost
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import ClusterAwareQueryId, Query, QueryTextId

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

_N = 8  # cache state / table vector dimension
_RUN_ID = "test-run"
_QTID = QueryTextId("ext_tpcds1000#042#001")
_QTID2 = QueryTextId("ext_tpcds1000#003#001")


def _make_slo_resolver(default_slo_s: float = 30.0) -> SloResolver:
    return SloResolver(SloResolverConfig(slo_s=default_slo_s, slo_dict_filename=None))


def _make_slo_objective(
    metric: str = "binary", threshold: float = 0.5
) -> SloObjective:
    return SloObjective(
        SloObjectiveConfig(slo_metric=metric, slo_threshold=threshold)
    )


def _make_router_config(policy: str = "round_robin", **kwargs: Any) -> QueryRouterConfig:
    return QueryRouterConfig(routing_policy_name=policy, **kwargs)


def _make_iconq_stub(
    n: int,
    predictions: dict[ClusterAwareQueryId, ModelPrediction] | None = None,
) -> MagicMock:
    stub = MagicMock(spec=IconqModel)
    stub.supports_stateful_inference = False
    stub.iconq_query_featurizer.num_tables = n
    stub.iconq_query_featurizer.table_vector_for.return_value = np.zeros(n)
    stub.predict_from_query_groups.return_value = predictions or {}
    return stub


def _make_pred(mean_s: float) -> ModelPrediction:
    mock = MagicMock(spec=ModelPrediction)
    mock.overall_mean_s.return_value = mean_s
    return mock


def _make_query(
    query_id: str,
    qtid: QueryTextId = _QTID,
    t: float = 0.0,
    stage_preds: dict[int, float] | None = None,
) -> Query:
    return Query(
        query_id=query_id,
        query_text_id=qtid,
        rel_start_time_s=t,
        stage_predictions_per_rpu=stage_preds or {},
    )


def _make_cluster(
    name: str,
    rpu: int = 16,
    n: int = _N,
    *,
    queries: list[Query] | None = None,
    predicted_latencies: dict[str, float] | None = None,
) -> ClusterView:
    c = Cluster(
        creation_time_s=0.0,
        rpu=rpu,
        cache_state=np.zeros(n),
        name=name,
        state=ClusterState.READY,
    )
    if queries:
        lat = predicted_latencies or {}
        for q in queries:
            c.add_query(
                q,
                {qq.query_id: lat.get(qq.query_id, 5.0) for qq in ([q] + c.active_queries)},
                np.zeros(n),
                {},
            )
    return ClusterView(c)


def _make_router(
    tmp_path: Path,
    policy: str = "round_robin",
    predictions: dict[ClusterAwareQueryId, ModelPrediction] | None = None,
    slo_s: float = 30.0,
    threshold: float = 0.5,
    **router_config_kwargs: Any,
) -> QueryRouter:
    return QueryRouter(
        slo_resolver=_make_slo_resolver(slo_s),
        slo_objective=_make_slo_objective(threshold=threshold),
        query_router_config=_make_router_config(policy, **router_config_kwargs),
        iconq_model=_make_iconq_stub(_N, predictions),
        out_dir=tmp_path,
    )


# ===========================================================================
# Group A — _collect_cluster_pairs
# ===========================================================================


class TestCollectClusterPairs:
    def _router(self, tmp_path: Path) -> QueryRouter:
        return _make_router(tmp_path)

    def test_a1_empty_query_list(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        result = router._collect_cluster_pairs(queries=[], predicted_latencies={})
        assert result == []

    def test_a2_one_query_global_slo(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, slo_s=30.0)
        q = _make_query("q1")
        pairs = router._collect_cluster_pairs(
            queries=[q], predicted_latencies={"q1": 20.0}
        )
        assert len(pairs) == 1
        assert pairs[0].latency_s == pytest.approx(20.0)
        assert pairs[0].slo_s == pytest.approx(30.0)

    def test_a3_three_queries(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, slo_s=30.0)
        queries = [_make_query(f"q{i}") for i in range(3)]
        latencies = {"q0": 5.0, "q1": 15.0, "q2": 40.0}
        pairs = router._collect_cluster_pairs(queries=queries, predicted_latencies=latencies)
        assert len(pairs) == 3
        got_latencies = {p.latency_s for p in pairs}
        assert got_latencies == {5.0, 15.0, 40.0}

    def test_a4_missing_query_id_raises(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        q = _make_query("q_missing")
        with pytest.raises(KeyError):
            router._collect_cluster_pairs(
                queries=[q], predicted_latencies={"q_other": 10.0}
            )

    def test_a5_no_slo_dict_uses_default(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, slo_s=42.0)
        q = _make_query("q1")
        pairs = router._collect_cluster_pairs(queries=[q], predicted_latencies={"q1": 1.0})
        assert pairs[0].slo_s == pytest.approx(42.0)


# ===========================================================================
# Group B — _updated_cluster_state
# ===========================================================================


class TestUpdatedClusterState:
    def _router(self, tmp_path: Path, alpha: float = 0.7) -> QueryRouter:
        return _make_router(
            tmp_path, cluster_cache_state_update_alpha=alpha
        )

    def test_b1_alpha_zero(self, tmp_path: Path) -> None:
        router = self._router(tmp_path, alpha=0.0)
        current = np.array([1.0, 0.0, 0.0])
        vec = np.array([0.0, 1.0, 0.0])
        result = router._updated_cluster_state(current, vec)
        np.testing.assert_allclose(result, vec)

    def test_b2_alpha_one(self, tmp_path: Path) -> None:
        router = self._router(tmp_path, alpha=1.0)
        current = np.array([1.0, 0.0, 0.0])
        vec = np.array([0.0, 1.0, 0.0])
        result = router._updated_cluster_state(current, vec)
        np.testing.assert_allclose(result, current)

    def test_b3_default_alpha(self, tmp_path: Path) -> None:
        router = self._router(tmp_path, alpha=0.7)
        current = np.array([1.0, 0.0, 0.0])
        vec = np.array([0.0, 1.0, 0.0])
        result = router._updated_cluster_state(current, vec)
        expected = np.array([0.7, 0.3, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_b4_immutability(self, tmp_path: Path) -> None:
        router = self._router(tmp_path, alpha=0.5)
        current = np.array([1.0, 0.0])
        vec = np.array([0.0, 1.0])
        current_copy = current.copy()
        vec_copy = vec.copy()
        router._updated_cluster_state(current, vec)
        np.testing.assert_array_equal(current, current_copy)
        np.testing.assert_array_equal(vec, vec_copy)

    def test_b5_zero_current_state(self, tmp_path: Path) -> None:
        router = self._router(tmp_path, alpha=0.7)
        current = np.zeros(3)
        vec = np.array([1.0, 2.0, 3.0])
        result = router._updated_cluster_state(current, vec)
        np.testing.assert_allclose(result, 0.3 * vec)


# ===========================================================================
# Group C — _score_cache_risk
# ===========================================================================


class TestScoreCacheRisk:
    def _router(self, tmp_path: Path, coverage: float = 0.9) -> QueryRouter:
        return _make_router(tmp_path, cache_risk_coverage=coverage)

    def test_c1_no_forecast(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        caches = np.ones((2, 4))
        assert router._score_cache_risk(caches, forecasted_table_vecs=None) == 0.0

    def test_c2_perfect_match(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        vec = np.array([[1.0, 0.0, 0.0, 0.0]])
        caches = vec.copy()
        score = router._score_cache_risk(caches, forecasted_table_vecs=vec)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_c3_orthogonal(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        caches = np.array([[1.0, 0.0, 0.0, 0.0]])
        forecast = np.array([[0.0, 1.0, 0.0, 0.0]])
        score = router._score_cache_risk(caches, forecasted_table_vecs=forecast)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_c4_two_clusters_max_across(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        # cluster 0 matches perfectly; cluster 1 is orthogonal
        caches = np.array([[1.0, 0.0], [0.0, 1.0]])
        forecast = np.array([[1.0, 0.0]])  # aligns with cluster 0
        score = router._score_cache_risk(caches, forecasted_table_vecs=forecast)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_c5_coverage_zero_returns_float(self, tmp_path: Path) -> None:
        router = self._router(tmp_path, coverage=0.0)
        cache = np.array([[1.0, 0.0, 0.0]])
        forecast = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        score = router._score_cache_risk(cache, forecasted_table_vecs=forecast)
        assert isinstance(score, float)

    def test_c6_coverage_one_returns_float(self, tmp_path: Path) -> None:
        router = self._router(tmp_path, coverage=1.0)
        cache = np.array([[1.0, 0.0, 0.0]])
        forecast = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        score = router._score_cache_risk(cache, forecasted_table_vecs=forecast)
        assert isinstance(score, float)

    def test_c7_zero_cache_state_no_nan(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        caches = np.zeros((1, 4))
        forecast = np.array([[1.0, 0.0, 0.0, 0.0]])
        score = router._score_cache_risk(caches, forecasted_table_vecs=forecast)
        assert not np.isnan(score)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_c8_zero_forecast_vector_no_nan(self, tmp_path: Path) -> None:
        router = self._router(tmp_path)
        caches = np.array([[1.0, 0.0, 0.0, 0.0]])
        forecast = np.zeros((1, 4))
        score = router._score_cache_risk(caches, forecasted_table_vecs=forecast)
        assert not np.isnan(score)
        assert score == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Group D — select_best
# ===========================================================================


def _make_cache_aware_router(
    tmp_path: Path,
    **router_config_kwargs: Any,
) -> QueryRouter:
    """Build a CACHE_AWARE router for select_best tests.

    We pre-write a minimal .npz file so the constructor bypasses the real
    forecaster (which needs data files).
    """
    npz_path = tmp_path / "rel_time_s_to_forecasted_table_vecs.npz"
    np.savez(npz_path, **{"0.0": np.zeros((_N,))})
    return QueryRouter(
        slo_resolver=_make_slo_resolver(),
        slo_objective=_make_slo_objective(),
        query_router_config=_make_router_config("cache_aware", **router_config_kwargs),
        iconq_model=_make_iconq_stub(_N),
        out_dir=tmp_path,
    )


class TestSelectBest:
    def _mk_vc(self, v: float, c: float) -> ViolationCost:
        return ViolationCost(violation=v, cost=c)

    def test_d1_round_robin_cycles(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="round_robin")
        viols = {"A": self._mk_vc(0.1, 10.0), "B": self._mk_vc(0.2, 5.0)}
        risks = {"A": 0.0, "B": 0.0}
        results = [router.select_best(viols, risks) for _ in range(4)]
        assert results == ["A", "B", "A", "B"]

    def test_d2_round_robin_ignores_violation_cost(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="round_robin")
        viols = {"A": self._mk_vc(0.9, 100.0), "B": self._mk_vc(0.0, 0.0)}
        risks = {"A": 0.0, "B": 0.0}
        results = [router.select_best(viols, risks) for _ in range(2)]
        assert set(results) == {"A", "B"}

    def test_d3_uniform_random_valid_key(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="uniform_random")
        viols = {"A": self._mk_vc(0.1, 5.0), "B": self._mk_vc(0.2, 3.0)}
        risks = {"A": 0.0, "B": 0.0}
        for _ in range(20):
            result = router.select_best(viols, risks)
            assert result in {"A", "B"}

    def test_d4_iconq_both_feasible_picks_cheaper(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="use_iconq_model", threshold=0.5)
        viols = {"A": self._mk_vc(0.1, 5.0), "B": self._mk_vc(0.1, 10.0)}
        risks = {"A": 0.0, "B": 0.0}
        assert router.select_best(viols, risks) == "A"

    def test_d5_iconq_both_infeasible_picks_lower_violation(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="use_iconq_model", threshold=0.1)
        viols = {"A": self._mk_vc(0.8, 5.0), "B": self._mk_vc(0.5, 10.0)}
        risks = {"A": 0.0, "B": 0.0}
        assert router.select_best(viols, risks) == "B"

    def test_d6_iconq_feasible_beats_infeasible_even_if_costlier(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="use_iconq_model", threshold=0.2)
        viols = {"A": self._mk_vc(0.1, 100.0), "B": self._mk_vc(0.5, 1.0)}
        risks = {"A": 0.0, "B": 0.0}
        assert router.select_best(viols, risks) == "A"

    def test_d7_single_candidate(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="use_iconq_model")
        viols = {"only": self._mk_vc(0.1, 5.0)}
        risks = {"only": 0.0}
        assert router.select_best(viols, risks) == "only"

    def test_d8_equal_violation_cost_no_crash(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="use_iconq_model")
        viols = {"A": self._mk_vc(0.1, 5.0), "B": self._mk_vc(0.1, 5.0)}
        risks = {"A": 0.0, "B": 0.0}
        result = router.select_best(viols, risks)
        assert result in {"A", "B"}

    def test_d9_stage_model_same_as_iconq_selection(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="use_stage_model", threshold=0.5)
        viols = {"A": self._mk_vc(0.1, 5.0), "B": self._mk_vc(0.1, 10.0)}
        risks = {"A": 0.0, "B": 0.0}
        assert router.select_best(viols, risks) == "A"

    def test_d10_cache_aware_high_risk_inflates_cost(self, tmp_path: Path) -> None:
        router = _make_cache_aware_router(tmp_path, cache_risk_cost_multiplier=1.0)
        # A: adjusted = 10 * (1 + 1.0 * 0.9) = 19; B stays 11 → B wins
        viols = {"A": self._mk_vc(0.1, 10.0), "B": self._mk_vc(0.1, 11.0)}
        risks = {"A": 0.9, "B": 0.0}
        assert router.select_best(viols, risks) == "B"

    def test_d11_cache_aware_zero_multiplier_same_as_plain(self, tmp_path: Path) -> None:
        router = _make_cache_aware_router(tmp_path, cache_risk_cost_multiplier=0.0)
        viols = {"A": self._mk_vc(0.1, 5.0), "B": self._mk_vc(0.1, 10.0)}
        risks = {"A": 0.9, "B": 0.0}
        assert router.select_best(viols, risks) == "A"


def _snapshot_single(
    name: str = "autoslo-16-0-0",
    rpu: int = 16,
    *,
    queries: list[Query] | None = None,
    predicted_latencies: dict[str, float] | None = None,
) -> dict[str, ClusterView]:
    return {name: _make_cluster(name, rpu, queries=queries, predicted_latencies=predicted_latencies)}


class TestRouteQuery:
    """Unit tests for route_query. Each test patches emit_structured so that
    structured log events can be captured without touching the filesystem."""

    def _route(
        self,
        router: QueryRouter,
        query: Query,
        snapshot: dict[str, ClusterView],
        rel_time_s: float = 0.0,
    ) -> tuple[str, dict[str, float], np.ndarray, list[Any]]:
        emitted: list[Any] = []
        with patch(
            "autoslo.routing.query_router.emit_structured",
            side_effect=emitted.append,
        ):
            result = router.route_query(query, snapshot, rel_time_s)
        return result[0], result[1], result[2], emitted

    def test_e1_round_robin_single_cluster_returns_name(self, tmp_path: Path) -> None:
        q = _make_query("q1")
        pred = _make_pred(5.0)
        router = _make_router(
            tmp_path,
            predictions={ClusterAwareQueryId.make("autoslo-16-0-0", "q1"): pred},
        )
        snapshot = _snapshot_single()
        cluster_name, lat_dict, _, _ = self._route(router, q, snapshot)
        assert cluster_name == "autoslo-16-0-0"

    def test_e2_return_value_shape(self, tmp_path: Path) -> None:
        q = _make_query("q1")
        pred = _make_pred(5.0)
        router = _make_router(
            tmp_path,
            predictions={ClusterAwareQueryId.make("autoslo-16-0-0", "q1"): pred},
        )
        snapshot = _snapshot_single()
        cluster_name, lat_dict, new_state, _ = self._route(router, q, snapshot)
        assert isinstance(cluster_name, str)
        assert isinstance(lat_dict, dict)
        assert isinstance(new_state, np.ndarray)

    def test_e3_predicted_latency_non_negative(self, tmp_path: Path) -> None:
        q = _make_query("q1")
        pred = _make_pred(3.0)
        router = _make_router(
            tmp_path,
            predictions={ClusterAwareQueryId.make("autoslo-16-0-0", "q1"): pred},
        )
        snapshot = _snapshot_single()
        _, lat_dict, _, _ = self._route(router, q, snapshot)
        for v in lat_dict.values():
            assert v >= 0.0

    def test_e4_non_decreasing_constraint(self, tmp_path: Path) -> None:
        """When a cluster already has a higher predicted latency, the max is used."""
        existing = _make_query("existing", t=0.0)
        cluster_name = "autoslo-16-0-0"
        c = Cluster(
            creation_time_s=0.0,
            rpu=16,
            cache_state=np.zeros(_N),
            name=cluster_name,
            state=ClusterState.READY,
        )
        c.add_query(existing, {"existing": 10.0}, np.zeros(_N), {})
        view = ClusterView(c)
        snapshot = {cluster_name: view}

        q = _make_query("q_new", t=0.0)
        pred_existing = _make_pred(2.0)
        pred_new = _make_pred(5.0)
        router = _make_router(
            tmp_path,
            predictions={
                ClusterAwareQueryId.make("autoslo-16-0-0", "existing"): pred_existing,
                ClusterAwareQueryId.make("autoslo-16-0-0", "q_new"): pred_new,
            },
        )
        _, lat_dict, _, _ = self._route(router, q, snapshot)
        # "existing" had predicted_latency=10, model says 2 → non-decreasing → 10
        assert lat_dict.get("existing", 0.0) == pytest.approx(10.0)

    def test_e5_correct_cluster_lookup_after_fix(self, tmp_path: Path) -> None:
        """Regression test: after the key-mismatch fix route_query correctly
        iterates snapshot.keys(), not iconq_predictions.keys().

        Before the fix, this would raise KeyError because run_id != cluster_name.
        """
        q = _make_query("q1")
        pred = _make_pred(7.0)
        router = _make_router(
            tmp_path,
            policy="round_robin",
            predictions={ClusterAwareQueryId.make("autoslo-16-0-0", "q1"): pred},
        )
        snapshot = _snapshot_single()
        cluster_name, _, _, _ = self._route(router, q, snapshot)
        assert cluster_name == "autoslo-16-0-0"

    def test_e6_round_robin_two_clusters_alternates(self, tmp_path: Path) -> None:
        q1 = _make_query("q1")
        q2 = _make_query("q2")
        pred1 = _make_pred(4.0)
        pred2 = _make_pred(4.0)
        router = _make_router(
            tmp_path,
            policy="round_robin",
            predictions={
                ClusterAwareQueryId.make("autoslo-16-A-0", "q1"): pred1,
                ClusterAwareQueryId.make("autoslo-16-B-0", "q1"): pred1,
            },
        )
        snapshot = {
            "autoslo-16-A-0": _make_cluster("autoslo-16-A-0"),
            "autoslo-16-B-0": _make_cluster("autoslo-16-B-0"),
        }
        emitted1: list[Any] = []
        emitted2: list[Any] = []
        with patch("autoslo.routing.query_router.emit_structured", side_effect=emitted1.append):
            cn1 = router.route_query(q1, snapshot, 0.0)[0]
        router._iconq_model.predict_from_query_groups.return_value = {
            ClusterAwareQueryId.make("autoslo-16-A-0", "q2"): pred2,
            ClusterAwareQueryId.make("autoslo-16-B-0", "q2"): pred2,
        }
        with patch("autoslo.routing.query_router.emit_structured", side_effect=emitted2.append):
            cn2 = router.route_query(q2, snapshot, 1.0)[0]
        assert cn1 != cn2
        assert {cn1, cn2} == {"autoslo-16-A-0", "autoslo-16-B-0"}

    def test_e7_empty_snapshot_raises(self, tmp_path: Path) -> None:
        q = _make_query("q1")
        router = _make_router(tmp_path)
        with pytest.raises(Exception):
            router.route_query(q, {}, 0.0)

    def test_e8_routing_score_events_one_per_cluster(self, tmp_path: Path) -> None:
        q = _make_query("q1")
        pred = _make_pred(4.0)
        router = _make_router(
            tmp_path,
            predictions={
                ClusterAwareQueryId.make("autoslo-16-A-0", "q1"): pred,
                ClusterAwareQueryId.make("autoslo-16-B-0", "q1"): pred,
            },
        )
        snapshot = {
            "autoslo-16-A-0": _make_cluster("autoslo-16-A-0"),
            "autoslo-16-B-0": _make_cluster("autoslo-16-B-0"),
        }
        _, _, _, emitted = self._route(router, q, snapshot)
        score_events = [e for e in emitted if e.event_type == EventType.ROUTING_SCORE]
        assert len(score_events) == 2

    def test_e9_routing_event_emitted_once(self, tmp_path: Path) -> None:
        q = _make_query("q1")
        pred = _make_pred(4.0)
        router = _make_router(
            tmp_path,
            predictions={ClusterAwareQueryId.make("autoslo-16-0-0", "q1"): pred},
        )
        snapshot = _snapshot_single()
        _, _, _, emitted = self._route(router, q, snapshot)
        routing_events = [e for e in emitted if e.event_type == EventType.ROUTING]
        assert len(routing_events) == 1

    def test_e10_no_query_routed_event_from_route_query(self, tmp_path: Path) -> None:
        """QUERY_ROUTED is emitted by route_and_update_bookkeeping, not route_query."""
        q = _make_query("q1")
        pred = _make_pred(4.0)
        router = _make_router(
            tmp_path,
            predictions={ClusterAwareQueryId.make("autoslo-16-0-0", "q1"): pred},
        )
        snapshot = _snapshot_single()
        _, _, _, emitted = self._route(router, q, snapshot)
        assert not any(
            getattr(e, "event_type", None) == EventType.QUERY_ROUTED
            for e in emitted
        )

    def test_e11_use_stage_model_no_crash(self, tmp_path: Path) -> None:
        """USE_STAGE_MODEL still calls the iconq model but uses stage latencies
        for routing pairs; the route_query call must not raise."""
        stage_preds = {16: 8.0}
        q = _make_query("q1", stage_preds=stage_preds)
        pred = _make_pred(4.0)
        router = _make_router(
            tmp_path,
            policy="use_stage_model",
            predictions={ClusterAwareQueryId.make("autoslo-16-0-0", "q1"): pred},
        )
        snapshot = _snapshot_single()
        cluster_name, _, _, _ = self._route(router, q, snapshot)
        assert cluster_name == "autoslo-16-0-0"


# ===========================================================================
# Group F — Constructor behaviour
# ===========================================================================


class TestQueryRouterConstructor:
    def test_f1_round_robin_no_file_io(self, tmp_path: Path) -> None:
        router = _make_router(tmp_path, policy="round_robin")
        assert router._rel_time_s_to_forecasted_table_vecs == {}

    def test_f2_non_cache_aware_out_dir_stays_empty(self, tmp_path: Path) -> None:
        _make_router(tmp_path, policy="round_robin")
        assert list(tmp_path.iterdir()) == []

    def test_f3_cache_aware_without_forecaster_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="forecaster_config"):
            QueryRouter(
                slo_resolver=_make_slo_resolver(),
                slo_objective=_make_slo_objective(),
                query_router_config=QueryRouterConfig(
                    routing_policy_name="cache_aware",
                    forecaster_config=None,
                ),
                iconq_model=_make_iconq_stub(_N),
                out_dir=tmp_path,
            )


# ===========================================================================
# Group G — Integration tests (require data files)
# ===========================================================================

_INTEGRATION_MODEL_ID = "1771539369"


@pytest.mark.integration
class TestQueryRouterIntegration:
    """These tests load a real IconqModel from disk.  They are slow and require
    data files to be present.  Skip automatically if the model directory is
    absent.
    """

    @pytest.fixture(autouse=True)
    def _check_model(self) -> None:
        import autoslo.filesystem.path_utils as pu

        model_dir = pu.get_data_dir() / "iconq_models" / _INTEGRATION_MODEL_ID
        if not model_dir.is_dir():
            pytest.skip(
                f"IconqModel {_INTEGRATION_MODEL_ID!r} not found at {model_dir}. "
                "Set a valid model ID to run integration tests."
            )

    @pytest.fixture
    def real_router(self, tmp_path: Path) -> QueryRouter:
        from autoslo.models.iconq_model import IconqModel

        model = IconqModel.load(_INTEGRATION_MODEL_ID)
        return QueryRouter(
            slo_resolver=_make_slo_resolver(default_slo_s=30.0),
            slo_objective=_make_slo_objective(threshold=0.5),
            query_router_config=_make_router_config("round_robin"),
            iconq_model=model,
            out_dir=tmp_path,
        )

    @pytest.fixture
    def model_n(self, real_router: QueryRouter) -> int:
        return real_router._iconq_model.iconq_query_featurizer.num_tables

    def test_g1_single_cluster_returns_valid_tuple(
        self, real_router: QueryRouter, model_n: int
    ) -> None:
        q = _make_query("q1")
        snapshot = {"autoslo-16-0-0": _make_cluster("autoslo-16-0-0", n=model_n)}
        with patch("autoslo.routing.query_router.emit_structured"):
            cn, lat_dict, new_state = real_router.route_query(q, snapshot, 0.0)
        assert isinstance(cn, str) and cn.startswith("autoslo-")
        assert isinstance(lat_dict, dict)
        assert isinstance(new_state, np.ndarray)

    def test_g2_two_clusters_round_robin_alternates(
        self, real_router: QueryRouter, model_n: int
    ) -> None:
        q1 = _make_query("q1")
        q2 = _make_query("q2")
        snapshot = {
            "autoslo-16-A-0": _make_cluster("autoslo-16-A-0", n=model_n),
            "autoslo-16-B-0": _make_cluster("autoslo-16-B-0", n=model_n),
        }
        with patch("autoslo.routing.query_router.emit_structured"):
            cn1 = real_router.route_query(q1, snapshot, 0.0)[0]
            cn2 = real_router.route_query(q2, snapshot, 1.0)[0]
        assert cn1 != cn2

    def test_g3_non_decreasing_latency_constraint(
        self, real_router: QueryRouter, model_n: int
    ) -> None:
        existing = _make_query("existing", t=0.0)
        cluster_name = "autoslo-16-0-0"
        c = Cluster(
            creation_time_s=0.0,
            rpu=16,
            cache_state=np.zeros(model_n),
            name=cluster_name,
            state=ClusterState.READY,
        )
        c.add_query(existing, {"existing": 20.0}, np.zeros(model_n))
        view = ClusterView(c)
        snapshot = {cluster_name: view}

        q = _make_query("q_new", t=1.0)
        with patch("autoslo.routing.query_router.emit_structured"):
            _, lat_dict, _ = real_router.route_query(q, snapshot, 1.0)

        for q_id, lat in lat_dict.items():
            existing_lat = view.predicted_latencies.get(q_id, 0.0)
            assert lat >= existing_lat - 1e-6, (
                f"Non-decreasing violated: lat_dict[{q_id!r}]={lat} "
                f"< existing {existing_lat}"
            )
