"""Tests for SloResolver.tightened() and Autoscaler.slo_tightening_factor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import Query, QueryTextId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query(
    query_id: str = "q1",
    template_id: str = "001",
    rel_start_time_s: float = 0.0,
) -> Query:
    return Query(
        query_id=query_id,
        query_text_id=QueryTextId(f"public#{template_id}#0"),
        rel_start_time_s=rel_start_time_s,
    )


def _ready_cluster(
    rpu: int = 8,
    name: str | None = None,
    creation_time_s: float = 0.0,
) -> Cluster:
    if name is None:
        name = f"autoslo-{rpu}-test"
    cluster = Cluster(
        creation_time_s=creation_time_s,
        rpu=rpu,
        name=name,
    )
    cluster.state = ClusterState.READY
    return cluster


def _autoscaler(
    slo_s: float = 10.0,
    slo_metric: SloMetric = SloMetric.RELATIVE,
    slo_threshold: float = 0.0,
    slo_tightening_factor: float = 1.0,
    min_observations_to_act: int = 1,
    allowed_rpu_sizes: list[int] | None = None,
    observation_window_s: float = 600.0,
) -> Autoscaler:
    return Autoscaler(
        slo_resolver=SloResolver(default_slo_s=slo_s),
        slo_objective=SloObjective(
            slo_metric=slo_metric,
            slo_threshold=slo_threshold,
        ),
        allowed_rpu_sizes=allowed_rpu_sizes or [8],
        iconq_model=MagicMock(),
        min_observations_to_act=min_observations_to_act,
        observation_window_s=observation_window_s,
        slo_tightening_factor=slo_tightening_factor,
    )


# ---------------------------------------------------------------------------
# SloResolver.tightened()
# ---------------------------------------------------------------------------


class TestSloResolverTightened:
    def test_scales_default(self):
        resolver = SloResolver(default_slo_s=10.0)
        tightened = resolver.tightened(0.8)
        assert tightened.default_slo_s == pytest.approx(8.0)

    def test_scales_overrides(self):
        resolver = SloResolver.from_dict(
            default_slo_s=10.0,
            slo_dict={"001": 5.0, "002": 20.0},
        )
        tightened = resolver.tightened(0.5)
        assert tightened.default_slo_s == pytest.approx(5.0)
        assert tightened.slo_dict["001"] == pytest.approx(2.5)
        assert tightened.slo_dict["002"] == pytest.approx(10.0)

    def test_factor_1_is_identity(self):
        resolver = SloResolver.from_dict(
            default_slo_s=10.0,
            slo_dict={"001": 5.0},
        )
        tightened = resolver.tightened(1.0)
        assert tightened.default_slo_s == resolver.default_slo_s
        assert tightened.resolve("001") == resolver.resolve("001")

    def test_rejects_zero(self):
        resolver = SloResolver(default_slo_s=10.0)
        with pytest.raises(ValueError):
            resolver.tightened(0.0)

    def test_rejects_negative(self):
        resolver = SloResolver(default_slo_s=10.0)
        with pytest.raises(ValueError):
            resolver.tightened(-0.5)


# ---------------------------------------------------------------------------
# Autoscaler with slo_tightening_factor
# ---------------------------------------------------------------------------


class TestAutoscalerTightening:
    """Test that consider_spin_up uses tightened SLOs for its trigger."""

    def _snapshot_with_query(
        self,
        pred_latency: float,
        query: Query | None = None,
    ) -> tuple[dict[str, Cluster], Query]:
        """Build a single-cluster snapshot with one active query."""
        query = query or _make_query()
        cluster = _ready_cluster()
        cluster.add_query(query)
        cluster.predicted_latencies[query.query_id] = pred_latency
        return {cluster.name: cluster}, query

    def test_factor_1_no_spinup_when_within_slo(self):
        """SLO=10s, pred=8.5s → within SLO, no spin-up with factor=1.0."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=1.0)
        snapshot, query = self._snapshot_with_query(pred_latency=8.5)

        # Seed the window with one query so consider_spin_up proceeds.
        scaler._window_start_time_s = 0.0
        scaler._snapshot_at_window_start = snapshot
        scaler._window_queries = [query]

        actions = scaler.consider_spin_up(
            rel_time_s=1.0,
            pool_snapshot_with_current_query=snapshot,
        )
        assert actions == []

    @patch.object(Autoscaler, "_select_rpu", return_value=8)
    def test_tightened_triggers_spinup(self, _mock_rpu):
        """SLO=10s, pred=8.5s, factor=0.8 → tightened SLO=8s → violation → spin-up."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=0.8)
        snapshot, query = self._snapshot_with_query(pred_latency=8.5)

        scaler._window_start_time_s = 0.0
        scaler._snapshot_at_window_start = snapshot
        scaler._window_queries = [query]

        actions = scaler.consider_spin_up(
            rel_time_s=1.0,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1
        assert "slo_tightening_factor=0.8" in actions[0].reason

    def test_tightened_no_spinup_when_well_within(self):
        """SLO=10s, pred=5s, factor=0.8 → tightened SLO=8s → still met → no spin-up."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=0.8)
        snapshot, query = self._snapshot_with_query(pred_latency=5.0)

        scaler._window_start_time_s = 0.0
        scaler._snapshot_at_window_start = snapshot
        scaler._window_queries = [query]

        actions = scaler.consider_spin_up(
            rel_time_s=1.0,
            pool_snapshot_with_current_query=snapshot,
        )
        assert actions == []

    @patch.object(Autoscaler, "_select_rpu", return_value=8)
    def test_factor_1_still_triggers_on_real_violation(self, _mock_rpu):
        """SLO=10s, pred=12s, factor=1.0 → real violation → spin-up."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=1.0)
        snapshot, query = self._snapshot_with_query(pred_latency=12.0)

        scaler._window_start_time_s = 0.0
        scaler._snapshot_at_window_start = snapshot
        scaler._window_queries = [query]

        actions = scaler.consider_spin_up(
            rel_time_s=1.0,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1

    def test_property_exposed(self):
        scaler = _autoscaler(slo_tightening_factor=0.75)
        assert scaler.slo_tightening_factor == 0.75

    def test_default_factor_is_1(self):
        scaler = _autoscaler()
        assert scaler.slo_tightening_factor == 1.0

    @patch.object(Autoscaler, "_select_rpu", return_value=8)
    def test_with_binary_metric(self, _mock_rpu):
        """Factor works with BINARY metric: SLO=10s, pred=8.5s, factor=0.8
        → tightened SLO=8s → binary violation → spin-up."""
        scaler = _autoscaler(
            slo_s=10.0,
            slo_metric=SloMetric.BINARY,
            slo_threshold=0.0,
            slo_tightening_factor=0.8,
        )
        snapshot, query = self._snapshot_with_query(pred_latency=8.5)

        scaler._window_start_time_s = 0.0
        scaler._snapshot_at_window_start = snapshot
        scaler._window_queries = [query]

        actions = scaler.consider_spin_up(
            rel_time_s=1.0,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1

    @patch.object(Autoscaler, "_select_rpu", return_value=8)
    def test_per_template_overrides_tightened(self, _mock_rpu):
        """Per-template SLOs are also tightened."""
        resolver = SloResolver.from_dict(
            default_slo_s=10.0,
            slo_dict={"001": 5.0},  # template 001 has tighter SLO
        )
        scaler = Autoscaler(
            slo_resolver=resolver,
            slo_objective=SloObjective(
                slo_metric=SloMetric.RELATIVE,
                slo_threshold=0.0,
            ),
            allowed_rpu_sizes=[8],
            iconq_model=MagicMock(),
            min_observations_to_act=1,
            observation_window_s=600.0,
            slo_tightening_factor=0.8,
        )
        # Template 001 real SLO=5s, tightened SLO=4s, pred=4.5s → violation
        query = _make_query(template_id="001")
        snapshot, _ = self._snapshot_with_query(pred_latency=4.5, query=query)

        scaler._window_start_time_s = 0.0
        scaler._snapshot_at_window_start = snapshot
        scaler._window_queries = [query]

        actions = scaler.consider_spin_up(
            rel_time_s=1.0,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1
