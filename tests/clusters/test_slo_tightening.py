"""Tests for SloResolver.tightened(), Autoscaler.slo_tightening_factor,
and Autoscaler trigger_slo_objective_config separation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from autoslo.clusters.actions import SpinUpAction
from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.cluster import ClusterState, ClusterView
from autoslo.config.component_configs import (
    AutoscalerConfig,
    ProvisionerConfig,
    QueryRouterConfig,
    SloObjectiveConfig,
    SloResolverConfig,
)
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import Query, QueryTextId


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# rel_time_s used in consider_spin_up tests.  Must be >= observation_window_s
# so that the window-start cutoff (rel_time_s − obs_window) ≥ 0, which is
# not strictly greater than the default _most_recent_cluster_ready_rel_time_s
# of 0.0, letting the SLO check proceed.
_REL_TIME_S = 700.0  # with observation_window_s=600.0

_DUMMY_PROVISIONER = ProvisionerConfig(
    aws_config_path="/dev/null",
    cluster_cache_state_dim=1,
    run_id="test",
)


def _make_query(
    query_id: str = "q1",
    template_id: str = "001",
    rel_start_time_s: float = 100.0,
) -> Query:
    return Query(
        query_id=query_id,
        query_text_id=QueryTextId(f"public#{template_id}#0"),
        rel_start_time_s=rel_start_time_s,
    )


def _view_with_query(
    pred_latency: float,
    query: Query | None = None,
    rpu: int = 8,
    name: str = "autoslo-8-test",
) -> tuple[dict[str, ClusterView], Query]:
    """Return a single-cluster snapshot (ClusterView) with one active query."""
    q = query or _make_query()
    view = ClusterView(
        creation_time_s=0.0,
        rpu=rpu,
        name=name,
        state=ClusterState.READY,
        queries={q.query_id: q},
        predicted_latencies={q.query_id: pred_latency},
    )
    return {name: view}, q


def _make_resolver(slo_s: float = 10.0) -> SloResolver:
    return SloResolver(SloResolverConfig(slo_s=slo_s))


def _autoscaler(
    slo_s: float = 10.0,
    slo_metric: SloMetric | str = SloMetric.RELATIVE,
    slo_threshold: float = 0.0,
    slo_tightening_factor: float = 1.0,
    trigger_slo_objective_config: SloObjectiveConfig | None = None,
    allowed_rpu_sizes: list[int] | None = None,
    observation_window_s: float = 600.0,
    slo_resolver: SloResolver | None = None,
) -> Autoscaler:
    return Autoscaler(
        slo_resolver=slo_resolver or _make_resolver(slo_s),
        slo_objective=SloObjective(
            SloObjectiveConfig(slo_metric=slo_metric, slo_threshold=slo_threshold)
        ),
        provisioner_config=_DUMMY_PROVISIONER,
        query_router_config=QueryRouterConfig(),
        autoscaler_config=AutoscalerConfig(
            allowed_rpu_sizes=allowed_rpu_sizes or [8],
            observation_window_s=observation_window_s,
            slo_tightening_factor=slo_tightening_factor,
            trigger_slo_objective_config=trigger_slo_objective_config,
        ),
        out_dir="/tmp",
        iconq_model=MagicMock(),
    )



# ---------------------------------------------------------------------------
# SloResolver.tightened()
# ---------------------------------------------------------------------------


class TestSloResolverTightened:
    def test_scales_default(self):
        resolver = _make_resolver(10.0)
        tightened = resolver.tightened(0.8)
        assert tightened.default_slo_s == pytest.approx(8.0)

    def test_scales_by_half(self):
        resolver = _make_resolver(10.0)
        tightened = resolver.tightened(0.5)
        assert tightened.default_slo_s == pytest.approx(5.0)

    def test_factor_1_is_identity(self):
        resolver = _make_resolver(10.0)
        tightened = resolver.tightened(1.0)
        assert tightened.default_slo_s == pytest.approx(resolver.default_slo_s)

    def test_rejects_zero(self):
        resolver = _make_resolver(10.0)
        with pytest.raises(ValueError):
            resolver.tightened(0.0)

    def test_rejects_negative(self):
        resolver = _make_resolver(10.0)
        with pytest.raises(ValueError):
            resolver.tightened(-0.5)


# ---------------------------------------------------------------------------
# Autoscaler with slo_tightening_factor
# ---------------------------------------------------------------------------


class TestAutoscalerTightening:
    """Test that consider_spin_up uses tightened SLOs for its trigger."""

    def test_factor_1_no_spinup_when_within_slo(self):
        """SLO=10s, pred=8.5s → within SLO, no spin-up with factor=1.0."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=1.0)
        snapshot, _ = _view_with_query(pred_latency=8.5)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert actions == []

    @patch.object(Autoscaler, "_select_rpu", return_value=(8, MagicMock()))
    def test_tightened_triggers_spinup(self, _):
        """SLO=10s, pred=8.5s, factor=0.8 → tightened SLO=8s → violation → spin-up."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=0.8)
        snapshot, _ = _view_with_query(pred_latency=8.5)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1
        assert "slo_tightening_factor=0.8" in actions[0].reason

    def test_tightened_no_spinup_when_well_within(self):
        """SLO=10s, pred=5s, factor=0.8 → tightened SLO=8s → still met → no spin-up."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=0.8)
        snapshot, _ = _view_with_query(pred_latency=5.0)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert actions == []

    @patch.object(Autoscaler, "_select_rpu", return_value=(8, MagicMock()))
    def test_factor_1_still_triggers_on_real_violation(self, _):
        """SLO=10s, pred=12s, factor=1.0 → real violation → spin-up."""
        scaler = _autoscaler(slo_s=10.0, slo_tightening_factor=1.0)
        snapshot, _ = _view_with_query(pred_latency=12.0)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1

    def test_property_exposed(self):
        scaler = _autoscaler(slo_tightening_factor=0.75)
        assert scaler.slo_tightening_factor == 0.75

    def test_default_factor_is_1(self):
        scaler = _autoscaler()
        assert scaler.slo_tightening_factor == 1.0

    @patch.object(Autoscaler, "_select_rpu", return_value=(8, MagicMock()))
    def test_with_binary_metric(self, _):
        """Factor works with BINARY metric: SLO=10s, pred=8.5s, factor=0.8
        → tightened SLO=8s → binary violation → spin-up."""
        scaler = _autoscaler(
            slo_s=10.0,
            slo_metric=SloMetric.BINARY,
            slo_threshold=0.0,
            slo_tightening_factor=0.8,
        )
        snapshot, _ = _view_with_query(pred_latency=8.5)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1

    @patch.object(Autoscaler, "_select_rpu", return_value=(8, MagicMock()))
    def test_tighter_base_slo_is_tightened(self, _):
        """A smaller base SLO is also tightened by the factor."""
        scaler = _autoscaler(
            slo_s=5.0,  # SLO=5s
            slo_metric=SloMetric.RELATIVE,
            slo_threshold=0.0,
            slo_tightening_factor=0.8,  # effective tightened SLO=4s
        )
        # pred=4.5s > tightened SLO=4s → violation
        snapshot, _ = _view_with_query(pred_latency=4.5)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1


# ---------------------------------------------------------------------------
# Autoscaler trigger_slo_objective_config
# ---------------------------------------------------------------------------


class TestTriggerSloObjective:
    """Test that trigger_slo_objective_config separates trigger from routing."""

    def test_default_trigger_objective_is_routing_objective(self):
        """When not configured, _trigger_slo_objective is the same object as _slo_objective."""
        scaler = _autoscaler()
        assert scaler._trigger_slo_objective is scaler._slo_objective

    def test_trigger_objective_set_when_configured(self):
        """When configured, _trigger_slo_objective is a separate SloObjective instance."""
        trigger_cfg = SloObjectiveConfig(slo_metric="binary", slo_threshold=0.02)
        scaler = _autoscaler(trigger_slo_objective_config=trigger_cfg)
        assert scaler._trigger_slo_objective is not scaler._slo_objective
        assert scaler._trigger_slo_objective.slo_metric == SloMetric.BINARY
        assert scaler._trigger_slo_objective.slo_threshold == pytest.approx(0.02)

    def test_routing_objective_unchanged_when_trigger_configured(self):
        """_slo_objective is not affected by trigger_slo_objective_config."""
        trigger_cfg = SloObjectiveConfig(slo_metric="binary", slo_threshold=0.02)
        scaler = _autoscaler(
            slo_metric=SloMetric.RELATIVE,
            slo_threshold=0.05,
            trigger_slo_objective_config=trigger_cfg,
        )
        assert scaler._slo_objective.slo_metric == SloMetric.RELATIVE
        assert scaler._slo_objective.slo_threshold == pytest.approx(0.05)

    def test_no_spinup_without_trigger_when_routing_is_permissive(self):
        """Routing SLO too permissive to trigger → no spin-up without separate trigger."""
        # relative threshold=1.0: a 10% overshoot is well within SLO
        scaler = _autoscaler(slo_metric=SloMetric.RELATIVE, slo_threshold=1.0)
        snapshot, _ = _view_with_query(pred_latency=11.0)  # 10% overshoot
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert actions == []

    @patch.object(Autoscaler, "_select_rpu", return_value=(8, MagicMock()))
    def test_trigger_fires_when_routing_would_not(self, _):
        """Trigger SLO (binary, threshold=0) fires even when routing SLO is met."""
        scaler = _autoscaler(
            slo_metric=SloMetric.RELATIVE,
            slo_threshold=1.0,  # routing: very permissive
            trigger_slo_objective_config=SloObjectiveConfig(
                slo_metric="binary",
                slo_threshold=0.0,  # trigger: any violation fires
            ),
        )
        snapshot, _ = _view_with_query(pred_latency=11.0)  # exceeds SLO=10s
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1
        assert isinstance(actions[0], SpinUpAction)

    @patch.object(Autoscaler, "_select_rpu", return_value=(8, MagicMock()))
    def test_reason_string_contains_trigger_field_names(self, _):
        """SpinUpAction.reason logs trigger_slo_metric and trigger_slo_threshold."""
        scaler = _autoscaler(
            slo_s=10.0,
            slo_tightening_factor=0.8,
            trigger_slo_objective_config=SloObjectiveConfig(
                slo_metric="binary", slo_threshold=0.0
            ),
        )
        # pred=8.5s, tightened SLO=8s → binary violation
        snapshot, _ = _view_with_query(pred_latency=8.5)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1
        assert "trigger_slo_metric=" in actions[0].reason
        assert "trigger_slo_threshold=" in actions[0].reason
        assert "slo_tightening_factor=0.8" in actions[0].reason

    @patch.object(Autoscaler, "_select_rpu", return_value=(8, MagicMock()))
    def test_tightening_and_trigger_objective_combine(self, _):
        """slo_tightening_factor and trigger_slo_objective_config are orthogonal."""
        # factor=0.8 tightens per-query SLO; trigger uses binary metric
        scaler = _autoscaler(
            slo_s=10.0,
            slo_tightening_factor=0.8,
            trigger_slo_objective_config=SloObjectiveConfig(
                slo_metric="binary", slo_threshold=0.0
            ),
        )
        # pred=8.5s, tightened SLO=8s → binary violation despite < untightened SLO
        snapshot, _ = _view_with_query(pred_latency=8.5)
        actions = scaler.consider_spin_up(
            rel_time_s=_REL_TIME_S,
            pool_snapshot_with_current_query=snapshot,
        )
        assert len(actions) == 1

    def test_yaml_absent_trigger_config_is_none(self):
        """Parsing a YAML without trigger_slo_objective_config gives None."""
        cfg = {"autoscaler_config": {}}
        config = AutoscalerConfig.from_config(cfg)
        assert config.trigger_slo_objective_config is None

    def test_yaml_roundtrip(self):
        """trigger_slo_objective_config survives from_config() parse."""
        cfg = {
            "autoscaler_config": {
                "trigger_slo_objective_config": {
                    "slo_metric": "binary",
                    "slo_threshold": 0.02,
                }
            }
        }
        config = AutoscalerConfig.from_config(cfg)
        assert config.trigger_slo_objective_config is not None
        assert config.trigger_slo_objective_config.slo_threshold == pytest.approx(0.02)
        assert config.trigger_slo_objective_config.slo_metric == "binary"

