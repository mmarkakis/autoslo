"""Smoke tests for CacheAwarePolicy wired through the config factory.

Verifies that ``routing_policy: cache_aware`` in the config correctly
instantiates a :class:`CacheAwarePolicy` with all parameters propagated,
and that routing a query through it produces valid results.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from autoslo.routing.cache_aware_policy import CacheAwarePolicy
from autoslo.routing.model_policy import ModelPolicy
from autoslo.routing.routing_core import PlacementScore, RoutingCore
from autoslo.workload_definition.query import Query, QueryTextId, SloMetric


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_TABLE = 3
M_OPERATOR = 2


def _mock_iconq_model() -> MagicMock:
    model = MagicMock()
    featurizer = MagicMock()
    featurizer._m = M_OPERATOR
    featurizer._n = N_TABLE

    def _featurize(qtid):
        ops = [0.0] * (2 * M_OPERATOR)
        return ops + [1.0, 0.0, 0.0]  # always table-0

    featurizer.featurize_from_query_text_id.side_effect = _featurize
    model.iconq_query_featurizer = featurizer
    model.iconq_interaction_featurizer = MagicMock()
    model.stage_model = MagicMock()
    return model


def _write_yamls(tmp_path: str) -> tuple[str, str]:
    forecast = {
        "schema_name": "smoke",
        "window_minutes": 60,
        "bins": [
            {
                "day_of_week": 0,
                "hour": 9,
                "templates": [
                    {"template_id": "1", "probability": 1.0},
                ],
            },
        ],
    }
    tightness = {
        "schema_name": "smoke",
        "reference_rpu": 8,
        "stage_model_id": "test",
        "slo_source": "test",
        "entries": {
            "1": {"isolated_prediction_s": 5.0, "slo_s": 10.0, "tightness": 0.5},
        },
    }
    fp = os.path.join(tmp_path, "forecast.yml")
    tp = os.path.join(tmp_path, "tightness.yml")
    with open(fp, "w") as f:
        yaml.dump(forecast, f, sort_keys=False)
    with open(tp, "w") as f:
        yaml.dump(tightness, f, sort_keys=False)
    return fp, tp


# ---------------------------------------------------------------------------
# Config factory smoke tests
# ---------------------------------------------------------------------------


class TestConfigFactoryConstructsCacheAwarePolicy:
    """Verify the factory branch in workload_simulator / workload_runner
    creates a CacheAwarePolicy with the right parameters."""

    def test_factory_creates_cache_aware_policy(self, tmp_path):
        fp, tp = _write_yamls(str(tmp_path))
        mock_model = _mock_iconq_model()

        routing_cfg = {
            "routing_policy": "cache_aware",
            "forecast_distribution_path": fp,
            "slo_tightness_path": tp,
            "cache_risk_lambda": 2.5,
            "cache_decay_strategy": "sliding_window",
            "cache_decay_params": {"max_queries": 10},
            "fallback_tightness": 0.3,
        }

        # Replicate the factory branch from workload_simulator.py.
        from autoslo.blueprint_selection.slo_resolver import SloResolver
        slo_resolver = SloResolver(10.0, None)

        with patch(
            "autoslo.routing.model_policy.IconqModel.load",
            return_value=mock_model,
        ):
            policy = CacheAwarePolicy(
                iconq_model_id="test_model",
                default_slo_s=10.0,
                slo_overrides=slo_resolver.slo_dict,
                slo_metric=SloMetric.RELATIVE,
                forecast_distribution_path=routing_cfg["forecast_distribution_path"],
                slo_tightness_path=routing_cfg["slo_tightness_path"],
                cache_risk_lambda=float(routing_cfg.get("cache_risk_lambda", 0.0)),
                cache_decay_strategy=routing_cfg.get("cache_decay_strategy", "exponential"),
                cache_decay_params=routing_cfg.get("cache_decay_params", {}),
                fallback_tightness=float(routing_cfg.get("fallback_tightness", 0.5)),
            )

        assert isinstance(policy, CacheAwarePolicy)
        assert isinstance(policy, ModelPolicy)
        assert policy._lambda == 2.5
        assert policy._decay_strategy_kind == "sliding_window"
        assert policy._n == N_TABLE
        assert policy._m == M_OPERATOR

    def test_cache_aware_is_instance_of_model_policy(self, tmp_path):
        """Ensures isinstance checks (e.g. in simulator RPU wiring) work."""
        fp, tp = _write_yamls(str(tmp_path))
        mock_model = _mock_iconq_model()

        with patch(
            "autoslo.routing.model_policy.IconqModel.load",
            return_value=mock_model,
        ):
            policy = CacheAwarePolicy(
                iconq_model_id="test",
                default_slo_s=10.0,
                forecast_distribution_path=fp,
                slo_tightness_path=tp,
                cache_risk_lambda=0.0,
            )

        assert isinstance(policy, ModelPolicy)


class TestEndToEndRouting:
    """Route multiple queries through CacheAwarePolicy and verify state."""

    def test_three_queries_accumulate_cache_state(self, tmp_path):
        fp, tp = _write_yamls(str(tmp_path))
        mock_model = _mock_iconq_model()

        with patch(
            "autoslo.routing.model_policy.IconqModel.load",
            return_value=mock_model,
        ):
            policy = CacheAwarePolicy(
                iconq_model_id="test",
                default_slo_s=10.0,
                forecast_distribution_path=fp,
                slo_tightness_path=tp,
                cache_risk_lambda=1.0,
                cache_decay_strategy="exponential",
                cache_decay_params={"alpha": 0.5},
            )

        pool = MagicMock()
        pool.cluster_names = ["c0", "c1"]
        pool.ready_cluster_names = ["c0", "c1"]
        pool.get_rpu.return_value = 8
        policy.on_attach(pool)

        # Route 3 queries, alternating between c0 and c1 as best.
        for i, best_cluster in enumerate(["c0", "c1", "c0"]):
            scores = {
                best_cluster: PlacementScore(
                    cluster_name=best_cluster,
                    marginal_slo_violation=0.1,
                    marginal_cost=10.0,
                    latencies={f"q{i}": 2.0},
                ),
            }
            # Only offer the "best" cluster for simplicity.

            def _fake(
                iconq_model, pool, query_id, query_text_id, start_time_s,
                slo_resolver, slo_metric, current_latencies,
                cluster_names=None, _scores=scores,
            ):
                incoming = Query(
                    query_id=query_id,
                    query_text_id=query_text_id,
                    rel_start_time_s=start_time_s,
                )
                return _scores, incoming, {cn: 1.0 for cn in _scores}

            with patch(
                "autoslo.routing.cache_aware_policy.RoutingCore.score_query_on_clusters",
                side_effect=_fake,
            ):
                result = policy.route_with_details(
                    query_id=f"q{i}",
                    query_text_id="smoke#1#000",
                    start_time_s=float(i * 100),
                    pool=pool,
                )

            assert result.cluster_name == best_cluster

        # After 3 routes, cache states should be non-zero for used clusters.
        c0_state = policy._cluster_caches["c0"].current_state(300.0)
        assert np.any(c0_state != 0.0)
