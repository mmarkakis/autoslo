import numpy as np
import pytest

from autoslo.blueprints.cluster import Cluster
from autoslo.capacity.headroom_policy import HeadroomPolicy
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver


class TestHeadroomPolicy:

    @pytest.mark.parametrize("seed", range(10))
    def test_property_initialization_getters_setters(self, seed):

        np.random.seed(seed)
        default_slo_s = np.random.uniform(1.0, 20.0)
        slo_threshold = np.random.uniform(0.1, 10.0)
        eta_crit = np.random.uniform(0.01, 0.5)
        idle_periods_before_teardown = np.random.randint(1, 10)
        min_cluster_lifetime_s = np.random.uniform(30.0, 300.0)
        all_cluster_sizes = Cluster.all_allowed_rpu_sizes()
        allowed_idxs = np.random.choice(
            len(all_cluster_sizes),
            size=np.random.randint(1, len(all_cluster_sizes) + 1),
            replace=False,
        )
        allowed_rpu_sizes = [all_cluster_sizes[i] for i in allowed_idxs]

        policy = HeadroomPolicy(
            slo_resolver=SloResolver(default_slo_s=default_slo_s),
            slo_objective=SloObjective(
                slo_metric="absolute_s", slo_threshold=slo_threshold
            ),
            eta_crit=eta_crit,
            idle_periods_before_tear_down=idle_periods_before_teardown,
            min_cluster_lifetime_s=min_cluster_lifetime_s,
            allowed_rpu_sizes=allowed_rpu_sizes,
        )

        assert policy.name == f"HeadroomPolicy(eta_crit={eta_crit:.3f})"
        assert policy.eta_crit == eta_crit
        assert (
            policy.idle_periods_before_tear_down == idle_periods_before_teardown
        )
        assert policy.min_cluster_lifetime_s == min_cluster_lifetime_s
        assert policy.pending_count == 0
        assert policy.allowed_rpu_sizes == sorted(allowed_rpu_sizes)

        # Use setters to change some properties and check they update correctly
        new_eta_crit = np.random.uniform(0.01, 0.5)
        policy.eta_crit = new_eta_crit
        assert policy.eta_crit == new_eta_crit

        new_idle_periods = np.random.randint(1, 10)
        policy.idle_periods_before_tear_down = new_idle_periods
        assert policy.idle_periods_before_tear_down == new_idle_periods

        new_allowed_idxs = np.random.choice(
            len(all_cluster_sizes),
            size=np.random.randint(1, len(all_cluster_sizes) + 1),
            replace=False,
        )
        new_allowed_rpu_sizes = [all_cluster_sizes[i] for i in new_allowed_idxs]
        policy.allowed_rpu_sizes = new_allowed_rpu_sizes
        assert policy.allowed_rpu_sizes == sorted(new_allowed_rpu_sizes)

        new_min_lifetime = np.random.uniform(30.0, 300.0)
        policy.min_cluster_lifetime_s = new_min_lifetime
        assert policy.min_cluster_lifetime_s == new_min_lifetime


    
