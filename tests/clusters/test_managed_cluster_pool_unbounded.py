from __future__ import annotations

from autoslo.clusters.actions import SpinUpAction
from autoslo.clusters.cluster_provisioner import SimulatedProvisioner
from autoslo.clusters.managed_cluster_pool import ManagedClusterPool
from autoslo.config.component_configs import (
    ManagedClusterPoolConfig,
    ProvisionerConfig,
    TuningConstraintsConfig,
)


def _make_simulated_provisioner() -> SimulatedProvisioner:
    return SimulatedProvisioner(
        ProvisionerConfig(
            aws_config_path="unused",
            cluster_cache_state_dim=1,
            run_id="test",
        )
    )


class TestManagedClusterPoolConfigValidation:
    def test_accepts_unbounded_max_clusters(self):
        cfg = ManagedClusterPoolConfig(initial_rpus=[4], max_clusters=None)
        assert cfg.max_clusters is None

    def test_rejects_negative_max_clusters(self):
        try:
            ManagedClusterPoolConfig(initial_rpus=[4], max_clusters=-1)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "max_clusters" in str(exc)

    def test_rejects_too_low_max_clusters_for_initial_and_reserved(self):
        try:
            ManagedClusterPoolConfig(
                initial_rpus=[4, 8],
                num_reserved_clusters=1,
                max_clusters=2,
            )
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "too low" in str(exc)


class TestTuningConstraintsConfigValidation:
    def test_accepts_none(self):
        cfg = TuningConstraintsConfig(simulation_max_clusters=None)
        assert cfg.simulation_max_clusters is None

    def test_rejects_negative(self):
        try:
            TuningConstraintsConfig(simulation_max_clusters=-1)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "simulation_max_clusters" in str(exc)


class TestManagedClusterPoolUnboundedMode:
    def test_unbounded_pool_allows_additional_spinups(self):
        pool = ManagedClusterPool(
            provisioner=_make_simulated_provisioner(),
            config=ManagedClusterPoolConfig(
                initial_rpus=[4],
                max_clusters=None,
            ),
        )
        pool.add_details_and_spin_up_initial_clusters()

        # With no cap, these additional spin-ups should all be admitted.
        assert pool.request_spin_up(SpinUpAction(reason="test", rpu=8), 1.0)
        assert pool.request_spin_up(SpinUpAction(reason="test", rpu=16), 2.0)

    def test_bounded_pool_blocks_when_exhausted(self):
        pool = ManagedClusterPool(
            provisioner=_make_simulated_provisioner(),
            config=ManagedClusterPoolConfig(
                initial_rpus=[4],
                max_clusters=1,
            ),
        )
        pool.add_details_and_spin_up_initial_clusters()

        # Already used 1/1 through the initial cluster.
        denied = pool.request_spin_up(SpinUpAction(reason="test", rpu=8), 1.0)
        assert denied is None
