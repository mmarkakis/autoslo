import math

import numpy as np
import pytest

from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.clusters.cluster_conn_info import ClusterConnInfo
from autoslo.workload_definition.query import Query


class TestClusterState:
    def test_cluster_state_values(self):
        """
        ClusterState enum has expected values.
        """
        assert ClusterState.PENDING.value == "pending"
        assert ClusterState.READY.value == "ready"
        assert ClusterState.DRAINING.value == "draining"
        assert ClusterState.COLLECTING_STATS.value == "collecting_stats"
        assert ClusterState.REMOVED.value == "removed"

    def test_is_valid_transition(self):
        """
        ClusterState.is_valid_transition correctly identifies valid and invalid transitions.
        """
        vld = ClusterState.is_valid_transition
        p = ClusterState.PENDING
        r = ClusterState.READY
        d = ClusterState.DRAINING
        st = ClusterState.COLLECTING_STATS
        rm = ClusterState.REMOVED

        # Self-transitions are okay
        assert (
            vld(p, p)
            and vld(r, r)
            and vld(d, d)
            and vld(st, st)
            and vld(rm, rm)
        )

        # Valid transitions
        assert vld(p, r)  # pending -> ready
        assert vld(r, d)  # ready -> draining
        assert vld(r, st)  # ready -> collecting_stats
        assert vld(r, rm)  # ready -> removed
        assert vld(d, st)  # draining -> collecting_stats
        assert vld(d, rm)  # draining -> removed
        assert vld(st, rm)  # collecting_stats -> removed

        # Invalid transitions
        assert not vld(p, d)
        assert not vld(p, st)
        assert not vld(p, rm)
        assert not vld(r, p)
        assert not vld(d, p)
        assert not vld(d, r)
        assert not vld(st, p)
        assert not vld(st, r)
        assert not vld(st, d)
        assert not vld(rm, p)
        assert not vld(rm, r)
        assert not vld(rm, d)
        assert not vld(rm, st)


class TestCluster:

    def get_conn_info(self):
        return ClusterConnInfo(
            host="localhost",
            port=5432,
            dbname="testdb",
            user="testuser",
            password="testpass",
        )

    def test_cluster_creation_invalid_name_throws(self):
        """
        Cluster creation with an invalid name should raise a ValueError.
        """
        with pytest.raises(ValueError):
            Cluster(
                name="invalid_name",
                creation_time_s=0.0,
                rpu=10,
                conn_info=self.get_conn_info(),
            )
            rpu = (10,)
            conn_info = (self.get_conn_info(),)

    def test_clone_creates_deep_copy(self):
        """
        Cluster.clone creates a deep copy of the cluster with the same attributes.
        """
        original = Cluster(
            name="autoslo-10-1234567890",
            creation_time_s=0.0,
            rpu=10,
            conn_info=self.get_conn_info(),
            state=ClusterState.READY,
            billing_window_start_s=1234567800.0,
        )
        q1 = Query(
            query_id="q1",
            query_text_id="qt1",
            rel_start_time_s=0.0,
        )
        q2 = Query(
            query_id="q2",
            query_text_id="qt2",
            rel_start_time_s=10.0,
        )
        q3 = Query(
            query_id="q3",
            query_text_id="qt3",
            rel_start_time_s=20.0,
        )
        original.queries = {q2.query_id: q2, q3.query_id: q3}
        original.id_to_neighbors = {
            q2.query_id: [q1, q3],
            q3.query_id: [q2],
        }
        clone = original.clone()

        assert clone is not original
        assert clone.name == original.name
        assert clone.rpu == original.rpu
        assert clone.conn_info == original.conn_info
        assert clone.state == original.state
        assert clone.billing_window_start_s == original.billing_window_start_s
        assert clone.queries == original.queries
        assert clone.id_to_neighbors == original.id_to_neighbors
      
        # Now mutate original
        q4 = Query(
            query_id="q4",
            query_text_id="qt4",
            rel_start_time_s=50.0,
        )
        original.queries = {q4.query_id: q4}
        original.id_to_neighbors = {q4.query_id: [q4]}

        assert clone.queries == {q2.query_id: q2, q3.query_id: q3}
        assert clone.id_to_neighbors == {
            q2.query_id: [q1, q3],
            q3.query_id: [q2],
        }
   
    def test_update_state_valid_transitions(self):
        """
        Cluster.update_state allows valid state transitions.
        """
        cluster = Cluster(
            name="autoslo-10-1234567890",
            creation_time_s=0.0,
            rpu=10,
            conn_info=self.get_conn_info(),
        )
        cluster.update_state(ClusterState.READY)
        assert cluster.state == ClusterState.READY
        cluster.update_state(ClusterState.DRAINING)
        assert cluster.state == ClusterState.DRAINING
        cluster.update_state(ClusterState.COLLECTING_STATS)
        assert cluster.state == ClusterState.COLLECTING_STATS
        cluster.update_state(ClusterState.REMOVED)
        assert cluster.state == ClusterState.REMOVED

    def test_update_state_invalid_transitions(self):
        """
        Cluster.update_state raises ValueError on invalid state transitions.
        """
        cluster = Cluster(
            name="autoslo-10-1234567890",
            creation_time_s=0.0,
            rpu=10,
            conn_info=self.get_conn_info(),
        )
        with pytest.raises(ValueError):
            cluster.update_state(ClusterState.DRAINING)
        with pytest.raises(ValueError):
            cluster.update_state(ClusterState.COLLECTING_STATS)
        with pytest.raises(ValueError):
            cluster.update_state(ClusterState.REMOVED)

    def test_add_query_new_billing_window(self):
        """
        Adding a query that starts after the current billing window should
        update the billing window start time.
        """
        cluster = Cluster(
            name="autoslo-10-1234567890",
            creation_time_s=0.0,
            rpu=10,
            conn_info=self.get_conn_info(),
        )
        q1 = Query(
            query_id="q1",
            query_text_id="qt1",
            rel_start_time_s=150.0,
        )
        cluster.add_query(q1, )
        assert cluster.queries.keys() == {q1.query_id}
        assert cluster.queries[q1.query_id] == q1
        assert cluster.id_to_neighbors.keys() == {q1.query_id}
        assert cluster.id_to_neighbors[q1.query_id] == []
        assert cluster.billing_window_start_s == 150.0

    def test_add_query_same_billing_window(self):
        """
        Adding a query that starts within the current billing window should
        not update the billing window start time.
        """
        cluster = Cluster(
            name="autoslo-10-1234567890",
            creation_time_s=0.0,
            rpu=10,
            conn_info=self.get_conn_info(),
        )
        q1 = Query(
            query_id="q1",
            query_text_id="qt1",
            rel_start_time_s=150.0,
        )
        cluster.add_query(q1)
        q2 = Query(
            query_id="q2",
            query_text_id="qt2",
            rel_start_time_s=160.0,
        )
        cluster.add_query(q2)
        assert cluster.queries.keys() == {q1.query_id, q2.query_id}
        assert cluster.queries[q1.query_id] == q1
        assert cluster.queries[q2.query_id] == q2
        assert cluster.id_to_neighbors.keys() == {q1.query_id, q2.query_id}
        assert set(cluster.id_to_neighbors[q1.query_id]) == {q2}
        assert set(cluster.id_to_neighbors[q2.query_id]) == {q1}
        assert cluster.billing_window_start_s == 150.0

    def test_finish_query_without_clearing_billing_window(self):
        """
        Finishing a query that ends within the current billing window should
        not clear the billing window start time.
        """
        cluster = Cluster(
            name="autoslo-10-1234567890",
            creation_time_s=0.0,
            rpu=10,
            conn_info=self.get_conn_info(),
        )
        q1 = Query(
            query_id="q1",
            query_text_id="qt1",
            rel_start_time_s=150.0,
        )
        cluster.add_query(q1)
        cluster.finish_query(
            q1.query_id, rel_time_s=160.0, min_billing_window_size_s=60.0
        )
        assert cluster.queries == {}
        assert cluster.id_to_neighbors == {}
        assert cluster.billing_window_start_s == 150.0

    def test_finish_query_clearing_billing_window(self):
        """
        Finishing a query that ends after the current billing window should
        clear the billing window start time.
        """
        cluster = Cluster(
            name="autoslo-10-1234567890",
            creation_time_s=0.0,
            rpu=10,
            conn_info=self.get_conn_info(),
        )
        q1 = Query(
            query_id="q1",
            query_text_id="qt1",
            rel_start_time_s=150.0,
        )
        cluster.add_query(q1)
        cluster.finish_query(
            q1.query_id, rel_time_s=220.0, min_billing_window_size_s=60.0
        )
        assert cluster.queries == {}
        assert cluster.id_to_neighbors == {}
        assert cluster.billing_window_start_s is None

    def test_rpu_for_cluster_name_throws_on_invalid_name(self):
        """
        Cluster.rpu_for_cluster_name raises ValueError on invalid cluster name.
        """
        with pytest.raises(ValueError):
            Cluster.rpu_for_cluster_name("invalid_name")

    @pytest.mark.parametrize("seed", range(10))
    def test_cost_per_second(self, seed):
        """
        Cluster.cost_per_second returns expected values based on RPU.
        """

        np.random.seed(seed)
        rpu = np.random.randint(1, 100)
        cost_per_rpu_hour = np.random.uniform(0.05, 0.20)
        cluster = Cluster(
            name=f"autoslo-{rpu}-1234567890",
            creation_time_s=0.0,
            rpu=rpu,
            conn_info=self.get_conn_info(),
            cost_per_rpu_hour=cost_per_rpu_hour,
        )
        assert math.isclose(
            cluster.cost_per_second,
            rpu * cost_per_rpu_hour / 3600.0,
        )
