"""
managed_cluster_pool.py
-----------------------
Unified, thread-safe cluster pool with query bookkeeping.

Replaces the combination of :class:`ClusterPool` (blueprints) +
:class:`ClusterStateTracker` (routing) with a single class that
tracks the full cluster lifecycle — from spin-up through draining
and removal — while providing the routing-snapshot and billing APIs
that both the :class:`WorkloadSimulator` and :class:`WorkloadRunner`
need. The provisioner backend is injected at construction.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

import autoslo.utils.config as cfgu
from autoslo.clusters.actions import SpinUpAction, TearDownAction
from autoslo.clusters.cluster import Cluster, ClusterState
from autoslo.clusters.cluster_conn_info import ClusterConnInfo
from autoslo.clusters.cluster_provisioner import ClusterProvisioner
from autoslo.clusters.redshift_run_stats_collector import (
    RedshiftRunStatsCollector,
)
from autoslo.utils.billing import Billing
from autoslo.utils.logging import LOGGER_NAME, emit_structured
from autoslo.workload_definition.query import Query
from autoslo.workload_execution.conn_utils import ConnWithSetup

logger = logging.getLogger(__name__)
_has_structured = lambda: bool(logging.getLogger(LOGGER_NAME).handlers)


# ---------------------------------------------------------------------------
# Lifecycle log event types
# ---------------------------------------------------------------------------

_EVT_SPIN_UP_REQUESTED = "spin_up_requested"
_EVT_CLUSTER_READY = "cluster_ready"
_EVT_TEAR_DOWN_REQUESTED = "tear_down_requested"
_EVT_TEAR_DOWN_BLOCKED = "tear_down_blocked"
_EVT_STATS_COLLECTED = "stats_collected"
_EVT_CLUSTER_REMOVED = "cluster_removed"

class ManagedClusterPool:
    """Unified, thread-safe cluster pool with query bookkeeping.

    Parameters
    ----------
    provisioner :
        Backend that creates/destroys actual (or simulated) clusters.
    initial_rpus :
        RPU sizes for the clusters that should be available at
        construction time.  Each one triggers a ``provisioner.spin_up``
        call; for live provisioners the calls are parallelised.
    search_path :
        Postgres ``search_path`` for connection pools.
    """

    def __init__(
        self,
        provisioner: ClusterProvisioner,
        initial_rpus: list[int],
        maxconns: int = 1000,
        search_path: str = "public",
        collect_cluster_stats: bool = False,
        run_id: Optional[str] = None,
        write_text_log: bool = False,
    ) -> None:
        self._provisioner = provisioner
        self._initial_rpus = initial_rpus
        self._maxconns = maxconns
        self._search_path = search_path
        self._collect_cluster_stats = collect_cluster_stats

        self._lock = threading.Lock()
        self._clusters: dict[str, Cluster] = {}
        self._completed_queries: dict[str, list[tuple[float, Query]]] = {}
        self._conn_pools: dict[str, ThreadedConnectionPool] = {}

        self._run_id: Optional[str] = run_id
        self._write_text_log = write_text_log

        # Spin up initial clusters.
        rpus = self._initial_rpus
        with ThreadPoolExecutor(max_workers=len(rpus)) as executor:
            futures = [
                executor.submit(
                    self.request_spin_up,
                    SpinUpAction(rpu=rpu, reason="initial"),
                    0.0,
                )
                for rpu in rpus
            ]
            for f in futures:
                f.result()  # propagate exceptions

    @property
    def provisioner(self) -> ClusterProvisioner:
        return self._provisioner

    # ------------------------------------------------------------------
    # Cluster lifecycle
    # ------------------------------------------------------------------

    def request_spin_up(
        self, action: SpinUpAction, current_time_s: float
    ) -> str:
        """Spin up a new cluster.  Returns its name.

        The cluster starts as PENDING.  If the provisioner returns a
        ``Cluster`` with ``conn_info`` (live mode), it auto-promotes to
        READY and a connection pool is created.  Otherwise the caller
        must invoke :meth:`on_cluster_ready` when appropriate.
        """
        cluster = self._provisioner.spin_up(action.rpu, current_time_s)
        with self._lock:
            if cluster.name in self._clusters:
                raise ValueError(f"Cluster {cluster.name!r} already in pool.")
            self._clusters[cluster.name] = cluster

        if self._write_text_log:
            logging.debug(
                "Requested spin-up with %d RPU (reason: %s) at time %.2f",
                action.rpu,
                action.reason,
                current_time_s,
            )
        emit_structured(
            {
                "timestamp": current_time_s,
                "event_type": "request_spin_up",
                "cluster_name": cluster.name,
                "rpu": action.rpu,
                "reason": action.reason,
                "source": "ManagedClusterPool",
            }
        )

        if cluster.conn_info is not None:
            self.on_cluster_ready(cluster.name, current_time_s)

        return cluster.name

    def on_cluster_ready(self, cluster_name: str, ready_time_s: float) -> None:
        """Transition PENDING → READY.

        Creates a connection pool if ``conn_info`` is available.
        """
        with self._lock:
            cluster = self._clusters[cluster_name]
            if cluster.state != ClusterState.PENDING:
                raise ValueError(
                    f"Cannot mark {cluster_name!r} as ready — "
                    f"state is {cluster.state.value}, expected 'pending'."
                )

        # Build connection pool if possible.
        if (cluster.conn_info is not None) and (
            cluster.name not in self._conn_pools
        ):
            self._conn_pools[cluster_name] = self._make_conn_pool(
                cluster.conn_info
            )

        with self._lock:
            cluster.update_state(ClusterState.READY)

        if self._write_text_log:
            logging.debug(
                "Cluster %s is ready at time %.2f", cluster_name, ready_time_s
            )
        emit_structured(
            {
                "timestamp": ready_time_s,
                "event_type": _EVT_CLUSTER_READY,
                "cluster_name": cluster_name,
                "rpu": cluster.rpu,
                "source": "ManagedClusterPool",
            }
        )

    def request_tear_down(
        self,
        action: TearDownAction,
        current_time_s: float,
        force: bool = False,
    ) -> None:
        """Transition READY → DRAINING (or straight to REMOVED).

        If *force* is False, the method refuses to tear down the last READY
        cluster.
        """
        with self._lock:
            cluster = self._clusters[action.cluster_name]
            if cluster.state not in (ClusterState.READY, ClusterState.DRAINING):
                raise ValueError(
                    f"Cannot tear down {action.cluster_name!r} — "
                    f"state is {cluster.state.value}."
                )
            if cluster.state == ClusterState.DRAINING:
                return

            # Guard: refuse to tear down the last READY cluster.
            if not force:
                ready_count = sum(
                    1
                    for c in self._clusters.values()
                    if c.state == ClusterState.READY
                )
                if ready_count <= 1:
                    if self._write_text_log:
                        logging.debug(
                            "Skipping tear-down of %s — it is the last routable "
                            "cluster.",
                            action.cluster_name,
                        )
                    if _has_structured():
                        emit_structured(
                            {
                                "timestamp": current_time_s,
                                "source": "ManagedClusterPool",
                                "event_type": _EVT_TEAR_DOWN_BLOCKED,
                                "cluster_name": action.cluster_name,
                                "rpu": cluster.rpu,
                                "reason": "last_routable_cluster",
                            }
                        )
                return

            cluster.update_state(ClusterState.DRAINING)
            active_queries = cluster.active_queries
            if self._write_text_log:
                logging.debug(
                    "Cluster %s marked as draining.",
                    action.cluster_name,
                    len(active_queries),
                )
            if _has_structured():
                emit_structured(
                    {
                        "timestamp": current_time_s,
                        "source": "ManagedClusterPool",
                        "event_type": _EVT_TEAR_DOWN_REQUESTED,
                        "cluster_name": action.cluster_name,
                        "rpu": cluster.rpu,
                    }
                )

        # If no active queries, proceed to removal immediately.
        if not active_queries:
            self._finalize_removal(action.cluster_name, current_time_s)

    def _finalize_removal(
        self, cluster_name: str, current_time_s: float
    ) -> None:
        """Stats collection → REMOVED → provisioner.tear_down.

        Called when a DRAINING cluster has zero active queries.
        Runs outside the lock for the (potentially blocking) stats
        collection and provisioner tear-down.
        """
        with self._lock:
            cluster = self._clusters[cluster_name]
            if cluster.state == ClusterState.REMOVED:
                return  # already removed
            cluster.update_state(ClusterState.COLLECTING_STATS)

        # Stats collection (may sleep ~120 s in live mode).
        if (
            self._collect_cluster_stats
            and cluster.conn_info is not None
            and self._run_id is not None
        ):
            try:
                RedshiftRunStatsCollector.collect_cluster_stats(
                    cluster_name, cluster.conn_info, self._run_id
                )
            except Exception:
                logger.exception(
                    "Stats collection failed for cluster %s", cluster_name
                )
            current_time_s = datetime.now(tz=timezone.utc).timestamp()

        if _has_structured():
            emit_structured(
                {
                    "timestamp": current_time_s, 
                    "source": "ManagedClusterPool",
                    "event_type": _EVT_STATS_COLLECTED,
                    "cluster_name": cluster_name,
                    "rpu": cluster.rpu,
                }
            )

        # Destroy connection pool.
        if self._conn_pools.get(cluster_name) is not None:
            try:
                self._conn_pools[cluster_name].closeall()
            except Exception:
                logger.exception(
                    "Failed to close connection pool for cluster %s",
                    cluster_name,
                )
            del self._conn_pools[cluster_name]

        # Provisioner tear-down.
        try:
            self._provisioner.tear_down(cluster_name, current_time_s)
        except Exception:
            logger.exception(
                "Provisioner tear-down failed for %s", cluster_name
            )

        if self._write_text_log:
            logging.debug(
                "Cluster %s is being removed",
                cluster_name,
            )

        if _has_structured():
            emit_structured(
                {
                    "timestamp": current_time_s,
                    "source": "ManagedClusterPool",
                    "event_type": _EVT_CLUSTER_REMOVED,
                    "cluster_name": cluster_name,
                    "rpu": cluster.rpu,
                }
            )

        with self._lock:
            cluster.update_state(ClusterState.REMOVED)
            del self._clusters[cluster_name]

    # ------------------------------------------------------------------
    # Query lifecycle
    # ------------------------------------------------------------------

    def on_query_start(
        self,
        cluster_name: str,
        query: Query,
    ) -> None:
        """Register *query* as actively running on *cluster_name*."""
        with self._lock:
            self._clusters[cluster_name].add_query(query)

    def on_query_finish(
        self,
        query_id: str,
        cluster_name: str,
        current_time_s: float,
    ) -> None:
        """Move query from active → completed.

        If the cluster is DRAINING and no active queries remain,
        triggers the removal pipeline.
        """
        query_id = str(query_id)
        should_finalize = False

        with self._lock:
            cluster = self._clusters[cluster_name]
            query, latency_s = cluster.finish_query(
                query_id,
                current_time_s=current_time_s,
                min_billing_window_size_s=Billing.REDSHIFT_BILLING_THRESHOLD_S,
            )
            self._completed_queries.setdefault(cluster_name, []).append(
                (latency_s, query)
            )

            # Auto-removal for draining clusters.
            if (
                cluster.state == ClusterState.DRAINING
                and not cluster.active_queries
            ):
                should_finalize = True

        if should_finalize:
            self._finalize_removal(cluster_name, current_time_s)

    # ------------------------------------------------------------------
    # Routing and checkpointing support
    # ------------------------------------------------------------------

    def snapshot(self, only_ready: bool) -> dict[str, Cluster]:
        """Return clones the clusters in the pool"""
        with self._lock:
            cond = (
                (lambda c: c.state == ClusterState.READY)
                if only_ready
                else (lambda c: True)
            )
            return {
                cluster_name: cluster.clone()
                for cluster_name, cluster in self._clusters.items()
                if cond(cluster)
            }

    def get_predicted_latencies(self) -> dict[str, dict[str, float]]:
        """Snapshot the predicted latencies for all clusters.

        Returns a fresh ``{cluster_name: {query_id: latency_s}}`` dict
        that is safe to read and mutate without holding the pool lock.
        """
        with self._lock:
            return {
                cluster_name: dict(cluster.predicted_latencies)
                for cluster_name, cluster in self._clusters.items()
                if cluster.predicted_latencies
            }

    def commit_predicted_latencies(
        self,
        cluster_name: str,
        new_latencies: dict[str, float],
    ) -> None:
        """
        Atomically update predicted latencies on a cluster.
        """
        with self._lock:
            cluster = self._clusters[cluster_name]
            cluster.predicted_latencies = dict(new_latencies)

    def ready_and_pending_counts_per_rpu(self) -> Counter[int]:
        """RPU → count for READY + PENDING clusters.

        Used by :meth:`Autoscaler.reconcile_checkpoints_up_to` to
        compute the gap between desired and current capacity.
        Returns a plain dict usable as a :class:`collections.Counter`.
        """
        with self._lock:
            rpus = [
                cluster.rpu
                for cluster in self._clusters.values()
                if cluster.state in (ClusterState.READY, ClusterState.PENDING)
            ]
        return Counter(rpus)

    # ------------------------------------------------------------------
    # Billing / completed queries
    # ------------------------------------------------------------------

    def clusters_in_state(self, state: ClusterState) -> set[str]:
        """Names of clusters in *state*."""
        with self._lock:
            return {
                cluster_name
                for cluster_name, cluster in self._clusters.items()
                if cluster.state == state
            }

    def get_all_completed_queries(self) -> dict[str, list[tuple[float, Query]]]:
        """Return completed queries across all clusters."""
        with self._lock:
            return dict(self._completed_queries)

    # ------------------------------------------------------------------
    # Connection pools (live execution)
    # ------------------------------------------------------------------

    def conn_pool(self, cluster_name: str) -> ThreadedConnectionPool:
        """Get the connection pool for a READY cluster.

        Raises ``ValueError`` if the cluster has no connection pool.
        """
        with self._lock:
            conn_pool = self._conn_pools.get(cluster_name)
            if conn_pool is None:
                raise ValueError(
                    f"No connection pool for cluster {cluster_name!r}."
                )
            return conn_pool

    _GETCONN_MAX_RETRIES = 3
    _GETCONN_BASE_DELAY_S = 0.1  # 0.1 s, 0.2 s, 0.4 s

    def getconn(self, cluster_name: str):
        """Acquire a connection for *cluster_name* with retry.

        Retries transient ``OperationalError`` failures (e.g. Redshift
        Serverless waking from idle) up to ``_GETCONN_MAX_RETRIES``
        times with exponential back-off.  Checks cluster state before
        each attempt and raises immediately for non-transient errors
        or if the cluster is no longer READY/DRAINING.
        """
        last_exc: Exception | None = None
        for attempt in range(self._GETCONN_MAX_RETRIES):
            # Fail fast if the cluster has been removed.
            with self._lock:
                cluster = self._clusters[cluster_name]
                if cluster.state == ClusterState.REMOVED:
                    raise RuntimeError(
                        f"Cluster {cluster_name!r} has been removed."
                    )
                conn_pool = self._conn_pools.get(cluster_name)
                if conn_pool is None:
                    raise ValueError(
                        f"No connection pool for cluster {cluster_name!r}."
                    )

            try:
                return conn_pool.getconn()
            except psycopg2.OperationalError as exc:
                last_exc = exc
                delay = self._GETCONN_BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "getconn failed for %s (attempt %d/%d, retrying "
                    "in %.2fs): %s",
                    cluster_name,
                    attempt + 1,
                    self._GETCONN_MAX_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    def putconn(self, cluster_name: str, conn) -> None:
        """Return *conn* to the pool for *cluster_name*.

        Silently ignores errors (e.g. pool already closed).
        """
        try:
            with self._lock:
                conn_pool = self._conn_pools.get(cluster_name)
            if conn_pool is not None:
                conn_pool.putconn(conn)
        except Exception:
            pass

    def destroy_all_conn_pools(self) -> None:
        """Close all connection pools."""
        with self._lock:
            for cluster_name in self._conn_pools:
                try:
                    self._conn_pools[cluster_name].closeall()
                except Exception:
                    pass
            self._conn_pools.clear()

    def _make_conn_pool(
        self, conn_info: ClusterConnInfo, num_retries: int = 3
    ) -> ThreadedConnectionPool:
        """Create a ThreadedConnectionPool from connection info."""
        for attempt in range(num_retries):
            try:
                return ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self._maxconns,
                    host=conn_info.host,
                    port=conn_info.port,
                    user=conn_info.user,
                    password=conn_info.password,
                    dbname=conn_info.dbname,
                    connection_factory=lambda *args, **kwargs: ConnWithSetup(
                        *args,
                        search_path=self._search_path,
                        **kwargs,
                    ),
                )
            except Exception as ex:
                logger.warning(
                    "Failed to create connection pool (attempt %d/%d): %s",
                    attempt + 1,
                    num_retries,
                    ex,
                )
        raise RuntimeError(
            f"Failed to create connection pool after {num_retries} attempts."
        )
