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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

from autoslo.clusters.actions import SpinUpAction, TearDownAction
from autoslo.clusters.cluster import Cluster, ClusterState, ClusterView
from autoslo.clusters.cluster_conn_info import ClusterConnInfo
from autoslo.clusters.cluster_provisioner import ClusterProvisioner
from autoslo.clusters.redshift_run_stats_collector import (
    RedshiftRunStatsCollector,
)
from autoslo.clusters.spin_up_budget import SpinUpBudget
from autoslo.config.component_configs import ManagedClusterPoolConfig
from autoslo.filesystem.structured_events import BaseStructuredEvent, EventType
from autoslo.filesystem.structured_log import emit_structured
from autoslo.nn.lstm_state import AfterLSTMState
from autoslo.workload_definition.query import Query
from autoslo.workload_execution.conn_utils import ConnWithSetup

logger = logging.getLogger(__name__)


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
        config: ManagedClusterPoolConfig,
    ) -> None:
        self._provisioner = provisioner
        self._initial_rpus = config.initial_rpus
        self._max_clusters = config.max_clusters
        self._num_reserved_clusters = config.num_reserved_clusters
        self._maxconns = config.maxconns

        self._lock = threading.Lock()
        self._clusters: dict[str, Cluster] = {}
        self._completed_queries: dict[str, list[tuple[float, Query]]] = {}
        self._conn_pools: dict[str, ThreadedConnectionPool] = {}
        self._background_futures: list[Future] = []
        self._bg_futures_lock = threading.Lock()

        self._budget = SpinUpBudget(max_clusters=config.max_clusters)
        self._budget.reserve(config.num_reserved_clusters)

    def add_details_and_spin_up_initial_clusters(
        self,
        search_path: str = "public",
        background_executor: Optional[ThreadPoolExecutor] = None,
        run_id: Optional[str] = None,
        out_dir: Optional[Path] = None,
        write_text_log: bool = False,
    ) -> None:
        self._search_path = search_path
        self._background_executor = background_executor
        self._run_id: Optional[str] = run_id
        self._out_dir: Optional[Path] = out_dir
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
        self, action: SpinUpAction, rel_time_s: float
    ) -> Optional[str]:
        """Spin up a new cluster.  Returns its name, or ``None`` on denial.

        The cluster starts as PENDING.  If the provisioner returns a
        ``Cluster`` with ``conn_info`` (live mode), it auto-promotes to
        READY and a connection pool is created.  Otherwise the caller
        must invoke :meth:`on_cluster_ready` when appropriate.
        """

        with self._lock:
            if action.from_reserved_budget:
                success = self._budget.try_consume_reserved()
            else:
                success = self._budget.try_consume()
            snap = self._budget.snapshot()

        if not success:
            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=rel_time_s,
                    event_type=EventType.SPIN_UP_BLOCKED,
                    source="ManagedClusterPool",
                    cluster_name="",
                    details={
                        "reason": "max_clusters_exhausted",
                        "action_reason": action.reason,
                        "rpu": action.rpu,
                        "max": snap["max"],
                        "used": snap["used"],
                        "reserved": snap["reserved"],
                        "available": snap["available"],
                    },
                )
            )
            if self._write_text_log:
                logging.warning(
                    "Spin-up denied: max_clusters=%s exhausted "
                    "(used=%s, reserved=%s, available=%s). "
                    "action.reason=%s",
                    snap["max"],
                    snap["used"],
                    snap["reserved"],
                    snap["available"],
                    action.reason,
                )
            return None

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.SPIN_UP_REQUESTED,
                source="ManagedClusterPool",
                cluster_name="",
                details={
                    "reason": action.reason,
                },
            )
        )

        if self._write_text_log:
            logging.debug(
                "Requested spin-up with %d RPU (reason: %s) at time %.2f",
                action.rpu,
                action.reason,
                rel_time_s,
            )

        cluster = self._provisioner.spin_up(action.rpu, rel_time_s)
        with self._lock:
            if cluster.name in self._clusters:
                raise ValueError(f"Cluster {cluster.name!r} already in pool.")
            self._clusters[cluster.name] = cluster

        if cluster.conn_info is not None:
            self.on_cluster_ready(cluster.name, rel_time_s)

        return cluster.name

    def release_reserved_spinups(self, n: int) -> None:
        """Release up to ``n`` reserved spin-ups back to the available pool."""
        if n > 0:
            with self._lock:
                self._budget.release_reservation(n)

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
            BaseStructuredEvent(
                rel_time_s=ready_time_s,
                event_type=EventType.CLUSTER_READY,
                source="ManagedClusterPool",
                cluster_name=cluster.name,
            )
        )

    def request_tear_down(
        self,
        action: TearDownAction,
        rel_time_s: float,
        force: bool = False,
    ) -> None:
        """Transition READY → DRAINING (or straight to REMOVED).

        If *force* is False, the method refuses to tear down the last READY
        cluster.
        """
        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.TEAR_DOWN_REQUESTED,
                source="ManagedClusterPool",
                cluster_name=action.cluster_name,
                details={
                    "reason": action.reason,
                    "force": force,
                },
            )
        )

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
                    emit_structured(
                        BaseStructuredEvent(
                            rel_time_s=rel_time_s,
                            event_type=EventType.TEAR_DOWN_BLOCKED,
                            source="ManagedClusterPool",
                            cluster_name=action.cluster_name,
                        )
                    )
                    return

            cluster.update_state(ClusterState.DRAINING)
            active_queries = cluster.active_queries

            # When force=True, any remaining active queries are orphans
            # (the run has ended).  Clear them so finalization — and
            # therefore stats collection — always proceeds.
            if force and active_queries:
                # TODO: this could be handled in a more graceful way.
                logger.warning(
                    "Force tear-down of %s: clearing %d orphaned active "
                    "queries.",
                    action.cluster_name,
                    len(active_queries),
                )
                if cluster.billing_window_start_s is not None:
                    cluster.billing_accumulator.add_interval(
                        cluster.billing_window_start_s, rel_time_s
                    )
                    cluster.billing_window_start_s = None
                cluster.queries.clear()
                cluster.id_to_neighbors.clear()
                cluster.predicted_latencies.clear()
                active_queries = []

            if self._write_text_log:
                logging.debug(
                    "Cluster %s marked as draining (%d active queries).",
                    action.cluster_name,
                    len(active_queries),
                )

        # If no active queries, proceed to removal immediately.
        if not active_queries:
            self._dispatch_finalize(action.cluster_name, rel_time_s)

    def _finalize_removal(self, cluster_name: str, rel_time_s: float) -> None:
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
            cluster.conn_info is not None
            and self._run_id is not None
            and self._out_dir is not None
        ):
            _STATS_RETRY_DELAYS_S = [60, 60, 120]
            for attempt, delay in enumerate(_STATS_RETRY_DELAYS_S):
                try:
                    logger.info(
                        "Waiting %ds before collecting stats for cluster %s "
                        "to allow system tables to flush (attempt %d/%d)",
                        delay,
                        cluster_name,
                        attempt + 1,
                        len(_STATS_RETRY_DELAYS_S),
                    )
                    time.sleep(delay)
                    RedshiftRunStatsCollector.collect_cluster_stats(
                        cluster_name,
                        cluster.conn_info,
                        self._run_id,
                        self._out_dir,
                    )
                    emit_structured(
                        BaseStructuredEvent(
                            rel_time_s=rel_time_s,
                            event_type=EventType.STATS_COLLECTED,
                            source="ManagedClusterPool",
                            cluster_name=cluster_name,
                        )
                    )
                    break
                except Exception:
                    if attempt < len(_STATS_RETRY_DELAYS_S) - 1:
                        logger.warning(
                            "Stats collection failed for cluster %s "
                            "(attempt %d/%d), will retry.",
                            cluster_name,
                            attempt + 1,
                            len(_STATS_RETRY_DELAYS_S),
                        )
                    else:
                        logger.exception(
                            "Stats collection failed for cluster %s after "
                            "%d attempts, giving up.",
                            cluster_name,
                            len(_STATS_RETRY_DELAYS_S),
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
            self._provisioner.tear_down(cluster_name, rel_time_s)
        except Exception:
            logger.exception(
                "Provisioner tear-down failed for %s", cluster_name
            )

        with self._lock:
            cluster.update_state(ClusterState.REMOVED)
            del self._clusters[cluster_name]

        if self._write_text_log:
            logging.debug(
                "Cluster %s was removed",
                cluster_name,
            )

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=rel_time_s,
                event_type=EventType.CLUSTER_REMOVED,
                source="ManagedClusterPool",
                cluster_name=cluster_name,
            )
        )

    def wait_for_background_tasks(
        self, timeout: Optional[float] = None
    ) -> None:
        """Block until all background finalization tasks complete.

        Only relevant when a ``background_executor`` is configured;
        otherwise returns immediately.
        """
        with self._bg_futures_lock:
            pending = list(self._background_futures)
        for fut in pending:
            fut.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Query lifecycle
    # ------------------------------------------------------------------

    def on_query_start(
        self,
        cluster_name: str,
        query: Query,
        new_predicted_latencies_on_selected: dict[str, float],
        new_cluster_cache_state: np.ndarray,
        new_lstm_states: dict[str, AfterLSTMState],
    ) -> None:
        """Register *query* as actively running on *cluster_name*."""
        with self._lock:
            cluster = self._clusters[cluster_name]
            cluster.add_query(
                query,
                new_predicted_latencies_on_selected,
                new_cache_state=new_cluster_cache_state,
                new_lstm_states_on_selected=new_lstm_states,
            )

    def on_query_finish(
        self,
        query_id: str,
        cluster_name: str,
        rel_time_s: float,
    ) -> None:
        """Move query from active → completed.

        If the cluster is DRAINING and no active queries remain,
        triggers the removal pipeline.
        """
        query_id = str(query_id)
        should_finalize = False

        with self._lock:
            cluster = self._clusters.get(cluster_name)
            if cluster is None:
                logger.warning(
                    "on_query_finish for query %s on %s: "
                    "cluster already removed.",
                    query_id,
                    cluster_name,
                )
                return
            if query_id not in cluster.active_query_ids:
                logger.warning(
                    "on_query_finish for query %s on %s: "
                    "query not in active set (already cleared).",
                    query_id,
                    cluster_name,
                )
                return
            query, latency_s = cluster.finish_query(query_id, rel_time_s)
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
            self._dispatch_finalize(cluster_name, rel_time_s)

    def _dispatch_finalize(self, cluster_name: str, rel_time_s: float) -> None:
        """Run ``_finalize_removal`` inline or in the background executor.

        When a ``background_executor`` was provided at construction the
        heavy work (stats collection, AWS API calls) is submitted to
        that executor so the calling thread is not blocked.
        """
        if self._background_executor is not None:
            fut = self._background_executor.submit(
                self._finalize_removal, cluster_name, rel_time_s
            )
            with self._bg_futures_lock:
                self._background_futures.append(fut)
        else:
            self._finalize_removal(cluster_name, rel_time_s)

    # ------------------------------------------------------------------
    # Routing and spin-up support
    # ------------------------------------------------------------------

    def snapshot(self, only_ready: bool) -> dict[str, ClusterView]:
        """
        Return immutable, deep-copied ClusterViews for all clusters in the pool.
        """
        with self._lock:
            cond = (
                (lambda c: c.state == ClusterState.READY)
                if only_ready
                else (lambda c: True)
            )
            return {
                cluster_name: ClusterView.from_cluster(cluster)
                for cluster_name, cluster in self._clusters.items()
                if cond(cluster)
            }

    def get_predicted_latency(
        self, cluster_name: str, query_id: str
    ) -> Optional[float]:
        """Get the predicted latency for *query_id* on *cluster_name*."""
        with self._lock:
            cluster = self._clusters[cluster_name]
            return cluster.predicted_latencies.get(query_id, None)

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
