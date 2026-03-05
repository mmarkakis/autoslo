"""
managed_cluster_pool.py
-----------------------
Unified, thread-safe cluster pool with query bookkeeping.

Replaces the combination of :class:`ClusterPool` (blueprints) +
:class:`ClusterStateTracker` (routing) with a single class that
tracks the full cluster lifecycle — from spin-up through draining
and removal — while providing the routing-snapshot and billing APIs
that both the :class:`WorkloadSimulator` and :class:`WorkloadRunner`
need.

The provisioner backend is injected at construction:
* :class:`SimulatedProvisioner` — for the simulator (instant, no I/O).
* :class:`RedshiftServerlessProvisioner` — for live execution (AWS).
"""

from __future__ import annotations

import enum
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from psycopg2.pool import ThreadedConnectionPool

from autoslo.blueprints.cluster import Cluster
from autoslo.blueprints.cluster_conn_info import ClusterConnInfo
from autoslo.capacity.cluster_provisioner import ClusterProvisioner
from autoslo.routing.routing_core import ClusterSnapshot
from autoslo.utils.billing import Billing
from autoslo.workload_definition.query import Query
from autoslo.workload_execution.conn_utils import ConnWithSetup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _ClusterState(enum.Enum):
    PENDING = "pending"
    READY = "ready"
    DRAINING = "draining"
    COLLECTING_STATS = "collecting_stats"
    REMOVED = "removed"


class _ClusterEntry:
    """All mutable per-cluster data, owned exclusively by the pool."""

    __slots__ = (
        "cluster",
        "state",
        "active_queries",
        "completed_queries",
        "neighbors_per_active_query",
        "billing_window_start_s",
        "conn_pool",
        "ready_at_s",
    )

    def __init__(self, cluster: Cluster, state: _ClusterState) -> None:
        self.cluster = cluster
        self.state = state
        self.active_queries: dict[str, Query] = {}
        self.completed_queries: list[Query] = []
        self.neighbors_per_active_query: dict[str, list[Query]] = {}
        self.billing_window_start_s: Optional[float] = None
        self.conn_pool: Optional[ThreadedConnectionPool] = None
        self.ready_at_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Lifecycle log event types
# ---------------------------------------------------------------------------

_EVT_SPIN_UP_REQUESTED = "spin_up_requested"
_EVT_CLUSTER_READY = "cluster_ready"
_EVT_TEAR_DOWN_REQUESTED = "tear_down_requested"
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
    allowed_rpu_sizes :
        RPU sizes the capacity controller is allowed to spin up later.
    maxconns :
        Maximum connections per connection pool (live execution).
    search_path :
        Postgres ``search_path`` for connection pools.
    """

    def __init__(
        self,
        provisioner: ClusterProvisioner,
        initial_rpus: list[int] | None = None,
        allowed_rpu_sizes: list[int] | None = None,
        maxconns: int = 1000,
        search_path: str = "public",
    ) -> None:
        self._provisioner = provisioner
        self._allowed_rpu_sizes: list[int] = sorted(
            allowed_rpu_sizes
            if allowed_rpu_sizes is not None
            else Cluster.ALL_ALLOWED_RPU_SIZES
        )
        self._maxconns = maxconns
        self._search_path = search_path

        self._lock = threading.Lock()
        self._entries: dict[str, _ClusterEntry] = {}

        # Lifecycle log — append-only, never cleared between resets on
        # purpose (each reset writes fresh events for the new sample).
        self._lifecycle_log: list[dict[str, Any]] = []

        # Optional stats-collection callback (live execution only).
        self._stats_collector: Optional[
            Callable[[str, ClusterConnInfo, str], None]
        ] = None
        self._run_id: Optional[str] = None

        # Spin up initial clusters.
        if initial_rpus:
            self._spin_up_initial(initial_rpus)

    # ------------------------------------------------------------------
    # Initial spin-up (parallel for live provisioners)
    # ------------------------------------------------------------------

    def _spin_up_initial(self, rpus: list[int]) -> None:
        """Spin up initial clusters, in parallel when beneficial."""
        if len(rpus) <= 1:
            for rpu in rpus:
                self.request_spin_up(rpu, current_time_s=0.0)
            return

        # Parallel spin-up: useful for live provisioners where each call
        # blocks for 2-5 min.  For SimulatedProvisioner it's harmless.
        with ThreadPoolExecutor(max_workers=len(rpus)) as executor:
            futures = [
                executor.submit(self.request_spin_up, rpu, 0.0)
                for rpu in rpus
            ]
            for f in futures:
                f.result()  # propagate exceptions

    # ------------------------------------------------------------------
    # Cluster lifecycle
    # ------------------------------------------------------------------

    def request_spin_up(
        self, rpu: int, current_time_s: float
    ) -> str:
        """Spin up a new cluster.  Returns its name.

        The cluster starts as PENDING.  If the provisioner returns a
        ``Cluster`` with ``conn_info`` (live mode), it auto-promotes to
        READY and a connection pool is created.  Otherwise the caller
        must invoke :meth:`on_cluster_ready` when appropriate.
        """
        cluster = self._provisioner.spin_up(rpu, current_time_s)
        entry = _ClusterEntry(cluster, _ClusterState.PENDING)

        with self._lock:
            if cluster.name in self._entries:
                raise ValueError(
                    f"Cluster {cluster.name!r} already in pool."
                )
            self._entries[cluster.name] = entry
            self._lifecycle_log.append({
                "timestamp": current_time_s,
                "event_type": _EVT_SPIN_UP_REQUESTED,
                "cluster_name": cluster.name,
                "rpu": rpu,
            })

        # Auto-promote when the provisioner returned full conn_info
        # (RedshiftServerlessProvisioner blocks until AVAILABLE).
        if cluster.conn_info is not None:
            self.on_cluster_ready(cluster.name, current_time_s)

        return cluster.name

    def on_cluster_ready(
        self, cluster_name: str, ready_time_s: float
    ) -> None:
        """Transition PENDING → READY.

        Creates a connection pool if ``conn_info`` is available.
        """
        with self._lock:
            entry = self._entries[cluster_name]
            if entry.state != _ClusterState.PENDING:
                raise ValueError(
                    f"Cannot mark {cluster_name!r} as ready — "
                    f"state is {entry.state.value}, expected 'pending'."
                )
            entry.state = _ClusterState.READY
            entry.ready_at_s = ready_time_s
            self._lifecycle_log.append({
                "timestamp": ready_time_s,
                "event_type": _EVT_CLUSTER_READY,
                "cluster_name": cluster_name,
                "rpu": entry.cluster.rpu,
            })
            # Build connection pool if possible.
            if entry.cluster.conn_info is not None and entry.conn_pool is None:
                entry.conn_pool = self._make_conn_pool(entry.cluster.conn_info)

    def request_tear_down(
        self,
        cluster_name: str,
        current_time_s: float,
        force: bool = False,
    ) -> None:
        """Transition READY → DRAINING (or straight to REMOVED).

        Raises ``ValueError`` if this is the last READY cluster and
        ``force`` is False.
        """
        with self._lock:
            entry = self._entries[cluster_name]
            if entry.state not in (_ClusterState.READY, _ClusterState.DRAINING):
                raise ValueError(
                    f"Cannot tear down {cluster_name!r} — "
                    f"state is {entry.state.value}."
                )
            if entry.state == _ClusterState.DRAINING:
                return  # already draining, idempotent

            # Guard: refuse to tear down the last READY cluster.
            if not force:
                ready_count = sum(
                    1 for e in self._entries.values()
                    if e.state == _ClusterState.READY
                )
                if ready_count <= 1:
                    raise ValueError(
                        f"Cannot tear down {cluster_name!r} — it is the "
                        f"last READY cluster. Use force=True to override."
                    )

            entry.state = _ClusterState.DRAINING
            self._lifecycle_log.append({
                "timestamp": current_time_s,
                "event_type": _EVT_TEAR_DOWN_REQUESTED,
                "cluster_name": cluster_name,
                "rpu": entry.cluster.rpu,
            })
            has_active = bool(entry.active_queries)

        # If no active queries, proceed to removal immediately.
        if not has_active:
            self._finalize_removal(cluster_name, current_time_s)

    def _finalize_removal(
        self, cluster_name: str, current_time_s: float
    ) -> None:
        """Stats collection → REMOVED → provisioner.tear_down.

        Called when a DRAINING cluster has zero active queries.
        Runs outside the lock for the (potentially blocking) stats
        collection and provisioner tear-down.
        """
        with self._lock:
            entry = self._entries[cluster_name]
            if entry.state == _ClusterState.REMOVED:
                return  # already removed
            entry.state = _ClusterState.COLLECTING_STATS

        # Stats collection (may sleep ~120 s in live mode).
        if (
            self._stats_collector is not None
            and entry.cluster.conn_info is not None
            and self._run_id is not None
        ):
            try:
                self._stats_collector(
                    cluster_name, entry.cluster.conn_info, self._run_id
                )
            except Exception:
                logger.exception(
                    "Stats collection failed for cluster %s", cluster_name
                )

        with self._lock:
            self._lifecycle_log.append({
                "timestamp": current_time_s,
                "event_type": _EVT_STATS_COLLECTED,
                "cluster_name": cluster_name,
                "rpu": entry.cluster.rpu,
            })

        # Destroy connection pool.
        if entry.conn_pool is not None:
            try:
                entry.conn_pool.closeall()
            except Exception:
                logger.exception(
                    "Failed to close connection pool for %s", cluster_name
                )
            entry.conn_pool = None

        # Provisioner tear-down.
        try:
            self._provisioner.tear_down(cluster_name, current_time_s)
        except Exception:
            logger.exception(
                "Provisioner tear-down failed for %s", cluster_name
            )

        with self._lock:
            entry.state = _ClusterState.REMOVED
            self._lifecycle_log.append({
                "timestamp": current_time_s,
                "event_type": _EVT_CLUSTER_REMOVED,
                "cluster_name": cluster_name,
                "rpu": entry.cluster.rpu,
            })

    # ------------------------------------------------------------------
    # Query lifecycle
    # ------------------------------------------------------------------

    def on_query_start(self, query: Query) -> None:
        """Register *query* as actively running on ``query.cluster_name``."""
        with self._lock:
            cn = query.cluster_name
            entry = self._entries[cn]
            qid = query.query_id

            # Build neighbour lists.
            current_actives = list(entry.active_queries.values())
            entry.neighbors_per_active_query[qid] = list(current_actives)
            for active_q in current_actives:
                entry.neighbors_per_active_query[active_q.query_id].append(
                    query
                )

            entry.active_queries[qid] = query

            # Start billing window on first query.
            if entry.billing_window_start_s is None:
                entry.billing_window_start_s = query.rel_start_time_s

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
            entry = self._entries[cluster_name]
            if query_id not in entry.active_queries:
                raise KeyError(
                    f"Query {query_id!r} not in active queries for "
                    f"cluster {cluster_name!r}."
                )
            query = entry.active_queries.pop(query_id)
            entry.completed_queries.append(query)
            entry.neighbors_per_active_query.pop(query_id, None)

            # Close billing window if empty and threshold met.
            billing_start = entry.billing_window_start_s
            if (
                not entry.active_queries
                and billing_start is not None
                and (
                    billing_start + Billing.REDSHIFT_BILLING_THRESHOLD_S
                    <= current_time_s
                )
            ):
                entry.billing_window_start_s = None

            # Auto-removal for draining clusters.
            if (
                entry.state == _ClusterState.DRAINING
                and not entry.active_queries
            ):
                should_finalize = True

        if should_finalize:
            self._finalize_removal(cluster_name, current_time_s)

    # ------------------------------------------------------------------
    # Routing support
    # ------------------------------------------------------------------

    @property
    def ready_cluster_names(self) -> list[str]:
        """Names of READY clusters (safe to iterate outside lock)."""
        with self._lock:
            return [
                cn for cn, e in self._entries.items()
                if e.state == _ClusterState.READY
            ]

    @property
    def draining_cluster_names(self) -> set[str]:
        """Names of DRAINING clusters."""
        with self._lock:
            return {
                cn for cn, e in self._entries.items()
                if e.state == _ClusterState.DRAINING
            }

    @property
    def all_cluster_names_ever(self) -> list[str]:
        """Names of all clusters that were ever registered."""
        with self._lock:
            return list(self._entries.keys())

    # Backward-compat alias used by RoutingPolicy (same as CST).
    @property
    def cluster_names(self) -> list[str]:
        """Alias for :attr:`ready_cluster_names`.

        Matches the ``ClusterStateTracker.cluster_names`` API so that
        routing policies work unchanged.
        """
        return self.ready_cluster_names

    def build_routing_context(
        self, incoming: Query,
    ) -> tuple[
        dict[str, ClusterSnapshot],
        dict[str, dict[Query, list[Query]]],
    ]:
        """Atomic snapshot for routing (READY clusters only).

        Identical semantics to ``ClusterStateTracker.build_routing_context``.
        """
        with self._lock:
            snapshots: dict[str, ClusterSnapshot] = {}
            neighbor_map: dict[str, dict[Query, list[Query]]] = {}

            for cn, entry in self._entries.items():
                if entry.state != _ClusterState.READY:
                    continue

                active_list = list(entry.active_queries.values())
                snapshots[cn] = ClusterSnapshot(
                    cluster_name=cn,
                    cost_per_second=entry.cluster.cost_per_second,
                    active_queries=active_list,
                    billing_window_start_s=entry.billing_window_start_s,
                )
                neighbor_map[cn] = {
                    q: entry.neighbors_per_active_query[q.query_id] + [incoming]
                    for q in active_list
                }
                neighbor_map[cn][incoming] = active_list + [incoming]

            return snapshots, neighbor_map

    def build_snapshots(self) -> dict[str, ClusterSnapshot]:
        """Return snapshots without a neighbour map (for introspection)."""
        with self._lock:
            result: dict[str, ClusterSnapshot] = {}
            for cn, entry in self._entries.items():
                if entry.state in (_ClusterState.READY, _ClusterState.DRAINING):
                    result[cn] = ClusterSnapshot(
                        cluster_name=cn,
                        cost_per_second=entry.cluster.cost_per_second,
                        active_queries=list(entry.active_queries.values()),
                        billing_window_start_s=entry.billing_window_start_s,
                    )
            return result

    # ------------------------------------------------------------------
    # Query introspection
    # ------------------------------------------------------------------

    def get_active_queries(self, cluster_name: str) -> list[Query]:
        """Active queries on a specific non-REMOVED cluster."""
        with self._lock:
            entry = self._entries[cluster_name]
            return list(entry.active_queries.values())

    def get_all_active_queries(self) -> dict[str, list[Query]]:
        """Active queries across all non-REMOVED clusters."""
        with self._lock:
            return {
                cn: list(e.active_queries.values())
                for cn, e in self._entries.items()
                if e.state not in (_ClusterState.REMOVED, _ClusterState.PENDING)
            }

    def get_rpu(self, cluster_name: str) -> int:
        """Return the RPU for *cluster_name*."""
        with self._lock:
            return self._entries[cluster_name].cluster.rpu

    def get_cost_per_second(self, cluster_name: str) -> float:
        """Return the cost-per-second for *cluster_name*."""
        with self._lock:
            return self._entries[cluster_name].cluster.cost_per_second

    # ------------------------------------------------------------------
    # Billing / completed queries
    # ------------------------------------------------------------------

    def get_completed_queries(self, cluster_name: str) -> list[Query]:
        """Return a copy of completed queries for *cluster_name*."""
        with self._lock:
            return list(self._entries[cluster_name].completed_queries)

    def get_all_completed_queries(self) -> dict[str, list[Query]]:
        """Return completed queries across all clusters."""
        with self._lock:
            return {
                cn: list(e.completed_queries)
                for cn, e in self._entries.items()
                if e.completed_queries
            }

    @property
    def cost_per_second_map(self) -> dict[str, float]:
        """``{cluster_name: cost_per_second}`` for ALL clusters
        (including REMOVED), for billing analysis."""
        with self._lock:
            return {
                cn: e.cluster.cost_per_second
                for cn, e in self._entries.items()
            }

    @property
    def allowed_rpu_sizes(self) -> list[int]:
        """RPU sizes available for dynamic spin-up (sorted ascending)."""
        return list(self._allowed_rpu_sizes)

    # ------------------------------------------------------------------
    # Connection pools (live execution)
    # ------------------------------------------------------------------

    def conn_pool(self, cluster_name: str) -> ThreadedConnectionPool:
        """Get the connection pool for a READY cluster.

        Raises ``ValueError`` if the cluster has no connection pool.
        """
        with self._lock:
            entry = self._entries[cluster_name]
            if entry.conn_pool is None:
                raise ValueError(
                    f"No connection pool for cluster {cluster_name!r}."
                )
            return entry.conn_pool

    def conn_pool_map(self) -> dict[str, ThreadedConnectionPool]:
        """Return ``{cluster_name: conn_pool}`` for all READY clusters
        that have connection pools."""
        with self._lock:
            return {
                cn: e.conn_pool
                for cn, e in self._entries.items()
                if e.state == _ClusterState.READY and e.conn_pool is not None
            }

    def destroy_all_conn_pools(self) -> None:
        """Close all connection pools."""
        with self._lock:
            for entry in self._entries.values():
                if entry.conn_pool is not None:
                    try:
                        entry.conn_pool.closeall()
                    except Exception:
                        pass
                    entry.conn_pool = None

    def _make_conn_pool(
        self, conn_info: ClusterConnInfo
    ) -> ThreadedConnectionPool:
        """Create a ThreadedConnectionPool from connection info."""
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

    # ------------------------------------------------------------------
    # Lifecycle log
    # ------------------------------------------------------------------

    def get_lifecycle_log(self) -> list[dict[str, Any]]:
        """Chronological list of cluster lifecycle events.

        Each entry has keys: ``timestamp``, ``event_type``,
        ``cluster_name``, ``rpu``.
        """
        with self._lock:
            return list(self._lifecycle_log)

    # ------------------------------------------------------------------
    # Stats collection
    # ------------------------------------------------------------------

    def set_stats_collector(
        self,
        collector: Callable[[str, ClusterConnInfo, str], None] | None,
        run_id: str | None = None,
    ) -> None:
        """Register a callback for stats collection on tear-down.

        Parameters
        ----------
        collector :
            ``(cluster_name, conn_info, run_id) → None``.  Called when
            a cluster finishes draining before the provisioner destroys
            it.  May block (e.g. sleep 120 s for sys-table flush).
            Pass ``None`` to disable (simulation mode).
        run_id :
            Run identifier passed through to the collector.
        """
        self._stats_collector = collector
        self._run_id = run_id

    # ------------------------------------------------------------------
    # Reset (simulator multi-sample runs)
    # ------------------------------------------------------------------

    def reset(self, current_time_s: float = 0.0) -> None:
        """Tear down everything, clear state, re-spin-up initial rpus.

        Used between simulator samples so that each sample starts from
        a clean pool of fresh clusters.
        """
        # Gather initial RPUs before clearing.
        # We infer them from the READY clusters' RPUs at the time of
        # the first reset — alternatively callers pass them explicitly.
        with self._lock:
            initial_rpus = [
                e.cluster.rpu for e in self._entries.values()
                if e.state in (_ClusterState.READY, _ClusterState.PENDING)
            ]

        # Destroy everything (conn pools, provisioner tear-downs).
        self.destroy_all_conn_pools()
        with self._lock:
            self._entries.clear()
            self._lifecycle_log.clear()

        # Re-spin-up.
        if initial_rpus:
            self._spin_up_initial(initial_rpus)

    def reset_with_rpus(
        self,
        rpus: list[int],
        current_time_s: float = 0.0,
    ) -> None:
        """Like :meth:`reset` but with an explicit RPU list."""
        self.destroy_all_conn_pools()
        with self._lock:
            self._entries.clear()
            self._lifecycle_log.clear()
        if rpus:
            self._spin_up_initial(rpus)
