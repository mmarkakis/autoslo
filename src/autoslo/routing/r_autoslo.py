"""
r_autoslo.py
------------
Online query router that uses :mod:`autoslo.routing.routing_core` for
placement decisions.

Key differences from ``RIconq``:

* Uses marginal *cost* (not just SLO-violation count) as the secondary
  criterion, matching the simulator's policy exactly.
* Supports SLO-violation-amount optimisation and per-template SLOs via
  ``SloResolver``.
* Emits a ``capacity_pressure`` callback when *every* cluster would
  incur an SLO violation, so the capacity controller can react.
* Tracks per-cluster ``recent_tables`` for cache-affinity scoring
  (stub — tables are populated but affinity is not yet weighted).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.models.model_prediction import ModelPrediction
from autoslo.nn.concurrent_query_dataset import ConcurrentQueryDataset
from autoslo.routing.query_router import QueryRouter
from autoslo.routing.routing_core import (
    ClusterSnapshot,
    PlacementScore,
    RoutingCore,
)
from autoslo.workload_definition.query import Query, TPCDSTempAndQIdx

from autoslo.utils.billing import Billing

logger = logging.getLogger(__name__)


class RAutoSLO(QueryRouter):
    """Online router that minimises (marginal SLO violation, marginal cost)
    per query, using the shared :mod:`routing_core` scoring logic.

    Thread-safe: the active-query bookkeeping is protected by a lock so
    that ``route_query``, ``on_query_start``, and ``on_query_finish`` can
    be called from different threads (as happens in ``QueryRunner``).
    """

    # SLO-violation tolerance for the lexicographic comparison (seconds).
    TOLERANCE_S = 1e-4

    def __init__(
        self,
        iconq_model_id: str,
        eligible_cluster_names: list[str],
        default_slo_s: float = 10.0,
        slo_overrides: Optional[dict[int, float]] = None,
        optimize_by_amount: bool = True,
        on_capacity_pressure: Optional[Callable[[], None]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        iconq_model_id:
            Identifier passed to ``IconqModel.load()``.
        eligible_cluster_names:
            Cluster names that the router may route to.  They must exist
            in the system configuration.
        default_slo_s:
            Default latency SLO (seconds) for templates without an
            explicit override.
        slo_overrides:
            Optional ``{template_id: slo_s}`` dict for per-template SLOs.
        optimize_by_amount:
            If True, violations are measured in seconds of overshoot.
            If False, violations are binary (0/1) per query.
        on_capacity_pressure:
            Optional callback invoked (without arguments) when the router
            determines that *every* eligible cluster would incur an SLO
            violation for the incoming query.  The capacity controller
            hooks into this to trigger spin-up.
        """
        super().__init__(*args, **kwargs)

        # Model ---------------------------------------------------------------
        self._iconq_model_id = iconq_model_id
        self._iconq_model = IconqModel.load(model_id=iconq_model_id)

        # Clusters & blueprint ------------------------------------------------
        self._eligible_cluster_names = list(eligible_cluster_names)
        self._blueprint = Blueprint(
            clusters=[Cluster.from_config(cn) for cn in eligible_cluster_names]
        )
        self._cost_per_second: dict[str, float] = {
            cn: Cluster.from_config(cn).cost_per_second
            for cn in eligible_cluster_names
        }

        # SLO ------------------------------------------------------------------
        self._slo_resolver = SloResolver.from_dict(
            default_slo_s=default_slo_s,
            slo_dict=slo_overrides or {},
        )
        self._optimize_by_amount = optimize_by_amount

        # Per-cluster state (guarded by _lock) ---------------------------------
        self._lock = threading.Lock()
        # query_id → Query
        self._active_queries: dict[str, dict[str, Query]] = {
            cn: {} for cn in eligible_cluster_names
        }
        # Billing window tracking: cluster → start time of the current
        # billing window (None until the first query arrives).
        self._billing_window_start_s: dict[str, Optional[float]] = {
            cn: None for cn in eligible_cluster_names
        }
        # Per-query neighbor history: query_id → list of Query objects
        # that were / are co-running on the same cluster.  Mirrors the
        # simulator's _neighbors_per_active_query.  When a new query Q
        # starts on cluster C:
        #   1. Q's neighbor list = copy of current active queries on C.
        #   2. Q is appended to every existing active query's neighbor list.
        # When Q finishes, its entry is deleted, but Q *stays* in the
        # neighbor lists of queries that are still running (providing the
        # model with the full co-runner history).
        self._neighbors_per_active_query: dict[str, list[Query]] = {}

        # Per-cluster set of recently-touched tables (for future cache
        # affinity scoring).
        self._recent_tables: dict[str, set[str]] = {
            cn: set() for cn in eligible_cluster_names
        }

        # Capacity pressure callback ------------------------------------------
        self._on_capacity_pressure = on_capacity_pressure

        # RPU lookup (supports both config-based and dynamic clusters) ------
        self._rpu_per_cluster: dict[str, int] = {
            cn: Cluster.from_config(cn).rpu for cn in eligible_cluster_names
        }

        self._name = (
            f"RAutoSLO(iconq_model_id={repr(iconq_model_id)}, "
            f"default_slo_s={default_slo_s})"
        )

    # ------------------------------------------------------------------
    # QueryRouter interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def blueprint(self) -> Blueprint:
        return self._blueprint

    @property
    def slo_resolver(self) -> SloResolver:
        """Expose the resolver so the capacity controller can use it."""
        return self._slo_resolver

    def route_query(
        self,
        query_id: Any = None,
        tpcds_temp_and_q_idx: Any = None,
        start_time_s: Optional[float] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Route a query to the best eligible cluster.

        Parameters
        ----------
        query_id:
            Unique identifier for the query.  Falls back to ``kwargs``
            entries ``"seq_num"`` or ``"query_id"`` for compatibility
            with ``QueryRunner``.
        tpcds_temp_and_q_idx:
            TPC-DS template and query index string, e.g. ``"042_001"``.
        start_time_s:
            Arrival wall-clock (or simulated) time in seconds.  Defaults
            to ``time.time()``.

        Returns
        -------
        The name of the cluster the query should be sent to.
        """
        # Normalise arguments for compatibility with different callers.
        if query_id is None:
            query_id = kwargs.get("seq_num", kwargs.get("query_id"))
        if tpcds_temp_and_q_idx is None:
            tpcds_temp_and_q_idx = kwargs.get("tpcds_temp_and_q_idx")
        if start_time_s is None:
            start_time_s = time.time()

        query_id = str(query_id)
        tpcds_temp_and_q_idx = str(tpcds_temp_and_q_idx)

        # Build featurisation for the incoming query.
        featurization = self._iconq_model.iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
            tpcds_temp_and_q_idx
        )

        incoming = Query(
            query_id=query_id,
            tpcds_temp_and_q_idx=tpcds_temp_and_q_idx,
            rel_start_time_s=start_time_s,
            featurization=featurization,
            latency_s=-1,  # unknown
        )

        # -- Snapshot current state under the lock ----------------------------
        with self._lock:
            snapshots: dict[str, ClusterSnapshot] = {}
            run_to_base_to_neighbors: dict[str, dict[Query, list[Query]]] = {}

            for cn in self._eligible_cluster_names:
                active_list = list(self._active_queries[cn].values())

                snapshots[cn] = ClusterSnapshot(
                    cluster_name=cn,
                    cost_per_second=self._cost_per_second[cn],
                    active_queries=active_list,
                    billing_window_start_s=self._billing_window_start_s[cn],
                )

                # Neighbor sets: for each active query, use its
                # accumulated co-runner history (which may include
                # already-finished queries) plus the incoming query.
                # The incoming query itself sees only the currently
                # active queries + itself (it has no history yet).
                run_to_base_to_neighbors[cn] = {
                    q: self._neighbors_per_active_query[q.query_id] + [incoming]
                    for q in active_list
                }
                run_to_base_to_neighbors[cn][incoming] = active_list + [
                    incoming
                ]

        # -- Stage-model prediction (per cluster) ----------------------------
        stage_latency_predictions: dict[str, float] = {}
        for cn in self._eligible_cluster_names:
            incoming.cluster_name = cn
            stage_pred = (
                self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                    {query_id: tpcds_temp_and_q_idx}, cn
                )[query_id].overall_mean_s()
            )
            incoming.stage_latency_prediction_s = stage_pred
            stage_latency_predictions[cn] = stage_pred

        # -- Before-state per cluster ----------------------------------------
        before: dict[str, tuple[float, float]] = {}
        for cn, snap in snapshots.items():
            incoming.cluster_name = cn
            incoming.stage_latency_prediction_s = stage_latency_predictions[cn]
            before[cn] = RoutingCore.compute_before_state(
                snapshot=snap,
                current_time_s=start_time_s,
                slo_resolver=self._slo_resolver,
                optimize_by_amount=self._optimize_by_amount,
            )

        # -- Batched model prediction across all clusters --------------------
        dataset = ConcurrentQueryDataset.build_from_query_groups(
            iconq_interaction_featurizer=(
                self._iconq_model.iconq_interaction_featurizer
            ),
            run_to_base_to_neighbors=run_to_base_to_neighbors,
        )
        all_predictions = self._iconq_model.predict_from_dataset(dataset)

        # -- Score each cluster ----------------------------------------------
        scores: list[PlacementScore] = []
        for cn, predictions in all_predictions.items():
            bc, bv = before[cn]
            score = RoutingCore.score_placement(
                query=incoming,
                snapshot=snapshots[cn],
                predictions=predictions,
                current_time_s=start_time_s,
                slo_resolver=self._slo_resolver,
                optimize_by_amount=self._optimize_by_amount,
                before_cost=bc,
                before_slo_violation=bv,
            )
            scores.append(score)

        if not scores:
            # Fallback: shouldn't happen if eligible_cluster_names is non-empty
            logger.warning(
                "No scores produced for query %s; falling back to first "
                "eligible cluster.",
                query_id,
            )
            return self._eligible_cluster_names[0]

        best = RoutingCore.pick_best(scores, tolerance=self.TOLERANCE_S)

        # Emit capacity-pressure signal if every cluster would violate SLO.
        if all(s.marginal_slo_violation > 0 for s in scores):
            logger.info(
                "Capacity pressure: all %d clusters have positive marginal "
                "SLO violation for query %s.",
                len(scores),
                query_id,
            )
            if self._on_capacity_pressure is not None:
                try:
                    self._on_capacity_pressure()
                except Exception:
                    logger.exception("capacity_pressure callback failed")

        return best.cluster_name

    # ------------------------------------------------------------------
    # Lifecycle hooks (called by QueryRunner)
    # ------------------------------------------------------------------

    def on_query_start(
        self,
        query_id: Any,
        cluster_name: str,
        tpcds_temp_and_q_idx: TPCDSTempAndQIdx,
        current_time_s: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register a query as actively running on *cluster_name*."""
        if tpcds_temp_and_q_idx is None:
            tpcds_temp_and_q_idx = kwargs.get("tpcds_temp_and_q_idx", "0_0")

        query_id = str(query_id)
        featurization = self._iconq_model.iconq_query_featurizer.featurize_from_tpcds_temp_and_q_idx(
            str(tpcds_temp_and_q_idx)
        )
        stage_pred = (
            self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                {query_id: str(tpcds_temp_and_q_idx)}, cluster_name
            )[query_id].overall_mean_s()
        )

        q = Query(
            query_id=query_id,
            tpcds_temp_and_q_idx=str(tpcds_temp_and_q_idx),
            rel_start_time_s=current_time_s,
            cluster_name=cluster_name,
            featurization=featurization,
            stage_latency_prediction_s=stage_pred,
            latency_s=stage_pred,  # best estimate until finish
        )

        with self._lock:
            # Initialize this query's neighbor list with all currently
            # active co-runners, and append it to each of their lists.
            current_actives = list(self._active_queries[cluster_name].values())
            self._neighbors_per_active_query[query_id] = list(current_actives)
            for active_q in current_actives:
                self._neighbors_per_active_query[active_q.query_id].append(q)

            self._active_queries[cluster_name][query_id] = q
            # If this is the first query on this cluster, start a billing
            # window.
            if self._billing_window_start_s[cluster_name] is None:
                self._billing_window_start_s[cluster_name] = current_time_s

    def on_query_finish(
        self,
        query_id: Any,
        cluster_name: str,
        current_time_s: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Remove a query from the active set for *cluster_name*."""
        query_id = str(query_id)
        with self._lock:
            if query_id not in self._active_queries[cluster_name]:
                raise KeyError(
                    f"Query {query_id!r} not found in active queries "
                    f"for cluster {cluster_name}."
                )
            del self._active_queries[cluster_name][query_id]
            # Remove the neighbor-history entry (the query object itself
            # persists inside other queries' neighbor lists, which is
            # the desired behaviour).
            self._neighbors_per_active_query.pop(query_id, None)

            # If the cluster has no more running queries, close the billing
            # window only if it has also lasted at least aslong as specified
            # by the threshold.
            billing_window_start = self._billing_window_start_s[cluster_name]
            if (
                (not self._active_queries[cluster_name])
                and (billing_window_start is not None)
                and (
                    billing_window_start + Billing.REDSHIFT_BILLING_THRESHOLD_S
                    <= current_time_s
                )
            ):
                self._billing_window_start_s[cluster_name] = None

    # ------------------------------------------------------------------
    # Introspection helpers (used by capacity controller)
    # ------------------------------------------------------------------

    def get_active_queries(self, cluster_name: str) -> list[Query]:
        """Return a copy of the active query list for *cluster_name*."""
        with self._lock:
            return list(self._active_queries[cluster_name].values())

    def get_all_active_queries(self) -> dict[str, list[Query]]:
        """Return a snapshot of active queries across all clusters."""
        with self._lock:
            return {
                cn: list(qs.values()) for cn, qs in self._active_queries.items()
            }

    def get_slo_headroom(self) -> float:
        """Compute the minimum SLO headroom across all active queries on
        all clusters."""
        all_active: list[Query] = []
        with self._lock:
            for qs in self._active_queries.values():
                all_active.extend(qs.values())
        return RoutingCore.compute_slo_headroom(all_active, self._slo_resolver)

    def get_cluster_headroom(self, cluster_name: str) -> float:
        """Compute SLO headroom for a single cluster."""
        with self._lock:
            active = list(self._active_queries[cluster_name].values())
        return RoutingCore.compute_slo_headroom(active, self._slo_resolver)

    # ------------------------------------------------------------------
    # Dynamic cluster management
    # ------------------------------------------------------------------

    def add_cluster(self, cluster: Cluster) -> None:
        """Register a dynamically provisioned cluster for routing.

        Initialises all per-cluster bookkeeping so that the new cluster
        becomes eligible for ``route_query`` immediately.

        Parameters
        ----------
        cluster :
            The cluster to add.  Must not already be registered.

        Raises
        ------
        ValueError
            If a cluster with the same name is already registered.
        """
        cn = cluster.name
        with self._lock:
            if cn in self._active_queries:
                raise ValueError(
                    f"Cluster {cn!r} is already registered."
                )
            self._eligible_cluster_names.append(cn)
            self._active_queries[cn] = {}
            self._billing_window_start_s[cn] = None
            self._cost_per_second[cn] = cluster.cost_per_second
            self._recent_tables[cn] = set()
            self._rpu_per_cluster[cn] = cluster.rpu

    def remove_cluster(self, cluster_name: str) -> None:
        """
        Unregister a cluster, making it ineligible for routing. Fails if there
        are active queries on the cluster, or if the cluster is not registered.

        Parameters
        ----------
        cluster_name :
            Name of the cluster to remove.

        Raises
        ------
        KeyError
            If the cluster name is not registered.
        ValueError
            If there are active queries on the cluster.  The caller is 
            responsible for ensuring that the cluster is no longer being routed 
            to and that all active queries have finished before calling this 
            method.  

        """
        with self._lock:
            if cluster_name not in self._active_queries:
                raise KeyError(
                    f"Cluster {cluster_name!r} is not registered."
                )
            if self._active_queries[cluster_name]:
                raise ValueError(
                    f"Cluster {cluster_name!r} has active queries."
                )
            self._eligible_cluster_names.remove(cluster_name)
            # Clean up neighbor refs for any still-active queries on
            # the removed cluster.
            for qid in list(self._active_queries[cluster_name].keys()):
                self._neighbors_per_active_query.pop(qid, None)
            del self._active_queries[cluster_name]
            del self._billing_window_start_s[cluster_name]
            del self._cost_per_second[cluster_name]
            del self._recent_tables[cluster_name]
            del self._rpu_per_cluster[cluster_name]

    def get_rpu(self, cluster_name: str) -> int:
        """Return the RPU for *cluster_name*.

        Works for both config-based and dynamically added clusters.
        """
        with self._lock:
            return self._rpu_per_cluster[cluster_name]
