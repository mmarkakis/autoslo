import argparse
import asyncio
import concurrent.futures
import logging
import os
import queue
import threading
from functools import partial
from pathlib import Path
from typing import Optional, TypeAlias

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from autoslo.clusters.actions import ScalingAction, SpinUpAction, TearDownAction
from autoslo.clusters.cluster import Cluster, ClusterState, ClusterView
from autoslo.clusters.redshift_provisioner import RedshiftServerlessProvisioner
from autoslo.clusters.scheduled_spinup import ScheduledSpinUp
from autoslo.config.execution_config import ExecutionConfig
from autoslo.config.utils import make_run_id, parse_params
from autoslo.filesystem.path_utils import append_to_run_log
from autoslo.filesystem.structured_events import (
    BaseStructuredEvent,
    EventType,
    QueryRelatedEvent,
    wall_clock_utc,
)
from autoslo.filesystem.structured_log import emit_structured
from autoslo.filesystem.yaml_helpers import dump_yaml, load_yaml_with_params
from autoslo.routing.wrapper import route_and_update_bookkeeping
from autoslo.workload_definition.query import Query, QueryTextId
from autoslo.workload_definition.query_text_registry import QueryTextRegistry
from autoslo.workload_definition.schema import Schema
from autoslo.workload_definition.workload import Workload

# ---------------------------------------------------------------------------
# Background autoscaler thread helpers
# ---------------------------------------------------------------------------


class _ArrivalForAutoscalerProxy:
    """
    Enqueued by _AutoscalerProxy when a query arrives; forwarded to
    ``Autoscaler.inform()`` on the background thread.
    """

    __slots__ = ("rel_time_s", "query", "snapshot")

    def __init__(
        self,
        rel_time_s: float,
        query: "Query",
        snapshot: "dict[str, ClusterView]",
    ) -> None:
        self.rel_time_s = rel_time_s
        self.query = query
        self.snapshot = snapshot


class _CompletionForAutoscalerProxy:
    """
    Enqueued when a query finishes (successfully or not); forwarded to
    ``Autoscaler.record_completion()`` on the background thread.

    ``latency_s`` is ``None`` for failed queries.
    """

    __slots__ = ("rel_time_s", "query", "latency_s")

    def __init__(
        self,
        rel_time_s: float,
        query: "Query",
        latency_s: Optional[float],
    ) -> None:
        self.rel_time_s = rel_time_s
        self.query = query
        self.latency_s = latency_s


_AutoscalerProxyEvent: TypeAlias = (
    _CompletionForAutoscalerProxy | _ArrivalForAutoscalerProxy | None
)


class _AutoscalerProxy:
    """Passed to route_and_update_bookkeeping in place of the real Autoscaler.

    inform() enqueues the call onto the background thread's queue and returns
    [] immediately, removing autoscaler work from the routing critical path.
    SpinUp/TearDown actions are dispatched by the background thread instead.
    """

    def __init__(self, q: queue.SimpleQueue[_AutoscalerProxyEvent]) -> None:
        self._queue = q

    def inform(
        self,
        rel_time_s: float,
        current_query: "Query",
        pool_snapshot_with_current_query: "dict[str, ClusterView]",
    ) -> list[ScalingAction]:
        self._queue.put(
            _ArrivalForAutoscalerProxy(
                rel_time_s, current_query, pool_snapshot_with_current_query
            )
        )
        return []


class WorkloadRunner:
    """Execute a workload against live Redshift Serverless clusters.

    The constructor mirrors :class:`~WorkloadSimulator`'s signature so that
    both can be driven from the same YAML configuration (see
    :meth:`from_config`).  Runner-specific settings (``provisioner``,
    ``maxconns``, ``closed_loop``) live in a dedicated ``runner_config``
    section.
    """

    def __init__(
        self,
        cfg: dict,
        out_dir: Optional[str | Path] = None,
        write_text_log: bool = True,
    ) -> None:
        """
        Initialize the runner with the given configuration.
        """
        # ── Build, parse and dump structured config ──────────────────────────
        self._write_text_log = write_text_log
        execution_config = ExecutionConfig.build(
            cfg=cfg,
            out_dir=out_dir,
            write_text_log=write_text_log,
            is_runner=True,
        )
        self._run_id = execution_config.run_id
        self._out_dir = execution_config.out_dir
        self._workload = execution_config.workload
        self._pool = execution_config.pool
        self._scheduled_spinups = execution_config.scheduled_spinups
        self._router = execution_config.router
        self._autoscaler = execution_config.autoscaler
        self._structured_handler = execution_config.structured_log_handler
        self._workload_runner_config = execution_config.workload_runner_config

        self._closed_loop = self._workload_runner_config.closed_loop
        schema_name = self._workload_runner_config.schema_name
        self._query_text_registry = QueryTextRegistry(
            schema_name, one_statement_per_query=True
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._workload_runner_config.max_threads
        )

        dump_yaml(cfg, os.path.join(self._out_dir, "execution_config.yml"))

        # ── Instance Variables ───────────────────────────────────────────────
        self._routing_lock = threading.Lock()
        self._spin_ups_lock = threading.Lock()
        self._pending_spin_ups: list[concurrent.futures.Future] = []
        self._autoscaler_queue: queue.SimpleQueue[_AutoscalerProxyEvent] = (
            queue.SimpleQueue()
        )

    @property
    def workload(self) -> Workload:
        """The Workload being executed."""
        return self._workload

    # ------------------------------------------------------------------
    # Async scheduled spin-up (live runner)
    # ------------------------------------------------------------------

    async def _execute_scheduled_spinup_async(
        self,
        spinup: ScheduledSpinUp,
    ) -> None:
        """Wait until *spinup.rel_time_s* elapses, then spin up."""
        delay = spinup.rel_time_s - self._rel_time_s()
        if delay > 0:
            await asyncio.sleep(delay)
        spinup.execute(
            source="WorkloadRunner",
            on_spin_up=self._on_live_spin_up,
            write_text_log=self._write_text_log,
            rel_time_s_getter=self._rel_time_s,
        )

    # ------------------------------------------------------------------
    # Autoscaler callbacks
    # ------------------------------------------------------------------

    def _on_live_spin_up(self, action: SpinUpAction) -> None:
        """Autoscaler callback: spin up a new cluster (non-blocking).

        Submits the blocking provisioning to the thread-pool executor.
        Thread-safe — may be called from any thread (event loop,
        executor thread running routing, etc.).
        """

        future = self._executor.submit(
            self._pool.request_spin_up, action, self._rel_time_s()
        )

        def _on_spin_up_done(fut: concurrent.futures.Future) -> None:
            exc = fut.exception()
            if exc is not None:
                # Provisioning failed; clear the in-flight flag so the
                # autoscaler can attempt another spin-up on the next eligible
                # query rather than being blocked permanently.
                self._autoscaler.clear_spin_up_in_flight()
                return
            cluster_name = fut.result()
            if cluster_name is None:
                # Spin-up was denied by the budget; SPIN_UP_BLOCKED was
                # already emitted by the pool.  Disable future spin-up
                # considerations in the autoscaler.
                self._autoscaler.disable_spin_up()
                return
            rpu = Cluster.rpu_for_cluster_name(cluster_name)
            emit_structured(
                BaseStructuredEvent(
                    rel_time_s=self._rel_time_s(),
                    event_type=EventType.CLUSTER_READY,
                    source="WorkloadRunner",
                    cluster_name=cluster_name,
                    details={
                        "rpu": rpu,
                        "num_active_clusters": len(
                            self._pool.clusters_in_state(ClusterState.READY)
                        ),
                    },
                )
            )
            for cn in action.deferred_teardowns:
                self._pool.request_tear_down(
                    TearDownAction(reason="Deferred teardown", cluster_name=cn),
                    self._rel_time_s(),
                )

        future.add_done_callback(_on_spin_up_done)
        with self._spin_ups_lock:
            self._pending_spin_ups.append(future)

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def _rel_time_s(self) -> float:
        """Relative time in seconds since run start."""
        return wall_clock_utc() - self._async_reference_ts

    def _route_locked(self, query: Query) -> str:
        """Route a query under the serialisation lock.

        The lock ensures that routing (snapshot → model inference →
        ``on_query_start`` → autoscaler) is atomic, preventing
        concurrent routing calls from clobbering each other's
        ``predicted_latencies`` updates.

        autoscaler.inform() is dispatched to the background autoscaler
        thread via _AutoscalerProxy; it does not block routing.

        Called from an executor thread — never from the event loop.
        """
        with self._routing_lock:
            cluster_name, _ = route_and_update_bookkeeping(
                source="WorkloadRunner",
                rel_time_s_getter=self._rel_time_s,
                pool=self._pool,
                router=self._router,
                query=query,
                autoscaler=_AutoscalerProxy(self._autoscaler_queue),
                simulator_pending_events_heap=None,
            )
            # We can ignore the second return value (autoscaler actions) because
            # AutoscalerProxy enqueues the work item for the background thread
            # and returns [] immediately, instead of returning real actions to
            # be dispatched here.
            return cluster_name

    def _autoscaler_loop(self) -> None:
        """Background thread: drain _autoscaler_queue and call the autoscaler.

        Runs for the lifetime of run().  Exits when it dequeues the sentinel
        (None) that run()'s finally block enqueues after all routing is done.
        Actions returned by inform() are dispatched here, mirroring the
        dispatch logic in route_and_update_bookkeeping.
        """
        while True:
            item = self._autoscaler_queue.get()
            if item is None:  # sentinel — time to exit
                break
            if isinstance(item, _CompletionForAutoscalerProxy):
                try:
                    self._autoscaler.record_completion(
                        query=item.query,
                        latency_s=item.latency_s,
                        rel_time_s=item.rel_time_s,
                    )
                except Exception:
                    logging.exception(
                        (
                            "Autoscaler background thread failed to record "
                            "completion for query %s; continuing."
                        ),
                        item.query.query_id,
                    )
            else:
                try:
                    actions = self._autoscaler.inform(
                        rel_time_s=item.rel_time_s,
                        current_query=item.query,
                        pool_snapshot_with_current_query=item.snapshot,
                    )
                    for action in actions:
                        if isinstance(action, SpinUpAction):
                            self._on_live_spin_up(action)
                        elif isinstance(action, TearDownAction):
                            self._pool.request_tear_down(
                                action, self._rel_time_s()
                            )
                        elif self._write_text_log:
                            logging.warning(
                                "Unknown autoscaling action type: %s",
                                type(action),
                            )
                except Exception:
                    logging.exception(
                        (
                            "Autoscaler background thread failed for query %s; "
                            "continuing."
                        ),
                        item.query.query_id,
                    )

    def _run_query_sync(
        self,
        query_id: str,
        query_text_id: QueryTextId,
        query_text: str,
        cluster_name: str,
    ) -> Optional[float]:
        """
        Run a single query synchronously.

        Returns the measured client-side latency in seconds on success,
        or ``None`` if the query failed (connection error, SQL error, etc.).
        """
        logging.info(f"Starting query {query_id}")
        start_rel_time_s = self._rel_time_s()
        emit_structured(
            QueryRelatedEvent(
                rel_time_s=start_rel_time_s,
                event_type=EventType.QUERY_EXECUTION_START,
                source="WorkloadRunner",
                cluster_name=cluster_name,
                query_id=query_id,
                query_text_id=query_text_id,
            )
        )
        try:
            conn = self._pool.getconn(cluster_name)
        except Exception as e:
            logging.exception(
                f"Query {query_id} failed to acquire connection: {e}"
            )
            return None
        succeeded = False
        try:
            with conn.cursor() as cur:
                edited = f"--{self._run_id}/{query_id}\n{query_text}"
                cur.execute(edited)
                try:
                    _ = cur.fetchall()
                except Exception as e:
                    pass  # Some queries do not return results.
            conn.commit()
            succeeded = True
        except Exception as e:
            # Ensure errors don't prevent returning the connection to the pool.
            try:
                conn.rollback()
            except Exception:
                pass
            logging.exception(f"Query {query_id} failed: {e}")
        finally:
            self._pool.putconn(cluster_name, conn)
        end_rel_time_s = self._rel_time_s()
        latency_s = end_rel_time_s - start_rel_time_s
        logging.info(f"Query {query_id} finished after t={latency_s:.2f}s")
        emit_structured(
            QueryRelatedEvent(
                rel_time_s=end_rel_time_s,
                event_type=EventType.QUERY_EXECUTION_FINISH,
                source="WorkloadRunner",
                cluster_name=cluster_name,
                details={"latency_s": latency_s},
                query_id=query_id,
                query_text_id=query_text_id,
            )
        )
        return latency_s if succeeded else None

    async def _run_query_async(
        self,
        async_reference_ts: float,
        query: Query,
        skip_wait: bool = False,
    ) -> None:
        """
        Run a single query asynchronously, waiting until its scheduled start.

        Routing is performed **after** the sleep so that the routing
        decision reflects the live pool state at the query's actual
        arrival time — not the pool state at the start of the run.

        Parameters:
            async_reference_ts: Reference timestamp for scheduling.
            query: The Query object to execute.
            skip_wait: If True, skip the initial sleep and route immediately.
        """
        loop = asyncio.get_running_loop()
        selected_cluster_name: Optional[str] = None
        latency_s: Optional[float] = None
        try:
            delay = query.rel_start_time_s - self._rel_time_s()

            if skip_wait:
                logging.info(
                    f"Query {query.query_id} scheduled to start at "
                    f"relative time {query.rel_start_time_s:.2f}s "
                    f"(in {delay:.2f}s), but skip_wait=True. Starting immediately."
                )
            elif delay > 0:
                logging.info(
                    f"Query {query.query_id} scheduled to start at "
                    f"relative time {query.rel_start_time_s:.2f}s "
                    f"(in {delay:.2f}s). Waiting..."
                )
                await asyncio.sleep(delay)
            else:  # delay <= 0
                logging.warning(
                    f"Query {query.query_id} scheduled start time "
                    f"relative time {query.rel_start_time_s:.2f}s "
                    f"is in the past (delay={delay:.2f}s). Starting immediately."
                )

            # ── Find query text ────────────────────────────────────────────
            query_text = self._query_text_registry.get(query.query_text_id)
            if query_text is None:
                logging.error(
                    f"No text found for schema "
                    f"'{self._query_text_registry.schema_name}', "
                    f"query_text_id '{query.query_text_id}'. Skipping query "
                    f"{query.query_id}."
                )
                return

            # ── Emit arrival event (matches simulator) ────────────────
            emit_structured(
                QueryRelatedEvent(
                    rel_time_s=self._rel_time_s(),
                    event_type=EventType.ARRIVAL,
                    source="WorkloadRunner",
                    query_id=query.query_id,
                    query_text_id=query.query_text_id,
                )
            )

            # ── Route at arrival time ────────────────────────────────────
            # Routing involves model inference (CPU-bound) and must be
            # serialised to keep pool state consistent, so it runs in the
            # thread-pool under _routing_lock.
            selected_cluster_name = await loop.run_in_executor(
                self._executor, self._route_locked, query
            )

            # ── Execute ──────────────────────────────────────────────────
            fn = partial(
                self._run_query_sync,
                query.query_id,
                query.query_text_id,
                query_text,
                selected_cluster_name,
            )
            latency_s = await loop.run_in_executor(self._executor, fn)
        finally:
            if selected_cluster_name is not None:
                # on_query_finish may trigger _finalize_removal, which is
                # dispatched to the pool's background executor when one is
                # configured.  The bookkeeping itself is fast but we still
                # run it off the event loop to avoid lock contention.
                emit_structured(
                    QueryRelatedEvent(
                        rel_time_s=self._rel_time_s(),
                        event_type=EventType.COMPLETION,
                        source="WorkloadRunner",
                        cluster_name=selected_cluster_name,
                        details={"success": (latency_s is not None)},
                        query_id=query.query_id,
                        query_text_id=query.query_text_id,
                    )
                )
                self._autoscaler_queue.put(
                    _CompletionForAutoscalerProxy(
                        rel_time_s=self._rel_time_s(),
                        query=query,
                        latency_s=latency_s,
                    )
                )
                loop.run_in_executor(
                    self._executor,
                    partial(
                        self._pool.on_query_finish,
                        query_id=query.query_id,
                        cluster_name=selected_cluster_name,
                        rel_time_s=self._rel_time_s(),
                    ),
                )
            self._pbar.update(self._pbar_task, advance=1)

    async def run(self) -> None:
        """
        Run the queries from the workload file.
        """

        print(f"Run starting with ID {self._run_id}.")

        # Spin up initial clusters.
        schema = Schema.load(self._workload_runner_config.schema_name)
        self._pool.add_details_and_spin_up_initial_clusters(
            search_path=schema.search_path,
            background_executor=self._executor,
            run_id=self._run_id,
            out_dir=self._out_dir,
            write_text_log=self._write_text_log,
        )

        # Add a 30-second buffer between the reference timestamp and the first
        # query's scheduled start time for setup.
        self._async_reference_ts = wall_clock_utc() + 30

        # Propagate reference time to the provisioner so it can compute
        # relative timestamps for cluster creation_time_s and events.
        prov = self._pool.provisioner
        if isinstance(prov, RedshiftServerlessProvisioner):
            prov.reference_time_s = self._async_reference_ts

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=self._rel_time_s(),
                event_type=EventType.RUN_START,
                source="WorkloadRunner",
                details={
                    "workload_name": self._workload.workload_name,
                    "num_queries": self._workload.num_queries,
                    "routing_policy": self._router.routing_policy.value,
                    "closed_loop": self._closed_loop,
                },
            )
        )

        async_reference_ts = self._async_reference_ts
        logging.info(f"Async reference timestamp: {async_reference_ts:.2f}s")

        tasks: list[asyncio.Task] = []

        # Schedule spin-up executions.
        for su in self._scheduled_spinups:
            tasks.append(
                asyncio.ensure_future(self._execute_scheduled_spinup_async(su))
            )

        self._pbar = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._pbar_task = self._pbar.add_task(
            "Queries", total=self._workload.num_queries
        )
        self._pbar.start()

        # Start the background autoscaler thread.  It will drain
        # _autoscaler_queue until it sees the None sentinel in finally.
        autoscaler_thread = threading.Thread(
            target=self._autoscaler_loop,
            name="autoscaler-bg",
            daemon=True,
        )
        autoscaler_thread.start()

        try:
            for query in self._workload.queries():

                if not self._closed_loop:
                    task = asyncio.ensure_future(
                        self._run_query_async(async_reference_ts, query)
                    )
                    tasks.append(task)
                else:
                    # In closed loop, wait for each query to finish before
                    # starting the next.  Ignore rel_start_time_s.
                    await self._run_query_async(
                        async_reference_ts, query, skip_wait=True
                    )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logging.error("Query task failed: %s", r)

            # Wait for any in-flight background spin-ups to finish
            # before tearing down clusters.
            with self._spin_ups_lock:
                spin_up_snapshot = list(self._pending_spin_ups)
            if spin_up_snapshot:
                async_futs = [asyncio.wrap_future(f) for f in spin_up_snapshot]
                spin_up_results = await asyncio.gather(
                    *async_futs, return_exceptions=True
                )
                for r in spin_up_results:
                    if isinstance(r, Exception):
                        logging.error("Background spin-up failed: %s", r)
        finally:
            # Signal the background autoscaler thread to finish processing any
            # remaining queued items and exit before we begin cluster teardown.
            self._autoscaler_queue.put(None)
            autoscaler_thread.join()

            self._pbar.stop()

            # Graceful cleanup: tear down every remaining READY cluster.
            # request_tear_down transitions to DRAINING; _finalize_removal
            # (stats collection + AWS API calls) runs in the pool's
            # background executor.
            loop = asyncio.get_running_loop()
            remaining = list(self._pool.clusters_in_state(ClusterState.READY))
            for cn in remaining:
                try:
                    emit_structured(
                        BaseStructuredEvent(
                            rel_time_s=self._rel_time_s(),
                            event_type=EventType.TEAR_DOWN_DECISION,
                            source="WorkloadRunner",
                            cluster_name=cn,
                            details={"reason": "run_cleanup"},
                        )
                    )
                    await loop.run_in_executor(
                        self._executor,
                        partial(
                            self._pool.request_tear_down,
                            TearDownAction(
                                reason="run_cleanup",
                                cluster_name=cn,
                            ),
                            self._rel_time_s(),
                            force=True,
                        ),
                    )
                except Exception:
                    logging.exception("Failed to tear down cluster %s.", cn)

            # Wait for all background finalization tasks (stats collection,
            # provisioner tear-down) that the pool dispatched.
            self._pool.wait_for_background_tasks()

        logging.info(f"Run finished at {wall_clock_utc()}.")

        emit_structured(
            BaseStructuredEvent(
                rel_time_s=self._rel_time_s(),
                event_type=EventType.RUN_FINISH,
                source="WorkloadRunner",
                details={
                    "workload_name": self._workload.workload_name,
                },
            )
        )

        # Finalize structured log (consolidate shards).
        if (
            hasattr(self, "_structured_handler")
            and self._structured_handler is not None
        ):
            self._structured_handler.finalize()

    @property
    def run_id(self) -> str:
        """The unique identifier assigned to this run."""
        return self._run_id


if __name__ == "__main__":

    description = "Run the WorkloadRunner using a YAML execution config file."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "execution_config",
        help="Path to the YAML execution config file.",
    )
    parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Substitute <KEY> placeholder in the config with VALUE. "
            "May be repeated: --param TARGET_DATE=2024-05-27."
        ),
    )
    args = parser.parse_args()

    params = parse_params(args.param)
    cfg = load_yaml_with_params(args.execution_config, params)
    config_id = make_run_id([Path(args.execution_config).stem], params)

    qr = WorkloadRunner(cfg)
    append_to_run_log(
        run_id=qr.run_id,
        config_id=config_id,
        workload_id=qr.workload.workload_config.id(),
    )
    asyncio.run(qr.run())
