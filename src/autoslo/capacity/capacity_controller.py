"""
capacity_controller.py
----------------------
SLO-headroom-based autoscaler (Layer 2).

Runs in a background thread, periodically inspecting the active queries
on all clusters via the router's introspection API.  When SLO headroom
drops below a critical threshold (or the router emits a capacity-pressure
signal), the controller triggers a spin-up.  When a cluster is idle for
a configurable number of consecutive periods, a tear-down is triggered.

Spin-up / tear-down actions are communicated through callbacks so that
callers (the ``QueryRunner``, the simulator, or a test harness) can wire
them to the appropriate infra API or simulation event.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from autoslo.routing.routing_core import RoutingCore
from autoslo.workload_definition.query import Query

logger = logging.getLogger(__name__)


class CapacityController:
    """Background controller that watches SLO headroom and drives
    cluster spin-up / tear-down decisions.

    Parameters
    ----------
    get_active_queries:
        Callable that returns ``{cluster_name: [Query, ...]}``.
        Typically ``router.get_all_active_queries``.
    slo_resolver:
        Resolves per-query SLOs (shared with the router).
    on_spin_up:
        Callback ``(reason: str, rpu: int) -> None`` invoked when a
        spin-up is needed.  *rpu* is the RPU size the controller
        recommends (the smallest from *allowed_rpu_sizes*).
    on_tear_down:
        Callback ``(cluster_name: str) -> None`` invoked when a specific
        cluster should be torn down.
    poll_interval_s:
        How often (seconds) the controller checks headroom.  Default 60.
    eta_crit:
        Critical headroom threshold.  If ``H_t ≤ eta_crit``, spin up.
        Default 0.1 (10 % remaining headroom).
    idle_periods_before_tear_down:
        Number of consecutive poll periods a cluster must remain idle
        (zero active queries) before ``on_tear_down`` is called.
        Default 5.
    allowed_rpu_sizes:
        RPU sizes available for dynamic spin-up (sorted ascending).
        If *None*, defaults to ``[8]`` (a safe single-size default).
        The controller always picks the **smallest** available RPU.
    """

    def __init__(
        self,
        get_active_queries: Callable[[], dict[str, list[Query]]],
        slo_resolver: "SloResolver",  # type: ignore[name-defined]  # noqa: F821
        on_spin_up: Optional[Callable[[str, int], None]] = None,
        on_tear_down: Optional[Callable[[str], None]] = None,
        poll_interval_s: float = 60.0,
        eta_crit: float = 0.1,
        idle_periods_before_tear_down: int = 5,
        allowed_rpu_sizes: Optional[list[int]] = None,
    ) -> None:
        self._get_active_queries = get_active_queries
        self._slo_resolver = slo_resolver
        self._on_spin_up = on_spin_up
        self._on_tear_down = on_tear_down
        self._poll_interval_s = poll_interval_s
        self._eta_crit = eta_crit
        self._idle_periods_before_tear_down = idle_periods_before_tear_down
        self._allowed_rpu_sizes: list[int] = sorted(
            allowed_rpu_sizes if allowed_rpu_sizes is not None else [8]
        )

        # Idle tracking: cluster_name → consecutive idle polls
        self._idle_counts: dict[str, int] = {}

        # Pressure flag: set by the router's capacity_pressure callback.
        self._pressure_event = threading.Event()

        # Control
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # External trigger (hooked into the router's callback)
    # ------------------------------------------------------------------

    def signal_capacity_pressure(self) -> None:
        """Called (from any thread) when the router cannot find a
        non-violating cluster.  Wakes the controller early."""
        self._pressure_event.set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("CapacityController already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="capacity-ctrl"
        )
        self._thread.start()
        logger.info(
            "CapacityController started (poll=%.1fs, η_crit=%.3f, "
            "idle_periods=%d).",
            self._poll_interval_s,
            self._eta_crit,
            self._idle_periods_before_tear_down,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the controller to stop and wait for the thread to join."""
        self._stop_event.set()
        self._pressure_event.set()  # wake if sleeping
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("CapacityController stopped.")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Parameters (tunable by Layer 3)
    # ------------------------------------------------------------------

    @property
    def eta_crit(self) -> float:
        return self._eta_crit

    @eta_crit.setter
    def eta_crit(self, value: float) -> None:
        self._eta_crit = value

    @property
    def idle_periods_before_tear_down(self) -> int:
        return self._idle_periods_before_tear_down

    @idle_periods_before_tear_down.setter
    def idle_periods_before_tear_down(self, value: int) -> None:
        self._idle_periods_before_tear_down = value

    @property
    def poll_interval_s(self) -> float:
        return self._poll_interval_s

    @poll_interval_s.setter
    def poll_interval_s(self, value: float) -> None:
        self._poll_interval_s = value

    @property
    def allowed_rpu_sizes(self) -> list[int]:
        return list(self._allowed_rpu_sizes)

    @allowed_rpu_sizes.setter
    def allowed_rpu_sizes(self, value: list[int]) -> None:
        self._allowed_rpu_sizes = sorted(value)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop executed in the background thread."""
        while not self._stop_event.is_set():
            # Sleep until the poll interval elapses *or* a pressure signal
            # arrives (whichever comes first).
            self._pressure_event.wait(timeout=self._poll_interval_s)
            pressure_fired = self._pressure_event.is_set()
            self._pressure_event.clear()

            if self._stop_event.is_set():
                break

            try:
                self._tick(pressure_fired)
            except Exception:
                logger.exception("CapacityController tick failed")

    def _tick(self, pressure_fired: bool) -> None:
        """One evaluation cycle."""
        active_map = self._get_active_queries()

        # -- Global headroom --------------------------------------------------
        all_queries: list[Query] = []
        for qs in active_map.values():
            all_queries.extend(qs)

        headroom = RoutingCore.compute_slo_headroom(
            all_queries, self._slo_resolver
        )
        logger.debug(
            "CapacityController tick: headroom=%.4f, pressure=%s, "
            "active_queries=%d",
            headroom,
            pressure_fired,
            len(all_queries),
        )

        # -- Spin-up decision ------------------------------------------------
        if headroom <= self._eta_crit or pressure_fired:
            reason = (
                f"headroom={headroom:.4f}<=η_crit={self._eta_crit:.4f}"
                if headroom <= self._eta_crit
                else "capacity_pressure_signal"
            )
            rpu = self._allowed_rpu_sizes[0]  # smallest available
            logger.info("Spin-up triggered: %s (rpu=%d)", reason, rpu)
            if self._on_spin_up is not None:
                try:
                    self._on_spin_up(reason, rpu)
                except Exception:
                    logger.exception("on_spin_up callback failed")

        # -- Per-cluster idle tracking / tear-down ---------------------------
        for cn, qs in active_map.items():
            if len(qs) == 0:
                self._idle_counts[cn] = self._idle_counts.get(cn, 0) + 1
                if (
                    self._idle_counts[cn]
                    >= self._idle_periods_before_tear_down
                ):
                    logger.info(
                        "Tear-down triggered for cluster %s (idle for %d "
                        "periods).",
                        cn,
                        self._idle_counts[cn],
                    )
                    if self._on_tear_down is not None:
                        try:
                            self._on_tear_down(cn)
                        except Exception:
                            logger.exception("on_tear_down callback failed")
                    # Reset so we don't re-fire every tick.
                    self._idle_counts[cn] = 0
            else:
                # Reset idle counter when the cluster has work.
                self._idle_counts[cn] = 0

    # ------------------------------------------------------------------
    # Synchronous single-tick (for testing / simulation)
    # ------------------------------------------------------------------

    def tick_once(self, pressure: bool = False) -> None:
        """Run exactly one evaluation cycle synchronously.

        This is useful in unit tests and in the simulator (where the
        simulation loop controls time and doesn't need a background
        thread).
        """
        self._tick(pressure)
