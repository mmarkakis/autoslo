"""
policy_tuner.py
===============
Simulation-based policy tuning (Layer 3b).

Sweeps a grid of policy parameters (``eta_crit``,
``idle_periods_before_tear_down``) by running a user-supplied simulation
callable for each combination, then identifies the Pareto-optimal set of
trade-offs between two user-chosen objectives.

The tuner is agnostic to *how* the simulation is executed — callers supply a
function ``simulate_fn(params: PolicyParams) → ScenarioOutcome`` that runs one
(or many) scenario(s) and returns aggregate cost / violation statistics.
This keeps the tuner free of heavyweight simulator dependencies and makes it
trivially testable.

Typical usage
-------------
>>> from autoslo.capacity.policy_tuner import (
...     PolicyTuner, PolicyParams, ScenarioOutcome,
... )
>>>
>>> def run(p: PolicyParams) -> ScenarioOutcome:
...     # ... run WorkloadRoutingSimulator with p ...
...     return ScenarioOutcome(mean_cost=..., mean_violation_rate=...,
...                            mean_violation_amount_s=...,
...                            mean_relative_violation=...)
>>>
>>> grid = PolicyTuner.make_grid(
...     eta_crit=[0.1, 0.2, 0.3],
...     idle_periods=[2, 5, 10],
... )
>>> tuner = PolicyTuner(grid, simulate_fn=run)
>>> result = tuner.sweep()
>>> print(result.pareto_front)
>>>
>>> # Visualise
>>> from autoslo.capacity.policy_tuner import plot_pareto_front
>>> plot_pareto_front(result)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt

try:
    from rich.console import Console
    from rich.table import Table as RichTable

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyParams:
    """A single point in the policy-parameter space.

    Attributes
    ----------
    eta_crit : float
        SLO headroom threshold for spin-up (Layer 2).
    idle_periods_before_tear_down : int
        Consecutive idle polls before a cluster is torn down (Layer 2).
    """

    eta_crit: float
    idle_periods_before_tear_down: int


@dataclass
class ScenarioOutcome:
    """Aggregate statistics from running one parameter combination across
    one or more forecast scenarios.

    Attributes
    ----------
    mean_cost : float
        Mean total billing cost across scenarios (USD).
    mean_violation_rate : float
        Mean fraction of queries violating their SLO.
    mean_violation_amount_s : float
        Mean total SLO violation seconds across scenarios.
    mean_relative_violation : float
        Mean relative violation across scenarios, defined as the average
        of ``(observed_latency - slo) / slo`` over violating queries.
        Zero when there are no violations.
    per_scenario_costs : list[float]
        Per-scenario costs (optional detail; may be empty).
    per_scenario_violation_rates : list[float]
        Per-scenario violation rates (optional detail; may be empty).
    per_scenario_violation_amounts_s : list[float]
        Per-scenario violation amounts (optional detail; may be empty).
    per_scenario_relative_violations : list[float]
        Per-scenario relative violations (optional detail; may be empty).
    """

    mean_cost: float
    mean_violation_rate: float
    mean_violation_amount_s: float = 0.0
    mean_relative_violation: float = 0.0
    per_scenario_costs: list[float] = field(default_factory=list)
    per_scenario_violation_rates: list[float] = field(default_factory=list)
    per_scenario_violation_amounts_s: list[float] = field(default_factory=list)
    per_scenario_relative_violations: list[float] = field(default_factory=list)


@dataclass
class SweepEntry:
    """One row in the sweep results table.

    Attributes
    ----------
    params : PolicyParams
        The parameter combination that was evaluated.
    outcome : ScenarioOutcome
        Aggregate simulation results for that combination.
    is_pareto : bool
        Whether this point is on the Pareto frontier.
    """

    params: PolicyParams
    outcome: ScenarioOutcome
    is_pareto: bool = False


@dataclass
class SweepResult:
    """Full output of :meth:`PolicyTuner.sweep`.

    Attributes
    ----------
    entries : list[SweepEntry]
        All evaluated points, in grid order.
    pareto_front : list[SweepEntry]
        Subset of entries on the Pareto frontier, sorted by ascending
        primary objective.
    objective_x : str
        Name of the primary (x-axis) objective used for the front.
    objective_y : str
        Name of the secondary (y-axis) objective used for the front.
    """

    entries: list[SweepEntry] = field(default_factory=list)
    pareto_front: list[SweepEntry] = field(default_factory=list)
    objective_x: str = "mean_cost"
    objective_y: str = "mean_violation_rate"


# ---------------------------------------------------------------------------
# Objective accessor
# ---------------------------------------------------------------------------

#: Valid objective names that can be extracted from a ScenarioOutcome.
VALID_OBJECTIVES = frozenset(
    {
        "mean_cost",
        "mean_violation_rate",
        "mean_violation_amount_s",
        "mean_relative_violation",
    }
)


def _get_objective(entry: SweepEntry, name: str) -> float:
    """Extract an objective value from a SweepEntry by name."""
    return getattr(entry.outcome, name)


# ---------------------------------------------------------------------------
# Pareto front — O(N log N) via sorting
# ---------------------------------------------------------------------------


def compute_pareto_front(
    entries: list[SweepEntry],
    objective_x: str = "mean_cost",
    objective_y: str = "mean_violation_rate",
) -> list[SweepEntry]:
    """Mark and return Pareto-optimal entries.

    A point *dominates* another if it is ≤ on **both** objectives and
    strictly < on at least one.  The Pareto front is the set of
    non-dominated points.

    This implementation runs in O(N log N) by sorting on the primary
    objective and sweeping the secondary.

    Parameters
    ----------
    entries : list[SweepEntry]
        The entries to evaluate.
    objective_x : str
        Primary objective name (an attribute of ``ScenarioOutcome``).
        Both objectives are *minimised*.
    objective_y : str
        Secondary objective name.

    Returns
    -------
    list[SweepEntry]
        The non-dominated entries, sorted by ascending ``objective_x``.
    """
    if objective_x not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{objective_x}'. "
            f"Choose from {sorted(VALID_OBJECTIVES)}."
        )
    if objective_y not in VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{objective_y}'. "
            f"Choose from {sorted(VALID_OBJECTIVES)}."
        )

    n = len(entries)
    if n == 0:
        return []

    # Sort by primary ascending, break ties by secondary ascending.
    indexed = sorted(
        range(n),
        key=lambda i: (
            _get_objective(entries[i], objective_x),
            _get_objective(entries[i], objective_y),
        ),
    )

    front_indices: list[int] = []
    best_y = float("inf")

    for idx in indexed:
        y = _get_objective(entries[idx], objective_y)
        if y <= best_y:
            front_indices.append(idx)
            best_y = y

    # Mark entries.
    pareto_set = set(front_indices)
    for i, entry in enumerate(entries):
        entry.is_pareto = i in pareto_set

    # Return front sorted by primary objective.
    front = [entries[i] for i in front_indices]
    return front


# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------


class PolicyTuner:
    """Grid-sweep policy tuner with Pareto-front identification.

    Parameters
    ----------
    grid : Iterable[PolicyParams]
        The parameter combinations to evaluate (use :meth:`make_grid`
        for convenience).
    simulate_fn : Callable[[PolicyParams], ScenarioOutcome]
        A function that, given a parameter combination, runs the
        simulation(s) and returns aggregate statistics.
    """

    def __init__(
        self,
        grid: Iterable[PolicyParams],
        simulate_fn: Callable[[PolicyParams], ScenarioOutcome],
    ) -> None:
        self._grid = list(grid)
        self._simulate_fn = simulate_fn

    # ------------------------------------------------------------------
    # Grid factory
    # ------------------------------------------------------------------

    @staticmethod
    def make_grid(
        eta_crit: Sequence[float],
        idle_periods: Sequence[int],
    ) -> list[PolicyParams]:
        """Build the Cartesian product of parameter values.

        Parameters
        ----------
        eta_crit : sequence of float
            Values of η_crit to try.
        idle_periods : sequence of int
            Values of ``idle_periods_before_tear_down`` to try.

        Returns
        -------
        list[PolicyParams]
        """
        return [
            PolicyParams(
                eta_crit=e,
                idle_periods_before_tear_down=i,
            )
            for e, i in itertools.product(eta_crit, idle_periods)
        ]

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    def sweep(
        self,
        objective_x: str = "mean_cost",
        objective_y: str = "mean_violation_rate",
    ) -> SweepResult:
        """Evaluate every point in the grid and compute the Pareto front.

        Parameters
        ----------
        objective_x : str
            Primary objective for the Pareto front (minimised).
        objective_y : str
            Secondary objective for the Pareto front (minimised).

        Returns
        -------
        SweepResult
        """
        entries: list[SweepEntry] = []
        total = len(self._grid)

        for idx, params in enumerate(self._grid):
            logger.info(
                "PolicyTuner: evaluating %d / %d — η_crit=%.3f, " "L_down=%d",
                idx + 1,
                total,
                params.eta_crit,
                params.idle_periods_before_tear_down,
            )
            outcome = self._simulate_fn(params)
            entries.append(SweepEntry(params=params, outcome=outcome))

        front = compute_pareto_front(
            entries, objective_x=objective_x, objective_y=objective_y
        )

        logger.info(
            "PolicyTuner: sweep complete — %d points evaluated, "
            "%d on Pareto front.",
            len(entries),
            len(front),
        )
        return SweepResult(
            entries=entries,
            pareto_front=front,
            objective_x=objective_x,
            objective_y=objective_y,
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "mean_cost": "Mean Cost (USD)",
    "mean_violation_rate": "Mean Violation Rate",
    "mean_violation_amount_s": "Mean Violation Amount (s)",
    "mean_relative_violation": "Mean Relative Violation",
}


def plot_pareto_front(
    result: SweepResult,
    *,
    ax: matplotlib.axes.Axes | None = None,
    show: bool = True,
) -> matplotlib.axes.Axes:
    """Plot all sweep points and highlight the Pareto front.

    Parameters
    ----------
    result : SweepResult
        Output of :meth:`PolicyTuner.sweep`.
    ax : matplotlib Axes, optional
        If provided, draw on this axes object.
    show : bool
        Whether to call ``plt.show()`` after plotting.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    obj_x = result.objective_x
    obj_y = result.objective_y

    # Non-Pareto points.
    non_pareto = [e for e in result.entries if not e.is_pareto]
    if non_pareto:
        ax.scatter(
            [_get_objective(e, obj_x) for e in non_pareto],
            [_get_objective(e, obj_y) for e in non_pareto],
            marker="o",
            color="silver",
            edgecolors="grey",
            alpha=0.6,
            label="Dominated",
            zorder=2,
        )

    # Pareto front.
    if result.pareto_front:
        xs = [_get_objective(e, obj_x) for e in result.pareto_front]
        ys = [_get_objective(e, obj_y) for e in result.pareto_front]
        ax.plot(xs, ys, "o-", color="tab:blue", label="Pareto Front", zorder=3)

    ax.set_xlabel(_LABEL_MAP.get(obj_x, obj_x))
    ax.set_ylabel(_LABEL_MAP.get(obj_y, obj_y))
    ax.set_title("Policy Tuner — Pareto Front")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:  # pragma: no cover
        plt.tight_layout()
        plt.show()

    return ax


def print_pareto_summary(result: SweepResult) -> None:
    """Print a rich table summarising the Pareto front.

    Falls back to plain ``print()`` if ``rich`` is not installed.
    """
    obj_x = result.objective_x
    obj_y = result.objective_y

    if _HAS_RICH:
        console = Console()
        table = RichTable(
            title="Pareto Front Summary",
            show_lines=True,
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("η_crit", justify="right")
        table.add_column("L_down", justify="right")
        table.add_column(_LABEL_MAP.get(obj_x, obj_x), justify="right")
        table.add_column(_LABEL_MAP.get(obj_y, obj_y), justify="right")

        for i, entry in enumerate(result.pareto_front, 1):
            table.add_row(
                str(i),
                f"{entry.params.eta_crit:.3f}",
                str(entry.params.idle_periods_before_tear_down),
                f"{_get_objective(entry, obj_x):.4f}",
                f"{_get_objective(entry, obj_y):.4f}",
            )

        console.print()
        console.print(table)
        console.print(
            f"\n[dim]{len(result.entries)} total points evaluated, "
            f"{len(result.pareto_front)} on Pareto front.[/dim]\n"
        )
    else:
        # Plain-text fallback.
        header = (
            f"{'#':>3}  {'η_crit':>8}  {'L_down':>6}  "
            f"{obj_x:>20}  {obj_y:>20}"
        )
        print("\nPareto Front Summary")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for i, entry in enumerate(result.pareto_front, 1):
            print(
                f"{i:>3}  {entry.params.eta_crit:>8.3f}  "
                f"{entry.params.idle_periods_before_tear_down:>6}  "
                f"{_get_objective(entry, obj_x):>20.4f}  "
                f"{_get_objective(entry, obj_y):>20.4f}"
            )
        print(
            f"\n{len(result.entries)} total points evaluated, "
            f"{len(result.pareto_front)} on Pareto front.\n"
        )
