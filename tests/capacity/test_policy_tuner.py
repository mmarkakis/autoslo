"""
Tests for :mod:`autoslo.capacity.policy_tuner`.

All tests use a trivial mock ``simulate_fn`` so they run without the
heavyweight simulator or any model artefacts.
"""

from __future__ import annotations

import io

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend for testing

from autoslo.capacity.policy_tuner import (
    VALID_OBJECTIVES,
    PolicyParams,
    PolicyTuner,
    ScenarioOutcome,
    SweepEntry,
    SweepResult,
    compute_pareto_front,
    plot_pareto_front,
    print_pareto_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _outcome(
    cost: float,
    viol: float,
    amount: float = 0.0,
    rel_viol: float = 0.0,
) -> ScenarioOutcome:
    return ScenarioOutcome(
        mean_cost=cost,
        mean_violation_rate=viol,
        mean_violation_amount_s=amount,
        mean_relative_violation=rel_viol,
    )


def _entry(
    cost: float,
    viol: float,
    amount: float = 0.0,
    rel_viol: float = 0.0,
) -> SweepEntry:
    params = PolicyParams(eta_crit=0.1, idle_periods_before_tear_down=5)
    return SweepEntry(
        params=params,
        outcome=_outcome(cost, viol, amount, rel_viol),
    )


# ---------------------------------------------------------------------------
# Tests: PolicyParams / make_grid
# ---------------------------------------------------------------------------


class TestPolicyParams:

    def test_frozen(self):
        p = PolicyParams(eta_crit=0.1, idle_periods_before_tear_down=5)
        with pytest.raises(AttributeError):
            p.eta_crit = 0.2  # type: ignore[misc]

    def test_equality(self):
        a = PolicyParams(0.1, 5)
        b = PolicyParams(0.1, 5)
        assert a == b

    def test_hashable(self):
        p = PolicyParams(0.1, 5)
        assert hash(p) == hash(PolicyParams(0.1, 5))


class TestMakeGrid:

    def test_single_point(self):
        grid = PolicyTuner.make_grid(eta_crit=[0.1], idle_periods=[5])
        assert len(grid) == 1
        assert grid[0] == PolicyParams(0.1, 5)

    def test_cartesian_product(self):
        eta_options = [0.1, 0.2]
        idle_options = [3, 5]
        grid = PolicyTuner.make_grid(
            eta_crit=eta_options,
            idle_periods=idle_options,
        )
        assert len(grid) == len(eta_options) * len(idle_options)

    def test_ordering(self):
        """First axis varies slowest, last axis fastest."""
        eta_options = [0.1, 0.2]
        idle_options = [3, 5]
        grid = PolicyTuner.make_grid(
            eta_crit=eta_options,
            idle_periods=idle_options,
        )
        assert grid[0].eta_crit == 0.1
        assert grid[0].idle_periods_before_tear_down == 3
        assert grid[1].eta_crit == 0.1
        assert grid[1].idle_periods_before_tear_down == 5
        assert grid[2].eta_crit == 0.2
        assert grid[2].idle_periods_before_tear_down == 3


# ---------------------------------------------------------------------------
# Tests: compute_pareto_front (default objectives: cost vs violation_rate)
# ---------------------------------------------------------------------------


class TestParetoFront:

    def test_trivial_single_point(self):
        """A single point is always Pareto-optimal."""
        entries = [_entry(10, 0.5)]
        front = compute_pareto_front(entries)
        assert len(front) == 1
        assert front[0].is_pareto

    def test_two_nondominated(self):
        """Two points that trade off cost vs violation — both Pareto."""
        entries = [
            _entry(cost=10, viol=0.5),
            _entry(cost=5, viol=1.0),
        ]
        front = compute_pareto_front(entries)
        assert len(front) == 2
        assert all(e.is_pareto for e in front)

    def test_one_dominates_another(self):
        """Point A dominates point B (lower on both axes)."""
        entries = [
            _entry(cost=10, viol=0.5),
            _entry(cost=20, viol=1.0),
        ]
        front = compute_pareto_front(entries)
        assert len(front) == 1
        assert front[0].outcome.mean_cost == 10

    def test_three_points_with_interior(self):
        """Classical 3-point test: two on front, one interior."""
        entries = [
            _entry(cost=10, viol=0.5),  # front
            _entry(cost=20, viol=1.0),  # dominated
            _entry(cost=5, viol=1.0),   # front
        ]
        front = compute_pareto_front(entries)
        assert len(front) == 2
        front_costs = {e.outcome.mean_cost for e in front}
        assert front_costs == {10, 5}

    def test_identical_points(self):
        """Two identical points — neither dominates the other → both Pareto."""
        entries = [_entry(10, 0.5), _entry(10, 0.5)]
        front = compute_pareto_front(entries)
        assert len(front) == 2

    def test_front_sorted_by_primary(self):
        """Pareto front is sorted by ascending primary objective."""
        entries = [
            _entry(cost=10, viol=0.5),
            _entry(cost=20, viol=0.25),
            _entry(cost=5, viol=1.0),
        ]
        front = compute_pareto_front(entries)
        costs = [e.outcome.mean_cost for e in front]
        assert costs == sorted(costs)

    def test_empty_list(self):
        assert compute_pareto_front([]) == []

    def test_all_dominated_by_one(self):
        """One point dominates all others."""
        entries = [
            _entry(cost=10, viol=0.5),
            _entry(cost=20, viol=1.0),
            _entry(cost=30, viol=1.5),
        ]
        front = compute_pareto_front(entries)
        assert len(front) == 1
        assert front[0].outcome.mean_cost == 10


# ---------------------------------------------------------------------------
# Tests: custom objectives
# ---------------------------------------------------------------------------


class TestCustomObjectives:

    def test_cost_vs_violation_amount(self):
        """Pareto front using cost vs violation_amount_s."""
        entries = [
            _entry(cost=10, viol=0.5, amount=100),
            _entry(cost=5, viol=0.8, amount=200),
            _entry(cost=20, viol=0.1, amount=50),
        ]
        front = compute_pareto_front(
            entries,
            objective_x="mean_cost",
            objective_y="mean_violation_amount_s",
        )
        # (5, 200), (10, 100), (20, 50) — all trade off cost vs amount
        assert len(front) == 3

    def test_cost_vs_relative_violation(self):
        """Pareto front using cost vs relative_violation."""
        entries = [
            _entry(cost=10, viol=0.5, rel_viol=0.3),
            _entry(cost=5, viol=0.8, rel_viol=0.8),
            _entry(cost=20, viol=0.1, rel_viol=0.1),
        ]
        front = compute_pareto_front(
            entries,
            objective_x="mean_cost",
            objective_y="mean_relative_violation",
        )
        assert len(front) == 3

    def test_violation_rate_vs_cost(self):
        """Swapping x and y axes works."""
        entries = [
            _entry(cost=10, viol=0.5),
            _entry(cost=5, viol=1.0),
        ]
        front = compute_pareto_front(
            entries,
            objective_x="mean_violation_rate",
            objective_y="mean_cost",
        )
        assert len(front) == 2
        # Sorted by violation rate.
        viols = [e.outcome.mean_violation_rate for e in front]
        assert viols == sorted(viols)

    def test_invalid_objective_x_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            compute_pareto_front([], objective_x="bad_name")

    def test_invalid_objective_y_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            compute_pareto_front([], objective_y="bad_name")

    def test_sweep_with_custom_objectives(self):
        """sweep() passes custom objectives through."""
        def sim(p):
            return _outcome(cost=p.eta_crit * 100, viol=1 - p.eta_crit,
                            amount=p.eta_crit * 50)

        grid = PolicyTuner.make_grid(eta_crit=[0.1, 0.5], idle_periods=[5])
        tuner = PolicyTuner(grid, simulate_fn=sim)
        result = tuner.sweep(
            objective_x="mean_cost",
            objective_y="mean_violation_amount_s",
        )
        assert result.objective_x == "mean_cost"
        assert result.objective_y == "mean_violation_amount_s"
        assert len(result.pareto_front) > 0


# ---------------------------------------------------------------------------
# Tests: ScenarioOutcome — relative violation
# ---------------------------------------------------------------------------


class TestRelativeViolation:

    def test_relative_violation_stored(self):
        o = ScenarioOutcome(
            mean_cost=10,
            mean_violation_rate=0.5,
            mean_relative_violation=0.42,
        )
        assert o.mean_relative_violation == 0.42

    def test_default_zero(self):
        o = ScenarioOutcome(mean_cost=10, mean_violation_rate=0.5)
        assert o.mean_relative_violation == 0.0

    def test_per_scenario_relative_violations(self):
        o = ScenarioOutcome(
            mean_cost=10,
            mean_violation_rate=0.5,
            per_scenario_relative_violations=[0.1, 0.3, 0.5],
        )
        assert o.per_scenario_relative_violations == [0.1, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Tests: PolicyTuner.sweep
# ---------------------------------------------------------------------------


class TestPolicyTunerSweep:

    def test_sweep_calls_simulate_fn_for_each_point(self):
        """simulate_fn is called once per grid point."""
        call_log: list[PolicyParams] = []

        def sim(p: PolicyParams) -> ScenarioOutcome:
            call_log.append(p)
            return _outcome(cost=p.eta_crit * 100, viol=1 - p.eta_crit)

        grid = PolicyTuner.make_grid(
            eta_crit=[0.1, 0.2, 0.3],
            idle_periods=[5],
        )
        tuner = PolicyTuner(grid, simulate_fn=sim)
        result = tuner.sweep()

        assert len(call_log) == len(grid)
        assert len(result.entries) == len(grid)

    def test_sweep_result_has_pareto_front(self):
        """Sweep result includes a non-empty Pareto front."""

        def sim(p: PolicyParams) -> ScenarioOutcome:
            return _outcome(cost=(1 - p.eta_crit) * 100, viol=p.eta_crit)

        grid = PolicyTuner.make_grid(
            eta_crit=[0.1, 0.5, 0.9],
            idle_periods=[5],
        )
        tuner = PolicyTuner(grid, simulate_fn=sim)
        result = tuner.sweep()

        assert len(result.pareto_front) > 0
        assert len(result.pareto_front) == len(grid)

    def test_sweep_single_point(self):
        """Single-point grid works correctly."""

        def sim(p: PolicyParams) -> ScenarioOutcome:
            return _outcome(cost=42.0, viol=0.05)

        grid = [PolicyParams(0.1, 5)]
        tuner = PolicyTuner(grid, simulate_fn=sim)
        result = tuner.sweep()

        assert len(result.entries) == 1
        assert len(result.pareto_front) == 1
        assert result.pareto_front[0].outcome.mean_cost == 42.0

    def test_sweep_preserves_per_scenario_details(self):
        """Per-scenario details are passed through."""

        def sim(p: PolicyParams) -> ScenarioOutcome:
            return ScenarioOutcome(
                mean_cost=10.0,
                mean_violation_rate=0.1,
                per_scenario_costs=[8.0, 12.0],
                per_scenario_violation_amounts_s=[0.5, 1.5],
            )

        grid = [PolicyParams(0.1, 5)]
        tuner = PolicyTuner(grid, simulate_fn=sim)
        result = tuner.sweep()

        entry = result.entries[0]
        assert entry.outcome.per_scenario_costs == [8.0, 12.0]
        assert entry.outcome.per_scenario_violation_amounts_s == [0.5, 1.5]

    def test_sweep_with_dominated_points(self):
        """Grid with dominated interior points produces correct front."""
        outcomes = {
            0.1: _outcome(cost=5, viol=0.9),
            0.2: _outcome(cost=15, viol=0.5),
            0.3: _outcome(cost=10, viol=0.2),
        }

        def sim(p: PolicyParams) -> ScenarioOutcome:
            return outcomes[p.eta_crit]

        grid = PolicyTuner.make_grid(
            eta_crit=[0.1, 0.2, 0.3],
            idle_periods=[5],
        )
        tuner = PolicyTuner(grid, simulate_fn=sim)
        result = tuner.sweep()

        front_costs = {e.outcome.mean_cost for e in result.pareto_front}
        # (15, 0.5) dominated by (10, 0.2)
        assert 15 not in front_costs
        assert {5, 10} == front_costs


# ---------------------------------------------------------------------------
# Tests: plotting / rich summary
# ---------------------------------------------------------------------------


class TestPlotAndSummary:

    def _make_result(self) -> SweepResult:
        """Build a small SweepResult for plotting tests."""
        entries = [
            _entry(cost=5, viol=0.9),
            _entry(cost=10, viol=0.2),
            _entry(cost=15, viol=0.5),
        ]
        front = compute_pareto_front(entries)
        return SweepResult(
            entries=entries,
            pareto_front=front,
            objective_x="mean_cost",
            objective_y="mean_violation_rate",
        )

    def test_plot_returns_axes(self):
        result = self._make_result()
        ax = plot_pareto_front(result, show=False)
        assert ax is not None

    def test_plot_on_existing_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = self._make_result()
        returned_ax = plot_pareto_front(result, ax=ax, show=False)
        assert returned_ax is ax
        plt.close(fig)

    def test_plot_empty_result(self):
        result = SweepResult()
        ax = plot_pareto_front(result, show=False)
        assert ax is not None

    def test_print_summary_runs(self, capsys):
        """print_pareto_summary produces output without error."""
        result = self._make_result()
        print_pareto_summary(result)
        captured = capsys.readouterr()
        # Rich output goes to its own console; plain-text goes to stdout.
        # Either way, the function should not raise.

    def test_print_summary_empty(self, capsys):
        result = SweepResult()
        print_pareto_summary(result)
        # Should not raise.
