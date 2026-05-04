"""Tests for SloObjective.rank_indices() and SloObjective.is_sufficient_improvement()."""

from __future__ import annotations

import pytest

from autoslo.config.component_configs import SloObjectiveConfig
from autoslo.slo.slo_objective import SloObjective, ViolationCost


def _obj(threshold: float = 0.05) -> SloObjective:
    return SloObjective(
        SloObjectiveConfig(slo_metric="binary", slo_threshold=threshold)
    )


# ---------------------------------------------------------------------------
# rank_indices
# ---------------------------------------------------------------------------


class TestRankIndices:
    def test_single_candidate(self):
        obj = _obj()
        assert obj.rank_indices([ViolationCost(0.01, 10.0)]) == [0]

    def test_all_feasible_sorted_by_cost(self):
        """All candidates meet threshold → ranked by cost ascending."""
        obj = _obj(threshold=0.5)
        candidates = [
            ViolationCost(0.10, 80.0),  # idx 0: feasible, cost 80
            ViolationCost(0.20, 30.0),  # idx 1: feasible, cost 30 (cheapest)
            ViolationCost(0.05, 50.0),  # idx 2: feasible, cost 50
        ]
        ranked = obj.rank_indices(candidates)
        # cheapest first: 1 (30), 2 (50), 0 (80)
        assert ranked == [1, 2, 0]

    def test_all_infeasible_sorted_by_violation_then_cost(self):
        """No candidate meets threshold → sorted by violation asc, cost as tiebreak."""
        obj = _obj(threshold=0.01)
        candidates = [
            ViolationCost(0.10, 80.0),  # idx 0: viol=0.10
            ViolationCost(0.05, 30.0),  # idx 1: viol=0.05 (best viol)
            ViolationCost(0.05, 20.0),  # idx 2: viol=0.05, cheaper (tiebreak)
        ]
        ranked = obj.rank_indices(candidates)
        assert ranked == [2, 1, 0]

    def test_mixed_feasible_first(self):
        """Feasible candidates come before infeasible ones regardless of cost."""
        obj = _obj(threshold=0.05)
        candidates = [
            ViolationCost(0.10, 5.0),  # idx 0: infeasible, very cheap
            ViolationCost(0.03, 90.0),  # idx 1: feasible, expensive
            ViolationCost(0.20, 1.0),  # idx 2: infeasible, cheapest
            ViolationCost(0.04, 40.0),  # idx 3: feasible, cheaper
        ]
        ranked = obj.rank_indices(candidates)
        # Feasible: idx 3 (cost=40) < idx 1 (cost=90)
        # Infeasible: idx 0 (viol=0.10) < idx 2 (viol=0.20)
        assert ranked == [3, 1, 0, 2]

    def test_empty_candidates(self):
        obj = _obj()
        assert obj.rank_indices([]) == []

    @pytest.mark.parametrize(
        "violations, costs, threshold",
        [
            ([0.01, 0.10, 0.03], [50.0, 20.0, 80.0], 0.05),
            ([0.08, 0.04, 0.06], [10.0, 30.0, 20.0], 0.01),
            ([0.08, 0.06, 0.04], [10.0, 20.0, 30.0], 0.05),
            ([0.01, 0.02, 0.03], [30.0, 20.0, 10.0], 0.5),
        ],
    )
    def test_first_ranked_matches_idx_of_best(
        self,
        violations: list[float],
        costs: list[float],
        threshold: float,
    ):
        """rank_indices()[0] always agrees with idx_of_best()."""
        obj = SloObjective(
            SloObjectiveConfig(slo_metric="binary", slo_threshold=threshold)
        )
        candidates = [ViolationCost(v, c) for v, c in zip(violations, costs)]
        assert obj.rank_indices(candidates)[0] == obj.idx_of_best(candidates)


# ---------------------------------------------------------------------------
# is_sufficient_improvement
# ---------------------------------------------------------------------------


class TestIsSufficientImprovement:
    """
    The four primary cases map to the axes of the design doc:

        baseline feasible?  |  candidate feasible?
        --------------------|---------------------
               No           |        Yes           → boundary crossing
               No           |        No            → both infeasible
              Yes           |        Yes           → both feasible
              Yes           |        No            → impossible (cmp returns 1)

    Within each case we test: accept (margin >= epsilon), reject (margin <
    epsilon), and the edge cases at exactly epsilon.
    """

    THRESHOLD = 0.10  # violation threshold
    EPSILON = 0.05  # min_rel_improvement

    def _obj(self) -> SloObjective:
        return _obj(threshold=self.THRESHOLD)

    # ------------------------------------------------------------------
    # Case 1: boundary crossing — baseline infeasible, candidate feasible
    # ------------------------------------------------------------------

    def test_boundary_crossing_accepted_without_epsilon(self):
        """Crossing the feasibility boundary is always accepted regardless of
        how small the improvement is."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.20, cost=100.0)  # infeasible
        candidate = ViolationCost(
            violation=0.09, cost=200.0
        )  # feasible, costlier
        assert obj.is_sufficient_improvement(baseline, candidate, self.EPSILON)

    def test_boundary_crossing_accepted_tiny_improvement(self):
        """An improvement that crosses the boundary is accepted even when the
        margin is small, as long as it exceeds cmp's COMPARISON_TOLERANCE.
        (The epsilon waiver applies once cmp confirms the candidate is strictly
        better; noise within the tolerance band is not a strict improvement.)
        """
        obj = self._obj()
        baseline = ViolationCost(
            violation=0.10 + 1e-3, cost=50.0
        )  # barely infeasible
        candidate = ViolationCost(
            violation=0.10 - 1e-3, cost=50.0
        )  # barely feasible
        assert obj.is_sufficient_improvement(baseline, candidate, self.EPSILON)

    # ------------------------------------------------------------------
    # Case 2: both infeasible — decisive dimension is violation
    # ------------------------------------------------------------------

    def test_both_infeasible_accepts_when_violation_reduces_enough(self):
        """Both infeasible and candidate reduces violation by more than epsilon."""
        obj = self._obj()
        # baseline violation=0.40; 5 % epsilon means need at least 0.02 reduction
        baseline = ViolationCost(violation=0.40, cost=50.0)
        candidate = ViolationCost(violation=0.34, cost=50.0)  # 15 % improvement
        assert obj.is_sufficient_improvement(baseline, candidate, self.EPSILON)

    def test_both_infeasible_rejects_when_violation_improves_too_little(self):
        """Both infeasible and violation improvement is below epsilon."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.40, cost=50.0)
        candidate = ViolationCost(
            violation=0.39, cost=50.0
        )  # 2.5 % improvement < 5 %
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    def test_both_infeasible_rejects_when_violation_does_not_improve(self):
        """Both infeasible and candidate is worse — cmp returns 1, so False."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.40, cost=50.0)
        candidate = ViolationCost(violation=0.50, cost=50.0)  # worse
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    def test_both_infeasible_rejects_equal_violation(self):
        """Both infeasible and violations are equal — cmp returns 0 (tie), so False."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.40, cost=50.0)
        candidate = ViolationCost(
            violation=0.40, cost=30.0
        )  # same violation, cheaper
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    def test_both_infeasible_at_exactly_epsilon_boundary(self):
        """Both infeasible and improvement is exactly epsilon — should accept."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.40, cost=50.0)
        # exactly 5 % relative improvement: 0.40 * 0.05 = 0.02 reduction -> 0.38
        candidate = ViolationCost(
            violation=0.40 * (1 - self.EPSILON), cost=50.0
        )
        assert obj.is_sufficient_improvement(baseline, candidate, self.EPSILON)

    def test_both_infeasible_cost_improvement_alone_does_not_rescue_small_violation_gain(
        self,
    ):
        """Cost reduction alone cannot compensate for a violation improvement
        below epsilon when both configs are infeasible."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.40, cost=100.0)
        candidate = ViolationCost(
            violation=0.39, cost=10.0
        )  # cheap but tiny viol gain
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    # ------------------------------------------------------------------
    # Case 3: both feasible — decisive dimension is cost
    # ------------------------------------------------------------------

    def test_both_feasible_accepts_when_cost_reduces_enough(self):
        """Both feasible and candidate reduces cost by more than epsilon."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(
            violation=0.08, cost=90.0
        )  # 10 % cost reduction
        assert obj.is_sufficient_improvement(baseline, candidate, self.EPSILON)

    def test_both_feasible_rejects_when_cost_improves_too_little(self):
        """Both feasible and cost improvement is below epsilon."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(
            violation=0.05, cost=98.0
        )  # 2 % cost reduction < 5 %
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    def test_both_feasible_rejects_when_cost_does_not_improve(self):
        """Both feasible and candidate is costlier — cmp returns 1, so False."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(
            violation=0.01, cost=110.0
        )  # better violation, costlier
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    def test_both_feasible_rejects_equal_cost(self):
        """Both feasible and costs are equal — cmp returns 0 (tie), so False."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(violation=0.01, cost=100.0)
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    def test_both_feasible_at_exactly_epsilon_boundary(self):
        """Both feasible and cost improvement is exactly epsilon — should accept."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(
            violation=0.05, cost=100.0 * (1 - self.EPSILON)
        )
        assert obj.is_sufficient_improvement(baseline, candidate, self.EPSILON)

    def test_both_feasible_violation_improvement_alone_does_not_accept(self):
        """THIS IS THE CORE BUG CASE from the design doc.

        When both configs are feasible the comparison must be purely by cost.
        A spin-up that reduces violation (good) but does NOT reduce cost must
        be rejected, because adding a cluster that does not save money is
        wasteful in the already-feasible regime.
        """
        obj = self._obj()
        # Both feasible (violations 0.05 and 0.01 both below threshold 0.10).
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(
            violation=0.01, cost=105.0
        )  # better SLO, costlier
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    def test_both_feasible_zero_violation_baseline_accepts_cost_reduction(self):
        """The original bug: baseline violation==0 must not hard-code improvement=0.

        A candidate that costs less while keeping violation at 0 must be accepted.
        """
        obj = self._obj()
        baseline = ViolationCost(violation=0.0, cost=100.0)
        candidate = ViolationCost(violation=0.0, cost=80.0)  # 20 % cheaper
        assert obj.is_sufficient_improvement(baseline, candidate, self.EPSILON)

    def test_both_feasible_zero_violation_baseline_rejects_negligible_cost_reduction(
        self,
    ):
        """Zero violation baseline still enforces the epsilon cost threshold."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.0, cost=100.0)
        candidate = ViolationCost(
            violation=0.0, cost=99.0
        )  # 1 % cheaper, below epsilon
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    # ------------------------------------------------------------------
    # Case 4: impossible — baseline feasible, candidate infeasible
    # ------------------------------------------------------------------

    def test_baseline_feasible_candidate_infeasible_always_rejects(self):
        """A regression from feasible to infeasible is always rejected.

        cmp() returns 1 in this case (baseline is better), so the first
        check inside is_sufficient_improvement short-circuits to False.
        """
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)  # feasible
        candidate = ViolationCost(
            violation=0.50, cost=10.0
        )  # infeasible, very cheap
        assert not obj.is_sufficient_improvement(
            baseline, candidate, self.EPSILON
        )

    # ------------------------------------------------------------------
    # epsilon=0 edge cases
    # ------------------------------------------------------------------

    def test_epsilon_zero_accepts_any_strict_improvement(self):
        """With epsilon=0 any strict improvement (cmp == -1) is accepted."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(violation=0.05, cost=99.99)  # tiny cost win
        assert obj.is_sufficient_improvement(
            baseline, candidate, min_rel_improvement=0.0
        )

    def test_epsilon_zero_still_rejects_ties(self):
        """With epsilon=0, a tie (cmp == 0) is still rejected."""
        obj = self._obj()
        baseline = ViolationCost(violation=0.05, cost=100.0)
        candidate = ViolationCost(violation=0.05, cost=100.0)
        assert not obj.is_sufficient_improvement(
            baseline, candidate, min_rel_improvement=0.0
        )
