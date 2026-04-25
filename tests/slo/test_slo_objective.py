"""Tests for SloObjective.rank_indices()."""

from __future__ import annotations

import pytest

from autoslo.slo.slo_objective import SloObjective, ViolationCost


def _obj(threshold: float = 0.05) -> SloObjective:
    return SloObjective(slo_metric="binary", slo_threshold=threshold)


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
        obj = SloObjective(slo_metric="binary", slo_threshold=threshold)
        candidates = [ViolationCost(v, c) for v, c in zip(violations, costs)]
        assert obj.rank_indices(candidates)[0] == obj.idx_of_best(candidates)
