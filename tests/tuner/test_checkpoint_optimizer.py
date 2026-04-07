"""Tests for CheckpointOptimizer and find_violation_windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoslo.blueprint_selection.slo_resolver import SloResolver
from autoslo.tuner.tuner_utils import SloObjective
from autoslo.capacity.autoscaling_policy import CapacityCheckpoint
from autoslo.tuner.checkpoint_optimizer import (
    new_checkpoints_to_config,
    find_next_checkpoint_time_df,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_completion_structured_log(
    start_times: list[float],
    end_times: list[float],
) -> pd.DataFrame:
    """
    Helper to create a structured log DataFrame from lists of start and end
    times.
    """
    return pd.DataFrame(
        {
            "timestamp": end_times,
            "event_type": "completion",
            "query_id": [f"q{i}" for i in range(len(start_times))],
            "query_text_id": [f"schema#{i}#1" for i in range(len(start_times))],
            "latency_s": [
                end_times[i] - start_times[i] for i in range(len(start_times))
            ],
        }
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestNewCheckpointsToConfig:
    def test_no_checkpoints(self):
        """
        If no checkpoints, overrides should be empty.
        """

        initial_config = {
            "autoscaling_config": {"autoscaling_policy": "headroom"},
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
        }
        new_config = new_checkpoints_to_config(
            config=initial_config, new_checkpoints=[]
        )
        assert (
            new_config == initial_config
        ), "Expected no change to config when no new checkpoints"

    def test_single_nonzero_checkpoint(self):
        """
        For a single checkpoint with nonzero min_rpus, overrides should set
        headroom_min_rpus to that value.
        """

        initial_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
            "autoscaling_config": {"autoscaling_policy": "headroom"},
        }
        checkpoints = [
            CapacityCheckpoint(
                rel_time_s=100,
                min_rpus=(2, 4),
            )
        ]
        new_config = new_checkpoints_to_config(
            config=initial_config, new_checkpoints=checkpoints
        )
        expected_new_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
            "autoscaling_config": {
                "autoscaling_policy": "headroom",
                "capacity_checkpoints": [
                    {
                        "rel_time_s": 100,
                        "min_rpus": [2, 4],
                    }
                ],
            },
        }
        assert (
            new_config == expected_new_config
        ), "Expected capacity_checkpoints override based on nonzero-time checkpoint"

    def test_single_zero_checkpoint(self):
        """
        For a single checkpoint with rel_time_s=0, overrides should add its min_rpus
        to initial_rpus.
        """

        initial_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
            "autoscaling_config": {"autoscaling_policy": "headroom"},
        }
        checkpoints = [
            CapacityCheckpoint(
                rel_time_s=0,
                min_rpus=(2, 4),
            )
        ]
        new_config = new_checkpoints_to_config(
            config=initial_config, new_checkpoints=checkpoints
        )
        expected_new_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 2, 4, 4]},
            "autoscaling_config": {
                "autoscaling_policy": "headroom",
            },
        }
        assert (
            new_config == expected_new_config
        ), "Expected initial_rpus override based on zero-time checkpoint"

    def test_mixed_checkpoints(self):
        """
        For multiple checkpoints with a mix of zero and nonzero rel_time_s, overrides
        should correctly set both initial_rpus and capacity_checkpoints.
        """

        initial_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
            "autoscaling_config": {"autoscaling_policy": "headroom"},
        }
        checkpoints = [
            CapacityCheckpoint(
                rel_time_s=0,
                min_rpus=(2, 4),
            ),
            CapacityCheckpoint(
                rel_time_s=100,
                min_rpus=(4, 8),
            ),
        ]
        new_config = new_checkpoints_to_config(
            config=initial_config, new_checkpoints=checkpoints
        )
        expected_new_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 2, 4, 4]},
            "autoscaling_config": {
                "autoscaling_policy": "headroom",
                "capacity_checkpoints": [
                    {
                        "rel_time_s": 100,
                        "min_rpus": [4, 8],
                    }
                ],
            },
        }
        assert (
            new_config == expected_new_config
        ), "Expected overrides based on mixed zero and nonzero rel_time_s checkpoints"


class TestFindNextCheckpointTimeDF:
    def test_one_log_no_violations(self):
        """
        For a single structured log with no violations, return None.
        """
        slo_s = 0.5
        slo_resolver = SloResolver(default_slo_s=slo_s)
        start_times = [0, 10, 20]
        end_times = [s + np.random.uniform(0, slo_s * 0.9) for s in start_times]
        log = _make_completion_structured_log(start_times, end_times)

        slo_objective = SloObjective(slo_metric="binary", slo_threshold=0.0)

        result = find_next_checkpoint_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            spin_up_delay_s=5.0,
        )
        assert (
            result is None
        ), "Expected no checkpoint time when there are no violations"

    def test_one_log_insufficient_violations(self):
        """
        For a single structured log with some violations but below the threshold,
        return None.
        """
        slo_s = 2
        slo_resolver = SloResolver(default_slo_s=slo_s)

        # The first query violates. But at any given point, there is at least one
        # more query running that doesn't,
        start_times = [10, 10, 10.5, 11, 11.5, 12, 12.5, 13]
        end_times = [13, 11, 11.5, 12, 12.5, 13, 13.5, 14]
        log = _make_completion_structured_log(start_times, end_times)

        slo_objective = SloObjective(slo_metric="binary", slo_threshold=0.6)

        result = find_next_checkpoint_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            spin_up_delay_s=5.0,
        )
        assert (
            result is None
        ), "Expected no checkpoint time when violation rate is below threshold"

    @pytest.mark.parametrize("seed", range(10))
    def test_one_log_one_violating_period(self, seed):
        """
        For a single structured log with a clear violating period above the threshold,
        return the start time of that period.
        """
        np.random.seed(seed)
        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        start_times = [10]
        end_times = [12]
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(0, 1)
        spin_up_delay_s = np.random.uniform(0, 10)
        slo_objective = SloObjective(
            slo_metric="binary", slo_threshold=slo_threshold
        )
        result = find_next_checkpoint_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            spin_up_delay_s=spin_up_delay_s,
        )
        assert (
            result == 10 - spin_up_delay_s
        ), "Expected checkpoint time to be the start of the violating period minus spin-up delay"

    @pytest.mark.parametrize("seed", range(10))
    def test_one_log_bottom_at_zero(self, seed):
        """
        For a single structured log with a clear violating period above the threshold,
        don't return a negative checkpoint time if the violating period starts near
        time 0.
        """
        np.random.seed(seed)
        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        start_times = [np.random.uniform(0, 5)]
        end_times = [start_times[0] + 2]
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(0, 1)
        spin_up_delay_s = np.random.uniform(5, 10)
        slo_objective = SloObjective(
            slo_metric="binary", slo_threshold=slo_threshold
        )
        result = find_next_checkpoint_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            spin_up_delay_s=spin_up_delay_s,
        )
        assert result == 0, "Expected checkpoint time to be zero"

    @pytest.mark.parametrize("seed", range(10))
    def test_one_log_multiple_violating_periods_above_threshold_returns_earliest(
        self, seed
    ):
        """
        For a single structured log with multiple violating periods above the threshold,
        return the start time of the earliest violating period.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        start_times = [10, 20]
        end_times = [
            start + 2 * slo_s + np.random.uniform(0, 1) for start in start_times
        ]  # 2* slo_s makes relative violation > 1, ensuring both periods are above
        # any threshold < 1.
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(0, 1)
        spin_up_delay_s = np.random.uniform(0, 10)
        slo_objective = SloObjective(
            slo_metric="relative", slo_threshold=slo_threshold
        )

        result = find_next_checkpoint_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            spin_up_delay_s=spin_up_delay_s,
        )
        expected_time = min(start_times) - spin_up_delay_s
        assert (
            abs(result - expected_time) < 1e-6
        ), "Expected checkpoint time to be the start of the earliest violating period minus spin-up delay"

    @pytest.mark.parametrize("seed", range(10))
    def test_one_log_multiple_violating_periods_ignores_below_threshold(
        self, seed
    ):
        """
        For a single structured log with multiple violating periods, ignore any
        that are below the threshold and return the earliest above-threshold period.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        start_times = [10, 20, 30]
        end_times = [
            start_times[0] + slo_s + slo_s * np.random.uniform(0, 1)
        ] + [
            start + 3 * slo_s + slo_s * np.random.uniform(0, 1)
            for start in start_times[1:]
        ]
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(1, 2)
        spin_up_delay_s = np.random.uniform(0, 10)
        slo_objective = SloObjective(
            slo_metric="relative", slo_threshold=slo_threshold
        )

        result = find_next_checkpoint_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            spin_up_delay_s=spin_up_delay_s,
        )
        expected_time = 20 - spin_up_delay_s
        assert abs(result - expected_time) < 1e-6, (
            "Expected checkpoint time to be the start of the earliest violating "
            "period above threshold minus spin-up delay"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_no_violations(self, seed):
        """
        For multiple structured logs with no violations, return None.
        """
        np.random.seed(seed)
        slo_s = 0.5
        slo_resolver = SloResolver(default_slo_s=slo_s)
        logs = []
        for i in range(3):
            start_times = [j * 10 for j in range(5)]
            end_times = [
                s + np.random.uniform(0, slo_s * 0.9) for s in start_times
            ]
            log = _make_completion_structured_log(start_times, end_times)
            logs.append(log)

        slo_threshold = np.random.uniform(0, 1)
        slo_objective = SloObjective(
            slo_metric="binary", slo_threshold=slo_threshold
        )
        min_delinquent_workloads = np.random.randint(1, 4)
        result = find_next_checkpoint_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=min_delinquent_workloads,
            spin_up_delay_s=5.0,
        )
        assert (
            result is None
        ), "Expected no checkpoint time when there are no violations"

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_one_violating_period(self, seed):
        """
        For multiple structured logs with a clear violating period above the threshold,
        return the start time based on that period.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        logs = []
        for _ in range(3):
            start_times = [100 + j * 10 for j in range(5)]
            end_times = [
                s + 2 * slo_s + np.random.uniform(0, 1) for s in start_times
            ]
            log = _make_completion_structured_log(start_times, end_times)
            logs.append(log)

        slo_threshold = np.random.uniform(0, 1)
        slo_objective = SloObjective(
            slo_metric="relative", slo_threshold=slo_threshold
        )
        spin_up_delay_s = np.random.uniform(0, 10)
        result = find_next_checkpoint_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            spin_up_delay_s=spin_up_delay_s,
        )
        expected_time = 100 - spin_up_delay_s
        condition = abs(result - expected_time) < 1e-6
        if not condition:
            breakpoint()
        assert (
            condition
        ), "Expected checkpoint time to be the start of the violating period minus spin-up delay"

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_multiple_violating_periods_finds(self, seed):
        """
        For multiple structured logs with multiple violating periods above the threshold,
        return the start time of the earliest violating period.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        logs = []
        num_workloads = np.random.randint(2, 5)
        for i in range(num_workloads):
            start_times = [100 + j * 10 for j in range(5)]
            end_times = [
                s + (2 + i) * slo_s + np.random.uniform(0, 1)
                for s in start_times
            ]  # Increasing violation severity across logs, but all above threshold.
            log = _make_completion_structured_log(start_times, end_times)
            logs.append(log)

        slo_threshold = np.random.uniform(0, 1)
        slo_objective = SloObjective(
            slo_metric="relative", slo_threshold=slo_threshold
        )
        min_delinquent_workloads = np.random.randint(1, num_workloads + 1)
        spin_up_delay_s = np.random.uniform(0, 10)
        result = find_next_checkpoint_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=min_delinquent_workloads,
            spin_up_delay_s=spin_up_delay_s,
        )
        expected_time = 100 - spin_up_delay_s
        assert (
            abs(result - expected_time) < 1e-6
        ), "Expected checkpoint time to be the start of the earliest violating period minus spin-up delay"

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_some_violate_but_not_enough(self, seed):
        """
        For multiple structured logs where some have violating periods but not
        enough to meet the minimum delinquent workloads, return None.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        nqueries = np.random.randint(3, 10)

        # Create the first log. This has no violations.
        start_times = [10 + i for i in range(nqueries)]
        end_times = [11 + i for i in range(nqueries)]
        log1 = _make_completion_structured_log(start_times, end_times)

        # Create the second log. This has violations but it's one of two scenarios.
        start_times = [10]
        end_times = [10 + nqueries]
        relative_slo_violation = (end_times[0] - start_times[0]) / slo_s - 1
        log2 = _make_completion_structured_log(start_times, end_times)

        logs = [log1, log2]
        slo_threshold = relative_slo_violation / 2  # So the second log violates
        slo_objective = SloObjective(
            slo_metric="relative", slo_threshold=slo_threshold
        )
        spin_up_delay_s = np.random.uniform(0, 10)

        result = find_next_checkpoint_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=2,  # Requires both logs to be delinquent
            spin_up_delay_s=spin_up_delay_s,
        )
        assert (
            result is None
        ), "Expected no checkpoint time when violation rate is below threshold"

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_multiple_violating_periods_ignores_below_threshold(
        self, seed
    ):
        """
        For multiple structured logs with multiple violating periods, ignore any
        that are below the threshold and return the earliest above-threshold period.
        """

        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(default_slo_s=slo_s)
        nqueries = 2 * np.random.randint(5, 10)  # Even number.

        # Create the first log. This always has relative violations between 3 
        # and 4. 
        start_times = [10 + i * 10 for i in range(nqueries)]
        end_times = [
            s + slo_s * (4 + np.random.uniform(0, 1)) for s in start_times
        ]
        log1 = _make_completion_structured_log(start_times, end_times)

        # Create the second log. This initially has relative violations between 
        # 1 and 2, but then gets worse with relative violations between 3 and 4. 
        start_times = [10 + i * 10 for i in range(nqueries)]
        end_times = [
            s + slo_s * (2 + np.random.uniform(0, 1))
            for s in start_times[: nqueries // 2]
        ] + [
            s + slo_s * (4 + np.random.uniform(0, 1))
            for s in start_times[nqueries // 2 :]
        ]
        log2 = _make_completion_structured_log(start_times, end_times)

        logs = [log1, log2]
        slo_threshold = np.random.uniform(2, 3)
        slo_objective = SloObjective(
            slo_metric="relative", slo_threshold=slo_threshold
        )
        spin_up_delay_s = np.random.uniform(0, 10)
        result = find_next_checkpoint_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=2,  # Requires both logs to be delinquent
            spin_up_delay_s=spin_up_delay_s,
        )
        expected_time = 10 + (nqueries // 2) * 10 - spin_up_delay_s
        condition = abs(result - expected_time) < 1e-6
        if not condition:
            breakpoint()
        assert condition, (
            "Expected checkpoint time to be the start of the earliest violating "
            "period above threshold minus spin-up delay"
        )
