"""Tests for SpinupOptimizer and find_next_spinup_time_df."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoslo.clusters.scheduled_spinup import ScheduledSpinUp
from autoslo.slo.slo_objective import SloObjective
from autoslo.config.component_configs import (
    SloObjectiveConfig,
    SloResolverConfig,
)
from autoslo.slo.slo_resolver import SloResolver
from autoslo.tuner.spinup_optimizer import (
    add_spinup_to_config,
    find_next_spinup_time_df,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_completion_structured_log(
    start_times: list[float],
    end_times: list[float],
) -> pd.DataFrame:
    """
    Helper to create a completion structured log DataFrame as expected by
    find_next_spinup_time_df: columns rel_time_s, latency_s, query_id,
    query_text_id.
    """
    return pd.DataFrame(
        {
            "rel_time_s": end_times,
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


class TestAddSpinupToConfig:
    def test_nonzero_time_appends_to_scheduled_spinups(self):
        """
        A spinup with nonzero rel_time_s should be appended to scheduled_spinups.
        """
        initial_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
            "autoscaling_config": {"autoscaling_policy": "headroom"},
        }
        spinup = ScheduledSpinUp(rel_time_s=100, rpu=8)
        new_config = add_spinup_to_config(config=initial_config, spinup=spinup)
        expected_new_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
            "autoscaling_config": {"autoscaling_policy": "headroom"},
            "scheduled_spinups": [{"rel_time_s": 100, "rpu": 8}],
        }
        assert new_config == expected_new_config

    def test_zero_time_folds_into_initial_rpus(self):
        """
        A spinup with rel_time_s=0 should fold its RPU into initial_rpus.
        """
        initial_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 4]},
            "autoscaling_config": {"autoscaling_policy": "headroom"},
        }
        spinup = ScheduledSpinUp(rel_time_s=0, rpu=2)
        new_config = add_spinup_to_config(config=initial_config, spinup=spinup)
        expected_new_config = {
            "managed_cluster_pool_config": {"initial_rpus": [1, 2, 2, 4]},
            "autoscaling_config": {"autoscaling_policy": "headroom"},
        }
        assert new_config == expected_new_config

    def test_multiple_nonzero_spinups_accumulate(self):
        """
        Calling add_spinup_to_config twice for nonzero times accumulates both entries.
        """
        initial_config = {
            "managed_cluster_pool_config": {"initial_rpus": [4]},
        }
        config = add_spinup_to_config(
            config=initial_config, spinup=ScheduledSpinUp(rel_time_s=100, rpu=8)
        )
        config = add_spinup_to_config(
            config=config, spinup=ScheduledSpinUp(rel_time_s=200, rpu=16)
        )
        assert config["scheduled_spinups"] == [
            {"rel_time_s": 100, "rpu": 8},
            {"rel_time_s": 200, "rpu": 16},
        ]

    def test_original_config_not_mutated(self):
        """
        add_spinup_to_config must return a new dict and not mutate the original.
        """
        initial_config = {
            "managed_cluster_pool_config": {"initial_rpus": [4]},
        }
        spinup = ScheduledSpinUp(rel_time_s=50, rpu=8)
        _ = add_spinup_to_config(config=initial_config, spinup=spinup)
        assert "scheduled_spinups" not in initial_config


class TestFindNextSpinupTimeDF:
    def test_one_log_no_violations(self):
        """
        For a single structured log with no violations, return None.
        """
        slo_s = 0.5
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
        start_times = [0, 10, 20]
        end_times = [s + np.random.uniform(0, slo_s * 0.9) for s in start_times]
        log = _make_completion_structured_log(start_times, end_times)

        slo_objective = SloObjective(
            SloObjectiveConfig(slo_metric="binary", slo_threshold=0.0)
        )

        result = find_next_spinup_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            lead_time_s=5.0,
        )
        assert (
            result == []
        ), "Expected no spinup time when there are no violations"

    def test_one_log_insufficient_violations(self):
        """
        For a single structured log with some violations but below the threshold,
        return None.
        """
        slo_s = 2
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))

        # The first query violates. But at any given point, there is at least one
        # more query running that doesn't,
        start_times = [10, 10, 10.5, 11, 11.5, 12, 12.5, 13]
        end_times = [13, 11, 11.5, 12, 12.5, 13, 13.5, 14]
        log = _make_completion_structured_log(start_times, end_times)

        slo_objective = SloObjective(
            SloObjectiveConfig(slo_metric="binary", slo_threshold=0.6)
        )

        result = find_next_spinup_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            lead_time_s=5.0,
        )
        assert (
            result == []
        ), "Expected no spinup time when violation rate is below threshold"

    @pytest.mark.parametrize("seed", range(10))
    def test_one_log_one_violating_period(self, seed):
        """
        For a single structured log with a clear violating period above the threshold,
        return the start time of that period.
        """
        np.random.seed(seed)
        slo_s = 1
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
        start_times = [10]
        end_times = [12]
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(0, 1)
        lead_time_s = np.random.uniform(0, 10)
        slo_objective = SloObjective(
            SloObjectiveConfig(slo_metric="binary", slo_threshold=slo_threshold)
        )
        result = find_next_spinup_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            lead_time_s=lead_time_s,
        )
        assert (
            result[0] == 10 - lead_time_s
        ), "Expected spinup time to be the start of the violating period minus spin-up delay"

    @pytest.mark.parametrize("seed", range(10))
    def test_one_log_bottom_at_zero(self, seed):
        """
        A violating period that starts well before ``lead_time_s`` still
        produces a valid placement candidate at 0.
        """
        np.random.seed(seed)
        slo_s = 1
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
        # Violation starts in [0, 5) and ends 2s later, so placement is clamped to 0.
        start_times = [np.random.uniform(0, 5)]
        end_times = [start_times[0] + 2]
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(0, 1)
        lead_time_s = np.random.uniform(5, 10)
        slo_objective = SloObjective(
            SloObjectiveConfig(slo_metric="binary", slo_threshold=slo_threshold)
        )
        result = find_next_spinup_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            lead_time_s=lead_time_s,
        )
        assert result, "Expected at least one candidate"
        assert result[0] == 0.0, "Expected candidate time to be clamped to 0"

    @pytest.mark.parametrize("seed", range(10))
    def test_one_log_multiple_violating_periods_in_detection_order(self, seed):
        """
        The result list is returned in chronological detection order.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
        # Two epochs, both above any slo_threshold < 1.
        start_times = [10, 100]
        end_times = [
            10 + slo_s * 2,  # short epoch
            100 + slo_s * 12,  # long epoch
        ]
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(0, 1)
        lead_time_s = 0.0  # placement times equal epoch starts exactly
        slo_objective = SloObjective(
            SloObjectiveConfig(
                slo_metric="relative", slo_threshold=slo_threshold
            )
        )

        result = find_next_spinup_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            lead_time_s=lead_time_s,
        )
        assert len(result) >= 1, "Expected at least one candidate"
        assert result == sorted(result), (
            "Expected candidates to be in chronological detection order"
        )

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
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
        start_times = [10, 20, 30]
        end_times = [
            start_times[0] + slo_s + slo_s * np.random.uniform(0, 1)
        ] + [
            start + 3 * slo_s + slo_s * np.random.uniform(0, 1)
            for start in start_times[1:]
        ]
        log = _make_completion_structured_log(start_times, end_times)

        slo_threshold = np.random.uniform(1, 2)
        lead_time_s = np.random.uniform(0, 10)
        slo_objective = SloObjective(
            SloObjectiveConfig(
                slo_metric="relative", slo_threshold=slo_threshold
            )
        )

        result = find_next_spinup_time_df(
            completion_structured_logs=[log],
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            lead_time_s=lead_time_s,
        )
        expected_time = 20 - lead_time_s
        assert abs(result[0] - expected_time) < 1e-6, (
            "Expected spinup time to be the start of the earliest violating "
            "period above threshold minus spin-up delay"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_no_violations(self, seed):
        """
        For multiple structured logs with no violations, return None.
        """
        np.random.seed(seed)
        slo_s = 0.5
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
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
            SloObjectiveConfig(slo_metric="binary", slo_threshold=slo_threshold)
        )
        min_delinquent_workloads = np.random.randint(1, 4)
        result = find_next_spinup_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=min_delinquent_workloads,
            lead_time_s=5.0,
        )
        assert (
            result == []
        ), "Expected no spinup time when there are no violations"

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_one_violating_period(self, seed):
        """
        For multiple structured logs with a clear violating period above the threshold,
        return the start time based on that period.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
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
            SloObjectiveConfig(
                slo_metric="relative", slo_threshold=slo_threshold
            )
        )
        lead_time_s = np.random.uniform(0, 10)
        result = find_next_spinup_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=1,
            lead_time_s=lead_time_s,
        )
        expected_time = 100 - lead_time_s
        condition = abs(result[0] - expected_time) < 1e-6
        if not condition:
            breakpoint()
        assert (
            condition
        ), "Expected spinup time to be the start of the violating period minus spin-up delay"

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_multiple_violating_periods_finds(self, seed):
        """
        For multiple structured logs with multiple violating periods above the threshold,
        return the start time of the earliest violating period.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
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
            SloObjectiveConfig(
                slo_metric="relative", slo_threshold=slo_threshold
            )
        )
        min_delinquent_workloads = np.random.randint(1, num_workloads + 1)
        lead_time_s = np.random.uniform(0, 10)
        result = find_next_spinup_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=min_delinquent_workloads,
            lead_time_s=lead_time_s,
        )
        expected_time = 100 - lead_time_s
        assert (
            abs(result[0] - expected_time) < 1e-6
        ), "Expected spinup time to be the start of the earliest violating period minus spin-up delay"

    @pytest.mark.parametrize("seed", range(10))
    def test_multi_log_some_violate_but_not_enough(self, seed):
        """
        For multiple structured logs where some have violating periods but not
        enough to meet the minimum delinquent workloads, return None.
        """
        np.random.seed(seed)

        slo_s = 1
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
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
            SloObjectiveConfig(
                slo_metric="relative", slo_threshold=slo_threshold
            )
        )
        lead_time_s = np.random.uniform(0, 10)

        result = find_next_spinup_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=2,  # Requires both logs to be delinquent
            lead_time_s=lead_time_s,
        )
        assert (
            result == []
        ), "Expected no spinup time when violation rate is below threshold"

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
        slo_resolver = SloResolver(SloResolverConfig(slo_s=slo_s))
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
            SloObjectiveConfig(
                slo_metric="relative", slo_threshold=slo_threshold
            )
        )
        lead_time_s = np.random.uniform(0, 10)
        result = find_next_spinup_time_df(
            completion_structured_logs=logs,
            slo_resolver=slo_resolver,
            slo_objective=slo_objective,
            min_delinquent_workloads=2,  # Requires both logs to be delinquent
            lead_time_s=lead_time_s,
        )
        expected_time = 10 + (nqueries // 2) * 10 - lead_time_s
        condition = abs(result[0] - expected_time) < 1e-6
        if not condition:
            breakpoint()
        assert condition, (
            "Expected spinup time to be the start of the earliest violating "
            "period above threshold minus spin-up delay"
        )
