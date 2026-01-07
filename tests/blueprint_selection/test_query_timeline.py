from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from intervaltree import Interval  # type: ignore[import]

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.query_timeline import QueryTimeline
from autoslo.blueprints.cluster import Cluster
from autoslo.featurization.iconq_interaction_featurizer import (
    IconqInteractionFeaturizer,
)
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_execution.trace import Trace

BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def dt_after(seconds: float) -> datetime:
    return BASE_TIME + timedelta(seconds=seconds)


def default_cluster_name() -> str:
    return Cluster.all_cluster_names()[0]


class DummyTrace:
    """Lightweight trace stub for QueryTimeline tests."""

    def __init__(self, specs: list[tuple[str, float, float]]) -> None:
        self.run_id = "dummy-run"
        self.query_ids = [query_id for query_id, _, _ in specs]
        self.tpcds_temp_and_q_idxs = {
            query_id: "tmpl" for idx, (query_id, _, _) in enumerate(specs)
        }
        self._arrival = {
            query_id: dt_after(start) for query_id, start, _ in specs
        }
        self._completion = {
            query_id: dt_after(end) for query_id, _, end in specs
        }

    def arrival_times(self) -> dict[str, datetime]:
        return self._arrival

    def completion_times(self) -> dict[str, datetime]:
        return self._completion

    def cluster_name_from_query_id(self, query_id: str) -> str:
        return default_cluster_name()


class DummyIconqQueryFeaturizer:
    """Lightweight IconqQueryFeaturizer stub for QueryTimeline tests."""

    def featurize_from_tpcds_temp_and_q_idx(
        self, tpcds_temp_and_q_idx: tuple[str, int]
    ) -> list[float]:
        return [0.0, 1.5, 2.5]


class DummyIconqInteractionFeaturizer:
    """Lightweight IconqInteractionFeaturizer stub for QueryTimeline tests."""

    def featurize_from_vectors(
        self,
        qa_features: list[float],
        qa_start_time_s: float,
        qa_latency_prediction: float,
        qb_features: list[float],
        qb_start_time_s: float,
        qb_latency_prediction: float,
    ) -> list[float]:
        return [0.0, 1.0, 2.0, 3.0]


def build_timeline(specs: list[tuple[str, float, float]]) -> QueryTimeline:
    trace = cast(Trace, DummyTrace(specs))
    featurizer = cast(IconqQueryFeaturizer, DummyIconqQueryFeaturizer())
    interaction_featurizer = cast(
        IconqInteractionFeaturizer, DummyIconqInteractionFeaturizer()
    )
    timeline = QueryTimeline(featurizer, interaction_featurizer)
    timeline.initialize_from_trace(trace)
    return timeline


def test_overlap_graph_builds_expected_edges() -> None:
    """Ensure overlaps produce graph edges."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 5.0, 15.0),
            ("q3", 20.0, 30.0),
        ]
    )
    for query_id_a in timeline.query_ids:
        for query_id_b in timeline.query_ids:
            if query_id_a == query_id_b:
                continue
            overlap = timeline.overlap(query_id_a, query_id_b)
            if {query_id_a, query_id_b} == {"q1", "q2"}:
                assert (
                    overlap
                ), f"Expected overlap between {query_id_a} and {query_id_b}"
            else:
                assert (
                    not overlap
                ), f"Did not expect overlap between {query_id_a} and {query_id_b}"


def test_update_latency_shorter_removes_overlap() -> None:
    """Ensure shrinking latency removes overlaps."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 9.0, 20.0),
        ]
    )
    assert timeline.overlap("q1", "q2")
    timeline.update_latency("q1", 5.0)
    assert not timeline.overlap("q1", "q2")


def test_update_latency_longer_adds_overlaps() -> None:
    """Ensure extending latency creates overlaps."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 5.0),
            ("q2", 6.0, 8.0),
            ("q3", 10.0, 12.0),
        ]
    )
    assert not timeline.overlap("q1", "q2")
    timeline.update_latency("q1", 7.0)
    assert timeline.overlap("q1", "q2")
    assert not timeline.overlap("q1", "q3")


def test_update_latency_unknown_query_id_raises_error() -> None:
    """Ensure updating missing query ids fails."""
    timeline = build_timeline([("q1", 0.0, 10.0)])
    with pytest.raises(ValueError):
        timeline.update_latency("missing", 5.0)


def test_move_to_cluster_removes_overlaps() -> None:
    """Ensure move_to_cluster leads to no overlaps."""
    timeline = build_timeline([("q1", 0.0, 10.0), ("q2", 5.0, 15.0)])
    assert timeline.overlap("q1", "q2")
    timeline.move_to_cluster(new_cluster_name="new-cluster", query_id="q2")
    assert not timeline.overlap("q1", "q2")


def test_move_to_cluster_unknown_query_id_raises_error() -> None:
    """Ensure moving missing query ids fails."""
    timeline = build_timeline([("q1", 0.0, 10.0)])
    with pytest.raises(KeyError):
        timeline.move_to_cluster(
            new_cluster_name="new-cluster", query_id="missing"
        )


def test_find_worst_offending_intervals_unweighted() -> None:
    """Ensure worst offending intervals are found correctly."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 5.0, 15.0),
            ("q3", 12.0, 20.0),
            ("q4", 18.0, 25.0),
        ]
    )
    worst_intervals = timeline.find_intervals_by_slo_adherence(
        slo_s=4.0, look_for_slo_violations=True, weigh_by_distance=False
    )
    expected_intervals = [
        (
            default_cluster_name(),
            Interval(dt_after(5.0).timestamp(), dt_after(10.0).timestamp()),
        ),
        (
            default_cluster_name(),
            Interval(dt_after(12.0).timestamp(), dt_after(15.0).timestamp()),
        ),
        (
            default_cluster_name(),
            Interval(dt_after(18.0).timestamp(), dt_after(20.0).timestamp()),
        ),
    ]
    assert (
        worst_intervals == expected_intervals
    ), f"Expected {expected_intervals}, got {worst_intervals}"


def test_find_worst_offending_intervals_weighted() -> None:
    """Ensure worst offending intervals are found correctly."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 5.0, 15.0),
            ("q3", 12.0, 20.0),
            ("q4", 18.0, 25.0),
        ]
    )
    worst_intervals = timeline.find_intervals_by_slo_adherence(
        slo_s=4.0, look_for_slo_violations=True, weigh_by_distance=True
    )
    expected_intervals = [
        (
            default_cluster_name(),
            Interval(dt_after(5.0).timestamp(), dt_after(10.0).timestamp()),
        ),
    ]
    assert (
        worst_intervals == expected_intervals
    ), f"Expected {expected_intervals}, got {worst_intervals}"


def test_find_worst_offending_intervals_no_violations() -> None:
    """Ensure no intervals are found when there are no SLO violations."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 3.0),
            ("q2", 4.0, 7.0),
            ("q3", 8.0, 11.0),
        ]
    )
    worst_intervals = timeline.find_intervals_by_slo_adherence(
        slo_s=5.0, look_for_slo_violations=True
    )
    expected_intervals: list[tuple[str, Interval]] = []
    assert (
        worst_intervals == expected_intervals
    ), f"Expected {expected_intervals}, got {worst_intervals}"


def test_find_slack_intervals() -> None:
    """Ensure slack intervals are found correctly."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 3.0),
            ("q2", 4.0, 7.0),
            ("q3", 8.0, 11.0),
        ]
    )
    slack_intervals = timeline.find_intervals_by_slo_adherence(
        slo_s=5.0, look_for_slo_violations=False
    )
    expected_intervals = [
        (
            default_cluster_name(),
            Interval(dt_after(0.0).timestamp(), dt_after(3.0).timestamp()),
        ),
        (
            default_cluster_name(),
            Interval(dt_after(4.0).timestamp(), dt_after(7.0).timestamp()),
        ),
        (
            default_cluster_name(),
            Interval(dt_after(8.0).timestamp(), dt_after(11.0).timestamp()),
        ),
    ]
    assert (
        slack_intervals == expected_intervals
    ), f"Expected {expected_intervals}, got {slack_intervals}"


def test_find_slack_intervals_overlapping() -> None:
    """Ensure slack intervals are found correctly with overlaps."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 5.0, 15.0),
            ("q3", 20.0, 30.0),
        ]
    )
    slack_intervals = timeline.find_intervals_by_slo_adherence(
        slo_s=20.0, look_for_slo_violations=False
    )
    expected_intervals = [
        (
            default_cluster_name(),
            Interval(dt_after(5.0).timestamp(), dt_after(10.0).timestamp()),
        ),
    ]
    assert (
        slack_intervals == expected_intervals
    ), f"Expected {expected_intervals}, got {slack_intervals}"
