from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from autoslo.blueprint_selection.query_timeline import QueryTimeline
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.workload_execution.trace import Trace

BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)


def dt_after(seconds: float) -> datetime:
    return BASE_TIME + timedelta(seconds=seconds)


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


class DummyIconqQueryFeaturizer:
    """Lightweight IconqQueryFeaturizer stub for QueryTimeline tests."""

    def featurize_from_tpcds_temp_and_q_idx(
        self, tpcds_temp_and_q_idx: tuple[str, int]
    ) -> list[float]:
        return [0.0, 1.5, 2.5]


def build_timeline(specs: list[tuple[str, float, float]]) -> QueryTimeline:
    trace = cast(Trace, DummyTrace(specs))
    featurizer = cast(IconqQueryFeaturizer, DummyIconqQueryFeaturizer())
    return QueryTimeline(trace, featurizer)


def test_overlap_graph_builds_expected_edges() -> None:
    """Ensure overlaps produce graph edges."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 5.0, 15.0),
            ("q3", 20.0, 30.0),
        ]
    )
    edges = {frozenset(edge) for edge in timeline.overlap_graph().edges()}
    assert edges == {frozenset({"q1", "q2"})}


def test_add_rejects_duplicate_query_id() -> None:
    """Ensure adding duplicate query ids fails."""
    timeline = build_timeline([("q1", 0.0, 10.0)])
    with pytest.raises(ValueError):
        timeline.add("q1", 0.0, 5.0, "tmpl")


def test_add_connects_overlapping_queries() -> None:
    """Ensure add links new overlapping queries."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 20.0, 30.0),
        ]
    )
    timeline.add(
        "q3", dt_after(9.0).timestamp(), dt_after(25.0).timestamp(), "tmpl"
    )
    graph = timeline.overlap_graph()
    assert graph.has_edge("q3", "q1")
    assert graph.has_edge("q3", "q2")


def test_remove_unknown_query_id_raises_error() -> None:
    """Ensure removing missing query ids fails."""
    timeline = build_timeline([("q1", 0.0, 10.0)])
    with pytest.raises(ValueError):
        timeline.remove("missing")


def test_update_latency_shorter_purges_edges() -> None:
    """Ensure shrinking latency drops trailing edges."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 10.0),
            ("q2", 9.0, 20.0),
        ]
    )
    graph = timeline.overlap_graph()
    assert graph.has_edge("q1", "q2")
    timeline.update_latency("q1", 5.0)
    graph = timeline.overlap_graph()
    assert not graph.has_edge("q1", "q2")


def test_update_latency_longer_adds_edges() -> None:
    """Ensure extending latency connects future overlaps."""
    timeline = build_timeline(
        [
            ("q1", 0.0, 5.0),
            ("q2", 6.0, 8.0),
            ("q3", 10.0, 12.0),
        ]
    )
    graph = timeline.overlap_graph()
    assert not graph.has_edge("q1", "q2")
    timeline.update_latency("q1", 7.0)
    graph = timeline.overlap_graph()
    assert graph.has_edge("q1", "q2")
    assert not graph.has_edge("q1", "q3")


def test_update_latency_unknown_query_id_raises_error() -> None:
    """Ensure updating missing query ids fails."""
    timeline = build_timeline([("q1", 0.0, 10.0)])
    with pytest.raises(ValueError):
        timeline.update_latency("missing", 5.0)
