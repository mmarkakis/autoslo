"""Tests for PoissonWorkloadCreator.create_poisson_workload_template_blocks."""

from __future__ import annotations

import re

import pytest

import autoslo.filesystem.path_utils as pu
from autoslo.workload_definition.poisson_workload_creator import (
    PoissonWorkloadCreator,
)

_QUERY_TEXT_ID_RE = re.compile(r"^ext_tpcds1000#(\d{3})#(\d{3})$")


@pytest.fixture(autouse=True)
def _isolated_workloads_dir(tmp_path, monkeypatch):
    """Redirect get_data_dir() so generated workloads never touch the real
    data/workloads directory.
    """
    monkeypatch.setattr(pu, "get_data_dir", lambda: tmp_path)
    (tmp_path / "workloads").mkdir(parents=True, exist_ok=True)


def _template_of(query_text_id: str) -> int:
    match = _QUERY_TEXT_ID_RE.match(query_text_id)
    assert (
        match is not None
    ), f"Unexpected query_text_id format: {query_text_id!r}"
    return int(match.group(1))


class TestCreatePoissonWorkloadTemplateBlocks:
    def test_total_query_count(self):
        workload = (
            PoissonWorkloadCreator.create_poisson_workload_template_blocks(
                template_blocks=[(1, 33), (34, 66), (67, 99)],
                num_query_texts_per_template=1,
                num_queries_per_query_text=3,
                poisson_lambda=0.2,
                seed=42,
                print_summary=False,
            )
        )
        # 33 templates/block * 1 query text/template * 3 queries/text = 99
        # queries per block, 3 blocks => 297 total.
        assert workload.num_queries == 297

    def test_blocks_are_issued_in_order_without_leakage(self):
        template_blocks = [(1, 33), (34, 66), (67, 99)]
        workload = (
            PoissonWorkloadCreator.create_poisson_workload_template_blocks(
                template_blocks=template_blocks,
                num_query_texts_per_template=1,
                num_queries_per_query_text=3,
                poisson_lambda=0.2,
                seed=42,
                print_summary=False,
            )
        )
        df = workload.df.sort_values("rel_start_time_s").reset_index(drop=True)
        templates_in_arrival_order = [
            _template_of(qtid) for qtid in df["query_text_id"]
        ]

        queries_per_block = 99
        for i, (first, last) in enumerate(template_blocks):
            start = i * queries_per_block
            end = start + queries_per_block
            block_templates = set(templates_in_arrival_order[start:end])
            # Every query in this chronological slice must come from this
            # block's template range...
            assert block_templates <= set(range(first, last + 1))
            # ...and the slice must not be trivially sorted (i.e. the
            # within-block order was actually shuffled).
            assert templates_in_arrival_order[start:end] != sorted(
                templates_in_arrival_order[start:end]
            )

    def test_single_constant_rate_arrival_process(self):
        """The block structure should not introduce phase-like jumps in the
        arrival rate: inter-arrival times should look like a single Poisson
        process throughout, not one lull/burst per block boundary.
        """
        workload = (
            PoissonWorkloadCreator.create_poisson_workload_template_blocks(
                template_blocks=[(1, 33), (34, 66), (67, 99)],
                num_query_texts_per_template=1,
                num_queries_per_query_text=3,
                poisson_lambda=0.2,
                seed=42,
                print_summary=False,
            )
        )
        rel_times = sorted(workload.df["rel_start_time_s"].tolist())
        inter_arrival = [b - a for a, b in zip(rel_times, rel_times[1:])]
        mean_inter_arrival = sum(inter_arrival) / len(inter_arrival)
        # Expected mean inter-arrival time for a Poisson(lambda=0.2) process
        # is 1/0.2 = 5s; allow generous slack since this is a single
        # 297-sample draw.
        assert 2.5 < mean_inter_arrival < 10.0

    def test_seed_reproducibility(self):
        w1 = PoissonWorkloadCreator.create_poisson_workload_template_blocks(
            template_blocks=[(1, 33), (34, 66), (67, 99)],
            num_query_texts_per_template=1,
            num_queries_per_query_text=3,
            poisson_lambda=0.2,
            seed=42,
            print_summary=False,
        )
        w2 = PoissonWorkloadCreator.create_poisson_workload_template_blocks(
            template_blocks=[(1, 33), (34, 66), (67, 99)],
            num_query_texts_per_template=1,
            num_queries_per_query_text=3,
            poisson_lambda=0.2,
            seed=42,
            print_summary=False,
        )
        assert w1.workload_name == w2.workload_name
        assert (
            w1.df["query_text_id"].tolist() == w2.df["query_text_id"].tolist()
        )
