from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from typing import Optional

import autoslo.filesystem.path_utils as pu
from autoslo.workload_definition.workload import Workload, WorkloadConfig


@dataclass(frozen=True)
class PoissonArrivalPhase:
    num_queries: int
    poisson_lambda: float

    @property
    def name(self) -> str:
        return f"{self.num_queries}q{self.poisson_lambda}l"

    def generate_rel_arrival_times_s(
        self,
        rng: np.random.Generator,
        reference_rel_time_s: float,
        include_gap_pre_first: bool = True,
    ) -> list[float]:
        inter_arrival_times_s = list(
            rng.exponential(1 / self.poisson_lambda, size=self.num_queries)
        )
        if not include_gap_pre_first:
            inter_arrival_times_s = [0] + inter_arrival_times_s[:-1]
        arrival_times_s = (
            np.cumsum(inter_arrival_times_s) + reference_rel_time_s
        )
        return arrival_times_s.tolist()


class PoissonWorkloadCreator:
    @staticmethod
    def name_from_params(
        num_templates: int,
        num_query_texts_per_template: int,
        num_queries_per_query_text: int | None,
        phases: list[PoissonArrivalPhase],
        seed: int,
        num_total_queries: int | None = None,
    ) -> str:
        global_poisson_lambda = (
            phases[0].poisson_lambda if len(phases) == 1 else None
        )
        name = "_".join(
            [
                "poisson",
                str(num_templates),
                str(num_query_texts_per_template),
                str(num_queries_per_query_text),
                str(global_poisson_lambda),
                str(seed),
            ]
        )

        if num_total_queries is not None:
            name += f"_{num_total_queries}"

        if len(phases) > 1:
            phase_str = "_".join(phase.name for phase in phases)
            name += f"_phased_{phase_str}"

        return name

    @staticmethod
    def make_phased_profile(
        num_queries_per_phase: list[int],
        poisson_lambda_per_phase: list[float],
    ) -> list[PoissonArrivalPhase]:
        """
        Create a multi-phase workload with the given number of queries and
        Poisson lambdas. The i-th phase will have num_queries_per_phase[i]
        queries and a Poisson lambda of poisson_lambda_per_phase[i]. The lambda
        governs interarrival times *before* each query in a phase.
        """
        return [
            PoissonArrivalPhase(num_queries=num_q, poisson_lambda=lam)
            for num_q, lam in zip(
                num_queries_per_phase, poisson_lambda_per_phase
            )
        ]

    @staticmethod
    def make_bursty_profile(
        total_num_queries: int,
        num_lull_burst_cycles: int,
        lull_poisson_lambda: float,
        burst_poisson_lambda: float,
        fraction_of_queries_in_bursts: float,
    ) -> list[PoissonArrivalPhase]:
        """
        Generate a phased workload alternating between "lull" phases with low
        arrival rate and "burst" phases with high arrival rate.

        Parameters
        ----------
        name: Name for the profile.
        total_num_queries: Total number of queries across all phases.
        num_lull_burst_cycles: Number of alternating "lull" and "burst" cycles.
        lull_poisson_lambda: Poisson lambda for "lull" phases.
        burst_poisson_lambda: Poisson lambda for "burst" phases.
        fraction_of_queries_in_bursts: Fraction of queries that arriva during
        "burst" phases.
        """

        num_queries_per_burst = int(
            total_num_queries
            * fraction_of_queries_in_bursts
            / num_lull_burst_cycles
        )
        pessimistic_num_queries_per_lull = (
            total_num_queries - num_queries_per_burst * num_lull_burst_cycles
        ) // num_lull_burst_cycles

        remaining_queries = total_num_queries
        phases = []
        while remaining_queries > 0:
            # Add lull phase
            num_queries_this_lull = min(
                pessimistic_num_queries_per_lull, remaining_queries
            )
            remaining_queries -= num_queries_this_lull
            phases.append(
                PoissonArrivalPhase(
                    num_queries=num_queries_this_lull,
                    poisson_lambda=lull_poisson_lambda,
                )
            )

            # Add burst phase
            num_queries_this_burst = min(
                num_queries_per_burst, remaining_queries
            )
            remaining_queries -= num_queries_this_burst
            phases.append(
                PoissonArrivalPhase(
                    num_queries=num_queries_this_burst,
                    poisson_lambda=burst_poisson_lambda,
                )
            )

        return phases

    @staticmethod
    def create_poisson_workload(
        num_templates: int,
        num_query_texts_per_template: int,
        num_queries_per_query_text: int,
        poisson_lambda: float,
        seed: int,
        print_summary: bool = True,
    ) -> Workload:
        return PoissonWorkloadCreator.create_poisson_workload_phased(
            num_templates=num_templates,
            num_query_texts_per_template=num_query_texts_per_template,
            num_queries_per_query_text=num_queries_per_query_text,
            phases=[
                PoissonArrivalPhase(
                    num_queries=(
                        num_templates
                        * num_query_texts_per_template
                        * num_queries_per_query_text
                    ),
                    poisson_lambda=poisson_lambda,
                )
            ],
            seed=seed,
            print_summary=print_summary,
            include_total_in_name=False,
        )

   

    @staticmethod
    def create_poisson_workload_phased(
        num_templates: int,
        num_query_texts_per_template: int,
        num_queries_per_query_text: Optional[int],
        phases: list[PoissonArrivalPhase],
        seed: int,
        print_summary: bool = True,
        include_total_in_name: bool = False,
    ) -> Workload:
        num_total_queries = sum(phase.num_queries for phase in phases)
        rng = np.random.default_rng(seed)
        workload_name = PoissonWorkloadCreator.name_from_params(
            num_templates=num_templates,
            num_query_texts_per_template=num_query_texts_per_template,
            num_queries_per_query_text=num_queries_per_query_text,
            phases=phases,
            seed=seed,
            num_total_queries=(
                None if not include_total_in_name else num_total_queries
            ),
        )

        # Determine which templates to use
        all_templates = list(range(1, 100))
        rng.shuffle(all_templates)
        selected_templates = all_templates[:num_templates]

        # Create the queries in sorted order and then shuffle.
        if num_queries_per_query_text is None:
            num_queries_per_query_text = num_total_queries
        query_text_ids = []
        for template_idx in sorted(selected_templates):
            for query_text_idx in range(1, num_query_texts_per_template + 1):
                for _ in range(1, num_queries_per_query_text + 1):
                    query_text_ids.append(
                        f"ext_tpcds1000#{template_idx:03d}#{query_text_idx:03d}"
                    )
        rng.shuffle(query_text_ids)
        query_text_ids = query_text_ids[:num_total_queries]

        return PoissonWorkloadCreator._package_queries_into_workload(
            query_text_ids=query_text_ids,
            phases=phases,
            workload_name=workload_name,
            print_summary=print_summary,
            rng=rng,
        )

    @staticmethod
    def _package_queries_into_workload(
        query_text_ids: list[str],
        phases: list[PoissonArrivalPhase],
        workload_name: str,
        print_summary: bool,
        rng: np.random.Generator,
    ) -> Workload:

        # Create submission times using a Poisson process for each phase.
        reference_rel_time_s = 0.0
        rel_arrival_times_s = phases[0].generate_rel_arrival_times_s(
            rng=rng,
            reference_rel_time_s=reference_rel_time_s,
            include_gap_pre_first=False,
        )
        for phase in phases[1:]:
            reference_rel_time_s = rel_arrival_times_s[-1]
            phase_rel_arrival_times_s = phase.generate_rel_arrival_times_s(
                rng=rng,
                reference_rel_time_s=reference_rel_time_s,
                include_gap_pre_first=True,
            )
            rel_arrival_times_s.extend(phase_rel_arrival_times_s)

        # Package into workload.
        gen_start_time = datetime.fromisoformat("2026-01-01T00:00:00")
        records = []
        for i, (query_text_id, rel_arrival_time_s) in enumerate(
            zip(query_text_ids, rel_arrival_times_s)
        ):
            record = {
                "query_id": f"query_{i}",
                "abs_start_time": gen_start_time
                + timedelta(seconds=rel_arrival_time_s),
                "query_text_id": query_text_id,
                "repetition_id": f"query_{i}",
            }
            records.append(record)
        df = pd.DataFrame(records)
        out_path = os.path.join(
            pu.get_workloads_dir(), f"{workload_name}.parquet"
        )
        df.to_parquet(out_path)

        # Print a nice summary
        workload_config = WorkloadConfig(
            workload_name=workload_name,
        )
        workload = Workload(workload_config)
        if print_summary:
            workload.print_summary()

        return workload

    @staticmethod
    def create_poisson_workload_with_n_queries(
        num_templates: int,
        num_query_texts_per_template: int,
        num_total_queries: int,
        poisson_lambda: float,
        seed: int,
        print_summary: bool = True,
    ) -> Workload:
        return PoissonWorkloadCreator.create_poisson_workload_phased(
            num_templates=num_templates,
            num_query_texts_per_template=num_query_texts_per_template,
            num_queries_per_query_text=None,
            phases=[
                PoissonArrivalPhase(
                    num_queries=num_total_queries,
                    poisson_lambda=poisson_lambda,
                )
            ],
            seed=seed,
            print_summary=print_summary,
            include_total_in_name=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Poisson workload.")
    parser.add_argument(
        "--poisson_lambda",
        type=float,
        default=0.5,
        help=(
            "The lambda parameter for the Poisson distribution, "
            "in queries per second."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_templates",
        type=int,
        default=99,
        help=(
            "Number of unique TPC-DS templates from which to draw queries. "
            "Derived after shuffling with `seed`, not ordered."
        ),
    )
    parser.add_argument(
        "--num_query_texts_per_template",
        type=int,
        default=1,
        help="Number of unique query_texts to generate per template.",
    )
    parser.add_argument(
        "--num_queries_per_query_text",
        type=int,
        default=3,
        help="Number of queries to generate per query_text.",
    )
    args = parser.parse_args()
    PoissonWorkloadCreator.create_poisson_workload(
        num_templates=args.num_templates,
        num_query_texts_per_template=args.num_query_texts_per_template,
        num_queries_per_query_text=args.num_queries_per_query_text,
        poisson_lambda=args.poisson_lambda,
        seed=args.seed,
    )
