import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import autoslo.filesystem.path_utils as pu
from autoslo.workload_definition.workload import Workload, WorkloadConfig


class PoissonWorkloadCreator:
    @staticmethod
    def name_from_params(
        num_templates: int,
        num_query_texts_per_template: int,
        num_queries_per_query_text: int,
        poisson_lambda: float,
        seed: int,
    ) -> str:
        return "_".join(
            [
                "poisson",
                str(num_templates),
                str(num_query_texts_per_template),
                str(num_queries_per_query_text),
                str(poisson_lambda),
                str(seed),
            ]
        )

    @staticmethod
    def create_poisson_workload(
        num_templates: int,
        num_query_texts_per_template: int,
        num_queries_per_query_text: int,
        poisson_lambda: float,
        seed: int,
        print_summary: bool = True,
    ) -> Workload:

        rng = np.random.default_rng(seed)
        workload_name = PoissonWorkloadCreator.name_from_params(
            num_templates=num_templates,
            num_query_texts_per_template=num_query_texts_per_template,
            num_queries_per_query_text=num_queries_per_query_text,
            poisson_lambda=poisson_lambda,
            seed=seed,
        )

        # Determine which templates to use
        all_templates = list(range(1, 100))
        rng.shuffle(all_templates)
        selected_templates = all_templates[:num_templates]

        # Create the queries in sorted order and then shuffle.
        query_text_ids = []
        for template_idx in sorted(selected_templates):
            for query_text_idx in range(1, num_query_texts_per_template + 1):
                for _ in range(1, num_queries_per_query_text + 1):
                    query_text_ids.append(
                        f"ext_tpcds1000#{template_idx:03d}#{query_text_idx:03d}"
                    )
        rng.shuffle(query_text_ids)

        # Create submission times using a Poisson process.
        n_queries = len(query_text_ids)
        inter_arrival_times_s = rng.exponential(
            1 / poisson_lambda, size=n_queries - 1
        )
        rel_arrival_times_s = [0] + list(np.cumsum(inter_arrival_times_s))

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
    create_poisson_workload(
        num_templates=args.num_templates,
        num_query_texts_per_template=args.num_query_texts_per_template,
        num_queries_per_query_text=args.num_queries_per_query_text,
        poisson_lambda=args.poisson_lambda,
        seed=args.seed,
    )
