import argparse
import os

import numpy as np
import pandas as pd

import autoslo.utils.paths as pu


def main(args) -> None:

    # Load the workload for the refrerence day, one week before the test day.
    reference_day = pd.to_datetime(args.test_day) - pd.Timedelta(days=7)
    reference_day_str = reference_day.strftime("%Y-%m-%d")
    reference_day_workload_path = os.path.join(
        args.workload_dir, "days", f"{reference_day_str}.parquet"
    )
    reference_day_df = pd.read_parquet(reference_day_workload_path)

    # 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workload_dir",
        type=str,
        default=os.path.join(
            pu.get_data_path(),
            "redset_workloads",
            "provisioned_cluster12_seed42",
        ),
        help="Path to input workload directory.",
    )

    parser.add_argument(
        "--test_day",
        type=str,
        default="2024-03-15",
        help="The test day for which to sample workloads (format: YYYY-MM-DD).",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of sample workloads to create for the selected test day.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    args = parser.parse_args()

    main(args)
