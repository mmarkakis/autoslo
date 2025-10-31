import os
from typing import Any, Optional

import pandas as pd
import yaml

QUERIES_PATH = "/home/markakis/tpc-ds-generator/queries/1721657313/redshift"

CHUNKLOAD_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DATA_PATH = os.path.join(CHUNKLOAD_ROOT, "data")
RUNS_PATH = os.path.join(DATA_PATH, "runs")

HEAVY_TEMPLATES_FILES = {
    "tpcds": os.path.join(DATA_PATH, "tpcds_heavy_templates.txt")
}


def get_data_path() -> str:
    """
    Return the absolute DATA_PATH used by chunkload.
    Useful for API routes that need to expose this to the UI.
    """
    return DATA_PATH


def list_composite_workloads() -> list[str]:
    """
    Return the names of subdirectories under DATA_PATH/composite_workloads.
    """
    base = os.path.join(DATA_PATH, "composite_workloads")
    if not os.path.isdir(base):
        return []
    return sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    )


class RunLocator:
    """
    The `data/runs` directory is organized by timestamp. This class is in charge of maintaining pointers
    to the most recent run of each workload, and providing easier access to the corresponding directories.
    """

    @staticmethod
    def get_runs_df(columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame indicating the most recent run for each set of configuration parameters.

        Parameters:
            columns: Optional list of columns to include in the returned DataFrame. If None, all columns are included.

        Returns:
            A pandas DataFrame with one row per unique run configuration.
        """
        last_run_id = RunLocator._check_refresh()
        run_summary_path = os.path.join(
            RUNS_PATH, f"run_summary_{last_run_id}.parquet"
        )
        run_summary = pd.read_parquet(run_summary_path, columns=columns)

        return run_summary

    @staticmethod
    def get_run_id(**kwargs: dict[str, Any]) -> list[str]:
        """
        Returns a list of run IDs that match the given filter criteria. For integer or float values, an exact match is performed.
        For string values, a substring match is performed.

        Parameters:
            **kwargs: Key-value pairs to filter the runs. Keys should be column names in the run summary DataFrame.

        Returns:
            A list of run IDs that match the filter criteria.
        """
        run_summary = RunLocator.get_runs_df(list(kwargs.keys()) + ["run_id"])

        filtered_summary = run_summary
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                filtered_summary = filtered_summary[filtered_summary[k] == v]
            else:
                filtered_summary = filtered_summary[
                    filtered_summary[k].str.contains(str(v))
                ]

        return list(filtered_summary["run_id"].values)

    @staticmethod
    def _check_refresh() -> str:
        """
        Check if the run summary file is up to date. If not, regenerate it.

        Returns:
            The ID of the most recent run, based on which the current summary file is named.
        """
        run_dirs = sorted(
            [
                d
                for d in os.listdir(RUNS_PATH)
                if os.path.isdir(os.path.join(RUNS_PATH, d))
            ]
        )
        last_run_id = run_dirs[-1]
        run_summary_files = [
            f
            for f in os.listdir(RUNS_PATH)
            if f.startswith("run_summary_") and f.endswith(".parquet")
        ]
        if (len(run_summary_files) != 1) or (
            last_run_id not in run_summary_files[0]
        ):
            RunLocator._regenerate(run_dirs)
            for f in run_summary_files:
                os.remove(os.path.join(RUNS_PATH, f))

        return last_run_id

    @staticmethod
    def _regenerate(run_dirs: list[str]) -> None:
        """
        Regenerate the run summary file based on the given list of run directories.

        Parameters:
            run_dirs: A list of run directory names to consider for the summary.
        """

        # For each run dir, read its run_params.yml and append as a dataframe row.
        l = []
        for run_dir in run_dirs:
            run_params_path = os.path.join(RUNS_PATH, run_dir, "run_params.yml")
            with open(run_params_path, "r") as f:
                run_params = yaml.safe_load(f)
            l.append(run_params)

        run_summary = pd.DataFrame(l)

        # Deduplicate; ignoring the 'run_dir' and 'run_id' columns, only keep the last entry.
        run_summary = run_summary.drop_duplicates(
            subset=[
                c for c in run_summary.columns if c not in ["run_dir", "run_id"]
            ],
            keep="last",
        ).reset_index(drop=True)

        # Write out the summary dataframe.
        last_run_id = run_dirs[-1]
        run_summary_path = os.path.join(
            RUNS_PATH, f"run_summary_{last_run_id}.parquet"
        )
        run_summary.to_parquet(run_summary_path, index=False)
