import argparse
import os
from datetime import datetime
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import yaml

# FIXME: bring these queries in eventually
QUERIES_PATH = "/home/markakis/tpc-ds-generator/queries/1721657313/redshift"

AUTOSLO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def get_cluster_dicts_from_config() -> dict[str, dict]:
    """
    Read in the cluster configurations from the autoslo config file.

    Returns:
        A dictionary where keys are cluster names and values are dictionaries
        representing cluster configurations.
    """
    config_path = os.path.join(AUTOSLO_ROOT, "config", "conn.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    cluster_dicts = config.get("clusters", {})
    return cluster_dicts


def get_blueprint_dicts_from_config() -> dict[str, dict]:
    """
    Read in the blueprint configurations from the autoslo config file.

    Returns:
        A dictionary where keys are blueprint names and values are dictionaries
        representing blueprint configurations.
    """
    config_path = os.path.join(AUTOSLO_ROOT, "config", "blueprints.yml")
    with open(config_path, "r") as f:
        blueprint_dicts = yaml.safe_load(f)
    return blueprint_dicts


def get_data_path() -> str:
    """
    Return the absolute DATA_PATH used by autoslo.
    Useful for API routes that need to expose this to the UI.
    """
    return os.path.join(AUTOSLO_ROOT, "data")


def get_runs_path() -> str:
    """
    Return the absolute RUNS_PATH used by autoslo.
    Useful for API routes that need to expose this to the UI.
    """
    return os.path.join(get_data_path(), "runs")


def get_heavy_templates_files() -> dict[str, str]:
    """
    Return the path to the heavy templates file for TPC-DS.
    """
    return {"tpcds": os.path.join(get_data_path(), "tpcds_heavy_templates.txt")}


def list_composite_workloads() -> list[str]:
    """
    Return the names of subdirectories under DATA_PATH/composite_workloads.
    """
    base = os.path.join(get_data_path(), "composite_workloads")
    if not os.path.isdir(base):
        return []
    return sorted(
        d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))
    )


def get_conn_info_path() -> str:
    """
    Return the absolute path to the connection info YAML file.

    Returns:
        The absolute path to the connection info YAML file.

    Raises:
        FileNotFoundError: If the connection info file does not exist.
    """
    conn_info_path = os.path.join(AUTOSLO_ROOT, "config", "conn.yml")
    if not os.path.exists(conn_info_path):
        raise FileNotFoundError(
            f"Connection info file {conn_info_path} does not exist."
        )
    return conn_info_path


class RunLocator:
    """
    The `data/runs` directory is organized by timestamp. This class is in charge
    of cataloguing the parameters of each run, and
    providing easier access to the corresponding directories.
    """

    LAST_RUN_ID: Optional[str] = None
    RUN_SUMMARY_COLS: Optional[list[str]] = None

    @staticmethod
    def get_runs_df(columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per unique run, summarizing the run
        parameters.

        Parameters:
            columns: Optional list of columns to include in the returned
                DataFrame. If None, all columns are included.

        Returns:
            A pandas DataFrame with one row per unique run configuration.

        Raises:
            ValueError: If any of the specified columns are not found in the
                run summary DataFrame.
        """

        last_run_id, run_summary_cols = RunLocator._check_refresh()

        if columns is not None:
            for col in columns:
                if col not in run_summary_cols:
                    raise ValueError(f"Column '{col}' not found in run summary")

        run_summary_path = os.path.join(
            get_runs_path(), f"run_summary_{last_run_id}.parquet"
        )
        run_summary = pd.read_parquet(run_summary_path, columns=columns)

        return run_summary

    @staticmethod
    def get_run_ids(**kwargs) -> list[str]:
        """
        Returns a list of run IDs that match the given filter criteria. For
        integer or float values, an exact match is performed. For string values,
        a substring match is performed. The run IDs are returned in
        chronological order.

        Parameters:
            **kwargs: Key-value pairs to filter the runs. Keys should be column
                names in the run summary DataFrame.

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
                    filtered_summary[k].str.contains(str(v), regex=False)
                ]

        return sorted(filtered_summary["run_id"].tolist())

    @staticmethod
    def _check_refresh(force: bool = False) -> tuple[str, list[str]]:
        """
        Check if the run summary file is already loaded in memory; if not,
        regenerate it if needed.

        Parameters:
            force: If True, force regeneration of the run summary file.

        Returns:
            last_run_id: The ID of the most recent run.
            run_summary_cols: A list of column names in the summary DataFrame.
        """
        if (
            RunLocator.LAST_RUN_ID is not None
            and RunLocator.RUN_SUMMARY_COLS is not None
            and not force
        ):
            last_run_id = RunLocator.LAST_RUN_ID
            run_summary_cols = RunLocator.RUN_SUMMARY_COLS
            return last_run_id, run_summary_cols

        run_dirs = sorted(
            [
                d
                for d in os.listdir(get_runs_path())
                if os.path.isdir(os.path.join(get_runs_path(), d))
            ]
        )
        last_run_id = run_dirs[-1]
        run_summary_files = [
            f
            for f in os.listdir(get_runs_path())
            if f.startswith("run_summary_") and f.endswith(".parquet")
        ]
        if (
            (len(run_summary_files) != 1)
            or (last_run_id not in run_summary_files[0])
            or force
        ):
            last_run_id_internal, run_summary_cols = RunLocator._regenerate(
                run_dirs
            )
            assert last_run_id_internal == last_run_id
            for f in run_summary_files:
                if last_run_id not in f:
                    os.remove(os.path.join(get_runs_path(), f))
        else:
            run_summary_path = os.path.join(
                get_runs_path(), f"run_summary_{last_run_id}.parquet"
            )
            run_summary_cols = pq.read_schema(run_summary_path).names

        return last_run_id, run_summary_cols

    @staticmethod
    def _regenerate(run_dirs: list[str]) -> tuple[str, list[str]]:
        """
        Regenerate the run summary file based on the given list of run
        directories.

        Parameters:
            run_dirs: A list of run directory names to consider for the summary.

        Returns:
            last_run_id: The ID of the most recent run.
            run_summary_cols: A list of column names in the summary DataFrame.
        """

        # For each run dir, read its run_params.yml and append as a dataframe
        # row.
        l = []
        for run_dir in run_dirs:
            run_params_path = os.path.join(
                get_runs_path(), run_dir, "run_params.yml"
            )
            with open(run_params_path, "r") as f:
                run_params = yaml.safe_load(f)
            l.append(run_params)

        run_summary = pd.DataFrame(l).reset_index(drop=True)
        run_summary_cols = run_summary.columns.tolist()

        # Write out the summary dataframe.
        last_run_id = run_dirs[-1]
        run_summary_path = os.path.join(
            get_runs_path(), f"run_summary_{last_run_id}.parquet"
        )
        run_summary.to_parquet(run_summary_path, index=False)

        return last_run_id, run_summary_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of the run summary file",
    )
    args = parser.parse_args()

    last_run_id, run_summary_cols = RunLocator._check_refresh(force=args.force)
    print(
        f"Last run ID: {last_run_id} "
        f"({datetime.fromtimestamp(int(last_run_id))})"
    )
    print(f"Run summary columns: {run_summary_cols}")
