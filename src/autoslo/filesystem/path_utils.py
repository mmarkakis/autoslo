import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pyarrow.parquet as pq
import yaml

AUTOSLO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

REDSET_RAW_PATH = "/home/markakis/redset"

QUERIES_PATH = os.path.join(AUTOSLO_ROOT, "data", "__query_texts")


def get_redset_raw_path() -> str:
    """
    Return the absolute REDSET_RAW_PATH used by autoslo.
    Useful for API routes that need to expose this to the UI.
    """
    return REDSET_RAW_PATH


def get_config_dir() -> str:
    """
    Return the absolute path to the config directory.
    """
    return os.path.join(AUTOSLO_ROOT, "config")


def get_redset_raw_data(
    cluster_type: str = "provisioned", cluster_id: Union[str, int] = 1
):
    """
    Return the absolute path to the Redset raw data for a given cluster type
    and cluster ID.

    Parameters:
        cluster_type: The type of the cluster ("provisioned" or "serverless").
        cluster_id: The ID of the cluster.

    Returns:
        The absolute path to the Redset raw data for the specified cluster.

    Raises:
        ValueError: If the cluster_type is not recognized.
        IndexError: If the cluster_id is not found in the directory.
    """
    if cluster_type not in ["provisioned", "serverless"]:
        raise ValueError(f"Unknown cluster type: {cluster_type}")

    path = os.path.join(
        REDSET_RAW_PATH, cluster_type, "parts", f"{cluster_id}.parquet"
    )

    if not os.path.exists(path):
        raise IndexError(
            f"Cluster ID {cluster_id} not found in {cluster_type} clusters."
        )

    return path

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


def is_up_to_date(output: Path, *inputs: Path) -> bool:
    """Return True iff *output* exists and is not older than any of *inputs*.

    A missing input path is ignored — it cannot be newer than anything.
    Intended for mtime-based incremental skipping: if this returns True the
    caller can safely skip recomputing *output*.
    """
    if not output.exists():
        return False
    out_mtime = output.stat().st_mtime
    return all(not p.exists() or p.stat().st_mtime <= out_mtime for p in inputs)


def append_to_run_log(
    run_id: str, config_id: str, workload_id: str = ""
) -> None:
    """Append one entry to ``data/runs/run_log.csv``.

    The file is created with a header row on first use, then each
    subsequent call appends a single data row.  The log is intentionally
    append-only and never rewritten.

    Parameters
    ----------
    run_id:
        The timestamp-ms string that names the run output directory.
    config_id:
        The compound ``__``-separated identifier produced by
        :func:`~autoslo.config.utils.make_run_id`, e.g.
        ``"base_iconq__TARGET_DATE=2024-05-27"``.
    workload_id:
        The workload identifier produced by :meth:`WorkloadConfig.id`.
        May be empty for callers that do not track workload.
    """

    log_path = os.path.join(get_runs_path(), "run_log.csv")
    write_header = not os.path.exists(log_path)
    started_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(get_runs_path(), exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "config_id", "workload_id", "started_at"])
        writer.writerow([run_id, config_id, workload_id, started_at])


def find_most_recent_live_run_id(
    config_id: str, workload_id: str
) -> Optional[str]:
    """Return the most recent ``run_id`` for a (workload_id, config_id) pair.

    Reads ``data/runs/run_log.csv`` and returns the entry with the largest
    ``run_id`` (a ms-epoch string) whose ``config_id`` and ``workload_id``
    both match.  Returns ``None`` if the log does not exist, the columns are
    absent, or no matching entry is found.
    """
    log_path = os.path.join(get_runs_path(), "run_log.csv")
    if not os.path.exists(log_path):
        return None
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        if "workload_id" not in (reader.fieldnames or []):
            return None
        best: Optional[str] = None
        for row in reader:
            if row["config_id"] == config_id and row["workload_id"] == workload_id:
                if best is None or int(row["run_id"]) > int(best):
                    best = row["run_id"]
    return best


def get_models_dir() -> str:
    """
    Return the absolute path to the models directory.
    """
    return os.path.join(get_data_path(), "models")


def get_schemas_path() -> str:
    """
    Return the absolute path to the schemas config directory.
    Schema config files live at ``{schemas_path}/{schema_name}.yml``.
    """
    return os.path.join(get_data_path(), "schemas")


def get_query_runner_configs_path() -> str:
    """
    Return the absolute path to the query runner configs directory.
    Config files live at ``{path}/{config_name}.yml``.
    """
    return os.path.join(get_data_path(), "query_runner_configs")


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


def get_workloads_dir() -> str:
    """
    Return the absolute path to the workloads directory.
    """
    return os.path.join(get_data_path(), "workloads")


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
                    filtered_summary[k].str.contains(
                        str(v), regex=False, na=False
                    )
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
    def _load_run_params(run_dir: str) -> dict:
        """
        Load run parameters from a run directory, handling both legacy and
        current config file formats.

        Legacy runs store a flat parameter dict in ``run_params.yml``.
        Newer runs store a deeply nested config in ``config.yml``; these are
        recursively flattened into dot-separated keys (e.g.
        ``workload_config.workload_name``) so that all rows in the summary
        DataFrame share a common ``run_id`` column.

        All non-scalar values (dicts, lists, etc.) are serialised to JSON
        strings so that every column in the resulting summary parquet file
        has a uniform scalar type.

        Parameters:
            run_dir: Name of the run directory (not the full path).

        Returns:
            A flat ``dict`` suitable for inclusion as a DataFrame row.
        """
        import json

        runs_path = get_runs_path()
        old_path = os.path.join(runs_path, run_dir, "run_params.yml")
        new_path = os.path.join(runs_path, run_dir, "config.yml")

        if os.path.exists(old_path):
            with open(old_path, "r") as f:
                raw = yaml.safe_load(f)
        else:
            with open(new_path, "r") as f:
                config = yaml.safe_load(f)

            def _flatten(d: dict, prefix: str = "") -> dict:
                result: dict = {}
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        result.update(_flatten(v, key))
                    else:
                        result[key] = v
                return result

            raw = _flatten(config)

            # Ensure run_id lives at the top level regardless of nesting depth.
            if "run_id" not in raw:
                for key in list(raw.keys()):
                    if key.endswith(".run_id"):
                        raw["run_id"] = raw.pop(key)
                        break

        # Sanitize: convert any remaining non-scalar values to JSON strings so
        # that every column in the parquet summary file has a uniform type.
        # This handles cases like `routing_policy = {}` in legacy runs.
        return {
            k: (
                v
                if isinstance(v, (str, int, float, bool)) or v is None
                else json.dumps(v)
            )
            for k, v in raw.items()
        }

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

        # For each run dir, read its config and append as a dataframe row.
        l = []
        for run_dir in run_dirs:
            run_params = RunLocator._load_run_params(run_dir)
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


class ModelLocator:
    """
    The `data/models` directory is organized by model training run ID. This
    class is in charge of cataloguing the parameters of each model training
    run, and providing easier access to the corresponding directories.

    TODO: refactor this and RunLocator to share code.
    """

    LAST_RUN_ID: Optional[str] = None
    RUN_SUMMARY_COLS: Optional[list[str]] = None
    RUN_SUMMARY_FILENAME = "model_training_run_summary_{}.parquet"

    @staticmethod
    def get_runs_df(
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per unique model training run,
        summarizing the training run parameters and metrics

        Parameters:
            columns: Optional list of columns to include in the returned
                DataFrame. If None, all columns are included.

        Returns:
            A pandas DataFrame with one row per unique model training run.

        """

        last_run_id, run_summary_cols = ModelLocator._check_refresh()
        if last_run_id == "":
            # No model training runs exist yet.
            return pd.DataFrame()

        if columns is not None:
            to_remove = []
            for col in columns:
                if col not in run_summary_cols:
                    to_remove.append(col)
            for col in to_remove:
                columns.remove(col)

        run_summary_path = os.path.join(
            get_models_dir(),
            ModelLocator.RUN_SUMMARY_FILENAME.format(last_run_id),
        )
        run_summary = pd.read_parquet(run_summary_path, columns=columns)

        return run_summary

    @staticmethod
    def get_run_ids(**kwargs) -> list[str]:
        """
        Returns a list of model training run IDs that match the given filter
        criteria. For integer or float values, an exact match is performed.
        For string values, a substring match is performed. The run IDs are
        returned in chronological order.

        Parameters:
            **kwargs: Key-value pairs to filter the model training runs. Keys
                should be column names in the model training run summary
                DataFrame.

        Returns:
            A list of model training run IDs that match the filter criteria.
        """
        # Flatten inputs if there are nested dicts
        for k, v in list(kwargs.items()):
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    kwargs[f"{k}_{k2}"] = v2
                del kwargs[k]

        run_summary = ModelLocator.get_runs_df(list(kwargs.keys()) + ["run_id"])
        if len(run_summary) == 0:
            return []

        filtered_summary = run_summary
        for k, v in kwargs.items():
            if k not in filtered_summary.columns:
                return []

            if isinstance(v, (int, float)):
                filtered_summary = filtered_summary[filtered_summary[k] == v]
            else:
                filtered_summary = filtered_summary[
                    filtered_summary[k].str.contains(str(v), regex=False)
                ]

        return sorted(filtered_summary["run_id"].tolist())

    @staticmethod
    def run_exists(d: dict) -> bool:
        """
        Check if a model training run exists that matches the given filter
        criteria.

        Parameters:
            d: A dictionary of key-value pairs to filter the model training
                runs. Keys should be column names in the model training run
                summary DataFrame.

        Returns:
            True if at least one matching model training run exists, False
            otherwise.
        """
        run_ids = ModelLocator.get_run_ids(**d)
        return len(run_ids) > 0

    @staticmethod
    def _check_refresh(force: bool = False) -> tuple[str, list[str]]:
        """
        Check if the model training parameters summary file is already loaded
        in memory; if not, regenerate it if needed.

        Parameters:
            force: If True, force regeneration of the model training run summary
                file.

        Returns:
            last_run_id: The ID of the most recent model training run.
            run_summary_cols: A list of column names in the summary DataFrame.
        """
        if (
            ModelLocator.LAST_RUN_ID is not None
            and ModelLocator.RUN_SUMMARY_COLS is not None
            and not force
        ):
            return ModelLocator.LAST_RUN_ID, ModelLocator.RUN_SUMMARY_COLS

        run_dirs = sorted(
            [
                d
                for d in os.listdir(get_models_dir())
                if os.path.isdir(os.path.join(get_models_dir(), d))
            ]
        )

        if len(run_dirs) == 0:
            return "", []

        last_run_id = run_dirs[-1].split("_")[0]
        pf, sf = ModelLocator.RUN_SUMMARY_FILENAME.split("{}")[:2]
        cond = lambda f: f.startswith(pf) and f.endswith(sf)
        run_summary_files = [f for f in os.listdir(get_models_dir()) if cond(f)]
        if (
            (len(run_summary_files) != 1)
            or (last_run_id not in run_summary_files[0])
            or force
        ):
            last_run_id_internal, run_summary_cols = ModelLocator._regenerate(
                run_dirs
            )
            assert last_run_id_internal == last_run_id
            for f in run_summary_files:
                if last_run_id not in f:
                    os.remove(os.path.join(get_models_dir(), f))
        else:
            run_summary_path = os.path.join(
                get_models_dir(),
                ModelLocator.RUN_SUMMARY_FILENAME.format(last_run_id),
            )
            run_summary_cols = pq.read_schema(run_summary_path).names

        return last_run_id, run_summary_cols

    @staticmethod
    def _regenerate(
        run_dirs: list[str],
    ) -> tuple[str, list[str]]:
        """
        Regenerate the model training parameters summary file based on the
        given list of model training run directories. This file includes both
        parameters that are inputs to the training process (e.g. the target)
        as well as metrics that are outputs of the training process
        (e.g. accuracy).

        Parameters:
            run_dirs: A list of model training run directory
                names to consider for the summary.

        Returns:
            last_run_id: The ID of the most recent model training run.
            summary_cols: A list of column names in the summary DataFrame.
        """

        # For each model training run dir, read its training_params.yml and
        # its metrics.yml and append as a dataframe row.
        l = []
        all_seen_training_keys = []
        all_seen_metrics_keys = []

        for run_dir in run_dirs:
            d = {}

            # Read training_params.yml
            training_params_path = os.path.join(
                get_models_dir(),
                run_dir,
                "training_params.yml",
            )
            with open(training_params_path, "r") as f:
                training_params = yaml.safe_load(f)

            def read_with_arbitrary_nesting(prefix: str, params: dict):
                for k2, v2 in params.items():
                    if isinstance(v2, dict):
                        read_with_arbitrary_nesting(f"{prefix}{k2}__", v2)
                    else:
                        new_key = f"{prefix}{k2}"
                        d[new_key] = v2
                        if new_key not in all_seen_training_keys:
                            all_seen_training_keys.append(new_key)

            read_with_arbitrary_nesting("", training_params)

            # Read metrics.yml
            metrics_path = os.path.join(
                get_models_dir(),
                run_dir,
                "metrics.yml",
            )
            with open(metrics_path, "r") as f:
                metrics = yaml.safe_load(f)
            d.update(metrics)
            for k in metrics.keys():
                if k not in all_seen_metrics_keys:
                    all_seen_metrics_keys.append(k)

            # Append the row dict
            l.append(d)

        # Create the summary dataframe.
        all_columns = all_seen_training_keys + all_seen_metrics_keys
        all_columns.remove("run_id")
        all_columns = ["run_id"] + all_columns
        summary = (
            pd.DataFrame(l, columns=all_columns)
            .sort_values(by="run_id")
            .reset_index(drop=True)
        )
        summary_cols = summary.columns.tolist()

        # Write out the summary dataframe.
        last_run_id = run_dirs[-1].split("_")[0]
        summary_path = os.path.join(
            get_models_dir(),
            ModelLocator.RUN_SUMMARY_FILENAME.format(last_run_id),
        )
        summary.to_parquet(summary_path, index=False)

        return last_run_id, summary_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of the run and model summary files.",
    )
    args = parser.parse_args()

    last_run_id, run_summary_cols = RunLocator._check_refresh(force=args.force)
    print(
        f"Last run ID: {last_run_id} "
        f"({datetime.fromtimestamp(int(last_run_id))})"
    )
    print(f"Run summary columns: {run_summary_cols}")
