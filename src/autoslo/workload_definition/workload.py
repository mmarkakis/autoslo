from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from rich.table import Table

import autoslo.utils.paths as pu
from autoslo.config.component_configs import WorkloadConfig
from autoslo.featurization.iconq_query_featurizer import IconqQueryFeaturizer
from autoslo.models.iconq_model import IconqModel
from autoslo.workload_definition.query import Query, QueryTextId

# Columns that every workload file is expected to provide.
WORKLOAD_SCHEMA_COLUMNS: list[str] = [
    "query_id",
    "abs_start_time",
    "query_text_id",
    "repetition_id",
]


class Workload:
    """A workload backed by a file in the ``workload`` data-schema format.

    Each row in the backing ``DataFrame`` corresponds to one submitted query
    and must include the columns listed in :data:`WORKLOAD_SCHEMA_COLUMNS`.
    The ``workload_name`` column is uniform across all rows and serves as the
    workload's :attr:`name`.

    The class can also be used as a base class.  Subclasses that manage their
    own data store should override :attr:`name` and :meth:`queries`; they do
    not need to call ``super().__init__()``.
    """

    WORKLOAD_SCHEMA_COLUMNS: list[str] = [
        "query_id",
        "abs_start_time",
        "query_text_id",
        "repetition_id",
    ]

    def __init__(
        self,
        workload_config: WorkloadConfig,
        df: pd.DataFrame | None = None,
    ) -> None:
        """
        Parameters
        ----------
        workload_config:
            The configuration for the workload.

        df:
            Optional DataFrame to use directly instead of loading from disk.
            Must contain all columns listed in :data:`WORKLOAD_SCHEMA_COLUMNS`.
            When *None* (default), the workload is loaded from the directory
            specified by *workload_config*.

        Raises
        ------
        ValueError
            If the DataFrame (loaded or supplied) is missing any of the
            required columns from :data:`WORKLOAD_SCHEMA_COLUMNS`.
        ValueError
            If the DataFrame references multiple distinct schema names in the
            ``query_text_id`` column.
        FileNotFoundError
            If *df* is *None* and no parquet file exists at the expected path.
        """
        self._workload_config = workload_config
        self._queries_cache: list[Query] | None = None
        self._rescale_factor = workload_config.rescale_factor
        self._df: pd.DataFrame
        self._dir = workload_config.workload_dir or (
            Path(pu.get_data_path()) / "workloads"
        )

        # Find the dataframe.
        if df is not None:
            self._df = df.copy()
        else:
            path = os.path.join(
                self._dir,
                f"{workload_config.workload_name}.parquet",
            )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Workload file not found at {path}. "
                    "Provide a DataFrame directly or ensure the file exists."
                )
            self._df = pd.read_parquet(path)

        # Validate the schema.
        for col in WORKLOAD_SCHEMA_COLUMNS:
            if col not in self._df.columns:
                raise ValueError(
                    f"DataFrame is missing required column {col!r} from "
                    f"WORKLOAD_SCHEMA_COLUMNS."
                )
        unique_schema_names = (
            self._df["query_text_id"]
            .apply(lambda x: QueryTextId(x).schema_name)
            .unique()
        )
        if len(unique_schema_names) != 1:
            raise ValueError(
                f"DataFrame has multiple schema names in query_text_id: "
                f"{unique_schema_names}"
            )
        self._schema_name = unique_schema_names[0]

        # Slice.
        tz = self._df["abs_start_time"].dt.tz
        mask = pd.Series(True, index=self._df.index)
        if workload_config.start_date_inclusive is not None:
            parsed_start = (
                pd.Timestamp(workload_config.start_date_inclusive)
                .normalize()
                .tz_localize(tz)
            )
            mask &= self._df["abs_start_time"] >= parsed_start
        if workload_config.end_date_inclusive is not None:
            parsed_end = pd.Timestamp(
                workload_config.end_date_inclusive
            ).normalize().tz_localize(tz) + pd.Timedelta(days=1)
            mask &= self._df["abs_start_time"] < parsed_end
        self._df = self._df[mask].reset_index(drop=True)

        # Apply timing transformations.
        self.rescale_rel_times(workload_config.rescale_factor)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @property
    def workload_name(self) -> str:
        """The workload identifier, taken from the ``workload_name`` column."""
        return self._workload_config.workload_name

    @property
    def rescale_factor(self) -> float:
        """The factor by which the workload's relative start times have been
        scaled, taken from the workload config."""
        return self._rescale_factor

    @property
    def df(self) -> pd.DataFrame:
        """
        The underlying :class:`~pandas.DataFrame`.
        """
        return self._df

    @property
    def num_queries(self) -> int:
        """The number of queries in the workload."""
        return len(self._df)

    def queries(self) -> list[Query]:
        """Return the list of :class:`~autoslo.workload_definition.query.Query`
        objects derived from the workload rows.

        Each row contributes exactly one :class:`Query`.  Optional execution
        fields (latency, featurization, …) are left at their defaults; callers
        should populate them after a run.
        """
        if self._queries_cache is not None:
            return self._queries_cache
        if self._df is None:
            raise NotImplementedError(
                f"{type(self).__name__} must either be backed by a DataFrame "
                "or override 'queries'."
            )
        result: list[Query] = []
        for _, row in self._df.iterrows():
            result.append(
                Query(
                    query_id=str(row["query_id"]),
                    query_text_id=QueryTextId(row["query_text_id"]),
                    repetition_id=str(row.get("repetition_id", "")),
                    rel_start_time_s=float(row.get("rel_start_time_s", -1)),
                    featurization=row.get("featurization", []),
                    stage_predictions_per_rpu=row.get(
                        "stage_predictions_per_rpu", {}
                    ),
                )
            )
        self._queries_cache = result
        return result

    def rescale_rel_times(self, factor: float) -> Workload:
        """
        Rescale the relative start times of the workload by the given factor.
        """
        min_timestamp = self._df["abs_start_time"].min().timestamp()
        self._df["rel_start_time_s"] = self._df["abs_start_time"].apply(
            lambda t: t.timestamp() - min_timestamp
        )
        self._df["rel_start_time_s"] = self._df["rel_start_time_s"] * factor
        self._rescale_factor = factor
        self._queries_cache = None
        return self

    def populate_featurizations_and_isolated_predictions(
        self, iconq_model: IconqModel, allowed_rpu_sizes: list[int]
    ) -> None:
        """Populate featurization and isolated predictions for all queries."""

        # Find the distinct query_text_ids in the workload
        query_text_ids = set(self._df["query_text_id"].unique())
        # For each query_text_id, compute the featurization and predictions
        featurization_cache: dict[
            str, IconqQueryFeaturizer.IconqQueryFeaturization
        ] = {}
        for query_text_id in query_text_ids:
            featurization = (
                iconq_model.iconq_query_featurizer.featurize_from_query_text_id(
                    query_text_id
                )
            )
            featurization_cache[query_text_id] = featurization

        # Now also use the stage model to populate stage predictions per RPU for each query
        isolated_predictions_cache: dict[str, dict[int, float]] = {}
        for query_text_id in query_text_ids:
            isolated_predictions_cache[query_text_id] = {}
            featurization = featurization_cache[query_text_id]
            for rpu in allowed_rpu_sizes:
                isolated_predictions_cache[query_text_id][rpu] = (
                    iconq_model.stage_model.predict_from_query_text_id(
                        {"0": QueryTextId(query_text_id)}, rpu
                    )["0"].overall_mean_s()
                )

        # Now add to the dataframe
        self._df["featurization"] = self._df["query_text_id"].apply(
            lambda qtid: featurization_cache[qtid]
        )
        self._df["stage_predictions_per_rpu"] = self._df["query_text_id"].apply(
            lambda qtid: isolated_predictions_cache[qtid]
        )

    def save(self, overwrite: bool = False) -> Path:
        """
        Persist the workload DataFrame to
        <workload_config.workload_name>.parquet under the directory specified
        when generating the workload (defaulting to the standard path under the
        ``workloads`` data directory if not specified).

        Parameters
        ----------
        overwrite:
            If *False* (default) and the file already exists, raises
            :class:`FileExistsError`.  Set to *True* to overwrite.

        Returns
        -------
        Path
            The path of the written file.

        Raises
        ------
        FileExistsError
            If a file already exists at the target path and *overwrite* is
            *False*.
        """
        path = (
            Path(self._dir) / f"{self._workload_config.workload_name}.parquet"
        )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Workload file already exists at {path}. "
                "Pass overwrite=True to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_parquet(path, index=False)
        return path

    def print_summary(self) -> None:
        """Print a summary of the workload using rich."""
        stats_table = Table(title="Workload Summary")

        stats_table.add_column("Stat", style="cyan", no_wrap=True)
        stats_table.add_column("Value", style="magenta")
        stats_table.add_row(
            "Workload Name", self._workload_config.workload_name
        )
        stats_table.add_row("Schema Name", self._schema_name)
        stats_table.add_row(
            "Start Date (inclusive)",
            str(self._df["abs_start_time"].min().date()),
        )
        stats_table.add_row(
            "End Date (inclusive)", str(self._df["abs_start_time"].max().date())
        )
        stats_table.add_row(
            "Rescale Factor", str(self._workload_config.rescale_factor)
        )
        stats_table.add_row("", "")
        stats_table.add_row("Total Queries", str(len(self._df)))
        num_unique_templates = (
            self._df["query_text_id"]
            .apply(lambda x: QueryTextId(x).template_id)
            .nunique()
        )
        stats_table.add_row("Unique Query Templates", str(num_unique_templates))
        stats_table.add_row(
            "Unique Template+Query Index",
            str(self._df["query_text_id"].nunique()),
        )
        stats_table.add_row(
            "Rescaled Duration (seconds)",
            str(self._df["rel_start_time_s"].max()),
        )
        stats_table.add_row(
            "Rescaled Mean Inter-Arrival Time (seconds)",
            str(self._df["rel_start_time_s"].diff().mean()),
        )
        print(stats_table)

    def abs_start_time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the minimum and maximum absolute start times in the workload."""
        return (
            self._df["abs_start_time"].min(),
            self._df["abs_start_time"].max(),
        )

    def get_rel_time_s_to_table_vecs(
        self, iconq_query_featurizer: IconqQueryFeaturizer
    ) -> dict[float, np.ndarray]:
        """
        For each hour in the workload based on *absolute* start times,
        map the relative time at the start of that hour to a matrix.

        The matrix has shape (num_queries_in_that_hour, table_featurization_dim)
        and contains the featurizations of the queries that start in that hour.
        """

        result: dict[float, np.ndarray] = {}
        for _, group in self._df.groupby(
            pd.Grouper(key="abs_start_time", freq="H")
        ):
            if group.empty:
                continue
            rel_time_s = group["rel_start_time_s"].min()
            table_feats = []
            for _, row in group.iterrows():
                table_feats.append(
                    iconq_query_featurizer.table_vector_for(
                        row["query_text_id"]
                    )
                )
            result[rel_time_s] = np.stack(table_feats, axis=0)
        return result
