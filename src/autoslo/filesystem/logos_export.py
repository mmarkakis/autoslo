"""logos_export.py
-----------------
Produces a Logos-ready DataFrame from a :class:`StructuredLog`.

Lives in its own module so it can freely import both
``autoslo.filesystem.structured_log`` and ``autoslo.models.iconq_model``
without closing the circular-import loop that would arise if ``IconqModel``
were imported directly from ``structured_log``.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from autoslo.config.component_configs import QueryRouterConfig
from autoslo.filesystem.structured_events import EventType
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.models.iconq_model import IconqModel
from autoslo.slo.slo_resolver import SloResolver


def logos_df(
    run_id: str,
    slo_resolver: Optional[SloResolver] = None,
    drop_fwd_queries: bool = True,
    include_named_query_features: bool = True,
    include_overlap_counts: bool = True,
) -> pd.DataFrame:
    """
    Return a Logos-ready event-level DataFrame from *log*.

    Parameters
    ----------
    run_id:
        The run ID of the log to export.
    slo_resolver:
        Optional resolver; when given, SLO outcome columns are added.
    drop_fwd_queries:
        Strip autoscaler forward-simulation phantom queries.
    include_named_query_features:
        Expand ``query_text_id`` into per-feature columns using the ICONQ
        model referenced in the log's execution config (if present).
    """
    # Get basic flattened log dataframe.
    log = StructuredLog.load(run_id)
    df = log.flat_df(drop_fwd_queries=drop_fwd_queries)

    # Bring in SLO compliance information.
    if slo_resolver is not None:
        outcomes = log.query_slo_outcomes(slo_resolver)
        df = df.merge(outcomes, how="left")

    # Bring in routed cluster information.
    assignments = log.query_cluster_assignments()
    if not assignments.empty:
        asgn = assignments.set_index("query_id")
        df["routed_cluster_name"] = df["query_id"].map(asgn["cluster_name"])
        df["routed_cluster_rpu"] = pd.to_numeric(
            df["query_id"].map(asgn["rpu"]), errors="coerce"
        )

    # Deal with the overloaded `latency_s` column. Only use the values from
    # QUERY_ROUTED and LATENCY_UPDATE events. Ignore the execution-phase
    # latency_s values reported in QUERY_EXECUTION_FINISH events.
    if "latency_s" in df.columns:
        initial_predicted_latency = df["latency_s"].where(
            df["event_type"] == EventType.QUERY_ROUTED.value
        )
        df["initial_predicted_latency_over_slo"] = (
            initial_predicted_latency
        ) / df["slo_s"]
        updated_predicted_latency = df["latency_s"].where(
            df["event_type"] == EventType.LATENCY_UPDATE.value
        )

        df["updated_predicted_latency_over_slo"] = (
            updated_predicted_latency
        ) / df["slo_s"]

        df = df.drop(columns=["latency_s"])

    # Bring in query features, if requested and available.
    exec_cfg = log.execution_config
    if (
        include_named_query_features
        and ("query_text_id" in df.columns)
        and (exec_cfg is not None)
    ):
        iconq_model_id = QueryRouterConfig.from_config(exec_cfg).iconq_model_id
        if iconq_model_id is not None:
            qf = IconqModel.load(iconq_model_id).iconq_query_featurizer
            unique_ids = df["query_text_id"].dropna().unique()
            query_features_df = pd.DataFrame.from_dict(
                {
                    qtid: qf.featurize_from_query_text_id_as_dict(qtid)
                    for qtid in unique_ids
                },
                orient="index",
            )
            query_features_df.index.name = "query_text_id"
            df = df.merge(
                query_features_df,
                how="left",
                left_on="query_text_id",
                right_index=True,
            )

    # Bring in overlap counts, if requested and available.
    if include_overlap_counts and ("query_id" in df.columns):
        overlap_counts = log.query_overlap_counts()
        if not overlap_counts.empty:
            df = df.merge(overlap_counts, how="left")

    return df
