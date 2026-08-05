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

from autoslo.filesystem.structured_events import EventType
from autoslo.filesystem.structured_log import StructuredLog
from autoslo.models.iconq_model import IconqModel
from autoslo.slo.slo_resolver import SloResolver


def logos_df(
    run_id: str,
    slo_resolver: Optional[SloResolver] = None,
    drop_fwd_queries: bool = True,
    include_named_query_features: bool = True,
) -> pd.DataFrame:
    """Return a Logos-ready event-level DataFrame from *log*.

    Calls :meth:`~StructuredLog.flat_df`, then — if *slo_resolver* is
    provided — broadcasts per-query SLO outcomes onto every row for that
    ``query_id``.  Also adds ``actual_execution_latency_s`` (from
    ``query_execution_finish`` rows only) and ``predicted_latency_s``
    (from ``query_routed`` / ``latency_update`` rows only), separating
    the two semantically distinct uses of the ``latency_s`` details field.

    When *include_named_query_features* is ``True`` and the log's stored
    execution config references an ICONQ model, the featurizer is loaded
    and each ``query_text_id`` is expanded into named feature columns.

    Non-query rows (cluster lifecycle events, run start/finish) receive
    NaN for the per-query outcome columns; Logos excludes them from the
    prepared log because they have no ``query_id`` to group by.

    Parameters
    ----------
    log:
        The structured log to export.
    slo_resolver:
        Optional resolver; when given, SLO outcome columns are added.
    drop_fwd_queries:
        Strip autoscaler forward-simulation phantom queries.
    include_named_query_features:
        Expand ``query_text_id`` into per-feature columns using the ICONQ
        model referenced in the log's execution config (if present).
    """
    log = StructuredLog.load(run_id)
    df = log.flat_df(drop_fwd_queries=drop_fwd_queries)

    if slo_resolver is not None:
        outcomes = log.query_slo_outcomes(slo_resolver)
        renamed = outcomes.rename(columns={"latency_s": "final_latency_s"})
        outcome_cols = [
            "final_latency_s",
            "slo_s",
            "slo_violated",
            "slo_overshoot_s",
            "relative_violation",
        ]
        mapping = renamed.set_index("query_id")
        for col in outcome_cols:
            df[col] = df["query_id"].map(mapping[col])

    assignments = log.query_cluster_assignments()
    if not assignments.empty:
        asgn = assignments.set_index("query_id")
        df["selected_cluster_name"] = df["query_id"].map(asgn["cluster_name"])
        df["selected_rpu"] = pd.to_numeric(
            df["query_id"].map(asgn["rpu"]), errors="coerce"
        )
        if slo_resolver is not None:
            df["prediction_error"] = df["final_latency_s"] - df["query_id"].map(
                asgn["latency_s_for_routing"]
            )

    if "latency_s" in df.columns:
        df["actual_execution_latency_s"] = df["latency_s"].where(
            df["event_type"] == EventType.QUERY_EXECUTION_FINISH.value
        )
        df["predicted_latency_s"] = df["latency_s"].where(
            df["event_type"].isin(
                {
                    EventType.QUERY_ROUTED.value,
                    EventType.LATENCY_UPDATE.value,
                }
            )
        )
        df = df.drop(columns=["latency_s"])

    if include_named_query_features and "query_text_id" in df.columns:
        iconq_model_id = (
            log.execution_config.get("query_router_config", {}).get(
                "iconq_model_id"
            )
            if log.execution_config
            else None
        )
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

    return df
