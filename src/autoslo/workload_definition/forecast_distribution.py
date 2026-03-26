"""Build per-(day_of_week, hour) forecast distributions from a Workload.

The resulting YAML artefact is consumed by
:class:`~autoslo.routing.forecast_loader.ForecastDistributionLoader` at
routing time.

Typical usage::

    from autoslo.workload_definition.workload import Workload
    from autoslo.workload_definition.forecast_distribution import (
        build_forecast_distribution,
        save_forecast_distribution,
    )

    wl = Workload("my_workload", "ext_tpcds1000")
    dist = build_forecast_distribution(
        wl,
        start="2024-05-27T00:00:00",
        end="2024-05-28T00:00:00",
    )
    save_forecast_distribution(dist, "data/forecast_distributions/my_dist.yml")
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from autoslo.workload_definition.query import QueryTextId
from autoslo.workload_definition.workload import Workload


def build_forecast_distribution(
    workload: Workload,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Compute a per-(day_of_week, hour) forecast distribution.

    The workload's ``abs_start_time`` column is used to assign each query
    to a ``(weekday, hour)`` bin.  Within each bin the template-level
    probabilities are computed and the total query count is recorded.

    Parameters
    ----------
    workload :
        A :class:`Workload` instance (may be file-backed or in-memory).
    start :
        Optional ISO-8601 lower bound (inclusive) for ``abs_start_time``.
    end :
        Optional ISO-8601 upper bound (inclusive) for ``abs_start_time``.

    Returns
    -------
    dict
        A forecast-distribution dictionary ready for
        :func:`save_forecast_distribution`.  Structure::

            {
                "schema_name": str,
                "bins": [
                    {
                        "day_of_week": int,
                        "hour": int,
                        "query_count": int,
                        "templates": [
                            {"template_id": str, "probability": float},
                            ...
                        ],
                    },
                    ...
                ],
            }
    """
    df = workload.df.copy()

    # Apply optional time bounds.
    if start is not None or end is not None:
        import pandas as pd  # noqa: PLC0415

        tz = df["abs_start_time"].dt.tz

        def _parse(ts_str: str) -> "pd.Timestamp":
            ts = pd.Timestamp(ts_str)
            if tz is not None and ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            elif tz is None and ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts

        if start is not None:
            df = df[df["abs_start_time"] >= _parse(start)]
        if end is not None:
            df = df[df["abs_start_time"] <= _parse(end)]

    # Bin by (day_of_week, hour).
    bin_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for _, row in df.iterrows():
        dt = row["abs_start_time"]
        tid = QueryTextId(row["query_text_id"]).template_id
        bin_counts[(dt.weekday(), dt.hour)][tid] += 1

    bins = []
    for (dow, hour), counter in sorted(bin_counts.items()):
        total = sum(counter.values())
        templates = [
            {
                "template_id": str(tid),
                "probability": round(cnt / total, 6),
            }
            for tid, cnt in sorted(counter.items(), key=lambda x: -x[1])
        ]
        bins.append(
            {
                "day_of_week": dow,
                "hour": hour,
                "query_count": total,
                "templates": templates,
            }
        )

    return {
        "schema_name": workload._schema_name,
        "bins": bins,
    }


def save_forecast_distribution(
    distribution: dict,
    output_path: str | Path,
) -> Path:
    """Write a forecast distribution dict to a YAML file.

    Parent directories are created automatically.

    Parameters
    ----------
    distribution :
        The dict returned by :func:`build_forecast_distribution`.
    output_path :
        Destination file path.

    Returns
    -------
    Path
        The resolved output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(distribution, f, sort_keys=False, default_flow_style=False)
    return path
