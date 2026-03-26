#!/usr/bin/env python3
"""Build a per-(day_of_week, hour) forecast distribution YAML from a workload.

Usage::

    python scripts/build_forecast_distribution.py \\
        --workload-name my_workload \\
        --schema-name ext_tpcds1000 \\
        --output data/forecast_distributions/tpcds1000.yml

    # Optionally restrict to a time window:
    python scripts/build_forecast_distribution.py \\
        --workload-name my_workload \\
        --schema-name ext_tpcds1000 \\
        --start "2024-05-27T00:00:00" \\
        --end "2024-05-28T00:00:00" \\
        --output data/forecast_distributions/tpcds1000.yml

Delegates to :func:`autoslo.workload_definition.forecast_distribution.build_forecast_distribution`.
"""

from __future__ import annotations

import argparse

from autoslo.workload_definition.forecast_distribution import (
    build_forecast_distribution,
    save_forecast_distribution,
)
from autoslo.workload_definition.workload import Workload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a forecast distribution YAML from a workload."
    )
    parser.add_argument(
        "--workload-name",
        required=True,
        help="Workload name (e.g. redbench_provisioned_157_0).",
    )
    parser.add_argument(
        "--schema-name",
        required=True,
        help="Schema name (e.g. ext_tpcds1000).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional ISO-8601 lower bound for abs_start_time (inclusive).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional ISO-8601 upper bound for abs_start_time (inclusive).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output YAML path (e.g. data/forecast_distributions/my_dist.yml).",
    )
    args = parser.parse_args()

    workload = Workload(
        workload_name=args.workload_name,
        schema_name=args.schema_name,
    )

    dist = build_forecast_distribution(
        workload=workload,
        start=args.start,
        end=args.end,
    )

    path = save_forecast_distribution(dist, args.output)

    n_bins = len(dist["bins"])
    n_templates = len(
        {
            t["template_id"]
            for b in dist["bins"]
            for t in b["templates"]
        }
    )
    total_queries = sum(b["query_count"] for b in dist["bins"])
    print(
        f"Wrote {n_bins} bins covering {n_templates} unique templates "
        f"({total_queries} total queries) to {path}"
    )


if __name__ == "__main__":
    main()
