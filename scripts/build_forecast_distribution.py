#!/usr/bin/env python3
"""Build a per-(day_of_week, hour) forecast distribution YAML from historical
query traces.

Usage::

    python scripts/build_forecast_distribution.py \\
        --run-ids 1710000000 1710100000 \\
        --schema-name ext_tpcds1000 \\
        --output data/forecast_distributions/tpcds1000_from_traces.yml

Each supplied ``run_id`` is loaded as a :class:`Trace`.  Arrival times are
binned by ``(weekday, hour)`` and template counts within each bin are
normalised to probabilities.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import yaml

import autoslo.utils.paths as pu
from autoslo.workload_execution.trace import Trace


def _build_forecast(
    run_ids: list[str],
    schema_name: str,
    window_minutes: int = 60,
) -> dict:
    """Return a forecast-distribution dict ready for YAML serialisation."""
    # (day_of_week, hour) → Counter[template_id]
    bin_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    for rid in run_ids:
        trace = Trace(rid)
        text_ids = trace.query_text_ids
        arrivals = trace.arrival_times()

        for qid in trace.query_ids:
            dt = arrivals[qid]
            tid = text_ids[qid].template_id
            bin_counts[(dt.weekday(), dt.hour)][tid] += 1

    # Build bins list.
    bins = []
    for (dow, hour), counter in sorted(bin_counts.items()):
        total = sum(counter.values())
        templates = []
        for tid, cnt in sorted(counter.items(), key=lambda x: -x[1]):
            templates.append(
                {
                    "template_id": str(tid),
                    "probability": round(cnt / total, 6),
                }
            )
        bins.append(
            {"day_of_week": dow, "hour": hour, "templates": templates}
        )

    return {
        "schema_name": schema_name,
        "window_minutes": window_minutes,
        "bins": bins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a forecast distribution YAML from historical traces."
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        required=True,
        help="One or more run IDs (timestamps) from data/runs/.",
    )
    parser.add_argument(
        "--schema-name",
        required=True,
        help="Schema name (e.g. ext_tpcds1000).",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=60,
        help="Bin width in minutes (default 60).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output YAML path (e.g. data/forecast_distributions/my_dist.yml).",
    )
    args = parser.parse_args()

    forecast = _build_forecast(
        run_ids=args.run_ids,
        schema_name=args.schema_name,
        window_minutes=args.window_minutes,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(forecast, f, sort_keys=False, default_flow_style=False)

    n_bins = len(forecast["bins"])
    n_templates = len(
        {
            t["template_id"]
            for b in forecast["bins"]
            for t in b["templates"]
        }
    )
    print(
        f"Wrote {n_bins} bins covering {n_templates} unique templates "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
