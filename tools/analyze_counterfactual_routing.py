"""
Analyze counterfactual routing breakdown in the autoscaler's forward-looking
counterfactual pass.

For each candidate cluster size (RPU), classifies queries into:
  1. Routed BEFORE the new cluster's spinup was complete
     (hypothetical cluster still PENDING → only old cluster available)
  2. Routed AFTER spinup to the OLD cluster
     (hypothetical cluster READY but old cluster was preferred)
  3. Routed AFTER spinup to the NEW (hypothetical) cluster

How the log encodes this:
  - Autoscaler.QueryRouter / routing_score events record every cluster that
    was *considered* for a query.  A routing_score entry for
    autoslo-{rpu}-hypothetical means the hypothetical cluster was READY when
    that query arrived (i.e. the query is post-spinup for this candidate RPU).
  - Autoscaler.QueryRouter / routing events record where each query was
    *actually sent*.
  - The query_id prefix (fwd-0, fwd-1, fwd-2, …) is the window-copy index
    inside the forward-looking replay.

Because there is one independent replay per candidate RPU, the same
(fwd_pass, query_id) key appears once in each replay's routing events.
Routing_score events against a hypothetical cluster name belong exclusively
to that RPU's replay, so (fwd_pass, query_id) pairs with a score against
autoslo-{rpu}-hypothetical are exactly the post-spinup queries for that RPU.

Usage:
    python analyze_counterfactual_routing.py <path/to/structured_log.parquet>
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


def extract_fwd_and_bare(query_id: pd.Series):
    """Split 'fwd-N:query_X' into ('fwd-N', 'query_X')."""
    fwd = query_id.str.extract(r"^(fwd-\d+):", expand=False)
    bare = query_id.str.extract(r"^fwd-\d+:(.*)", expand=False)
    return fwd, bare


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Analyze autoscaler counterfactual routing breakdown."
    )
    parser.add_argument(
        "log",
        metavar="structured_log.parquet",
        type=Path,
        help="Path to the structured_log.parquet file to analyze.",
    )
    return parser.parse_args().log


def main():
    log_path = parse_args()
    df = pd.read_parquet(log_path)

    aqr = df[df["source"] == "Autoscaler.QueryRouter"].copy()

    # Only keep records whose query_id has the fwd-N: prefix.
    mask_fwd = aqr["query_id"].str.match(r"^fwd-\d+:", na=False)
    aqr = aqr[mask_fwd].copy()

    aqr["fwd"], aqr["bare_qid"] = extract_fwd_and_bare(aqr["query_id"])
    aqr["pair"] = list(zip(aqr["fwd"], aqr["bare_qid"]))

    routing = aqr[aqr["event_type"] == "routing"]
    scores = aqr[aqr["event_type"] == "routing_score"]

    # All distinct (fwd, query_id) pairs that appear in ANY replica.
    # Since the old cluster is always available, a routing_score event for the
    # old cluster exists in every replica for every query.  The distinct pairs
    # are therefore the union across all replicas = the full query universe for
    # a single replica (replicas replay identical query sequences).
    old_cluster = routing["cluster_name"].loc[
        ~routing["cluster_name"].str.contains("hypothetical", na=False)
    ].iloc[0]  # e.g. 'autoslo-16-1779306643737-0'

    total_queries_per_replica = len(set(scores['query_id']))

    # Candidate RPU sizes are inferred from the hypothetical cluster names
    # that appear in either routing or routing_score events.
    hyp_clusters = set(
        routing.loc[
            routing["cluster_name"].str.contains("hypothetical", na=False),
            "cluster_name",
        ]
    ) | set(
        scores.loc[
            scores["cluster_name"].str.contains("hypothetical", na=False),
            "cluster_name",
        ]
    )

    def rpu_from_name(name: str) -> int:
        m = re.search(r"autoslo-(\d+)-hypothetical", name)
        return int(m.group(1)) if m else -1

    # SLO violation and cost per candidate RPU from rpu_counterfactual records.
    counterfactual_events = df[
        (df["source"] == "Autoscaler")
        & (df["event_type"] == "rpu_counterfactual")
    ].copy()
    cf_details = counterfactual_events["details"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else s
    )
    counterfactual_events["rpu"] = cf_details.apply(lambda d: d["rpu"])
    counterfactual_events["slo_violation"] = cf_details.apply(
        lambda d: d["slo_violation"]
    )
    counterfactual_events["cost"] = cf_details.apply(lambda d: d["cost"])
    cf_by_rpu = (
        counterfactual_events.set_index("rpu")[["slo_violation", "cost"]]
    )

    # Chosen RPU from the rpu_selection record.
    selection_rows = df[
        (df["source"] == "Autoscaler") & (df["event_type"] == "rpu_selection")
    ]
    chosen_rpu: int | None = None
    if not selection_rows.empty:
        sel_details = json.loads(selection_rows.iloc[-1]["details"])
        chosen_rpu = sel_details["rpu"]

    results = []
    for hyp_name in sorted(hyp_clusters, key=rpu_from_name):
        rpu = rpu_from_name(hyp_name)

        # Pairs that were scored against this hypothetical → post-spinup.
        post_spinup_pairs = set(
            zip(
                scores.loc[scores["cluster_name"] == hyp_name, "fwd"],
                scores.loc[scores["cluster_name"] == hyp_name, "bare_qid"],
            )
        )

        # Pairs that were actually routed to this hypothetical.
        to_new_pairs = set(
            zip(
                routing.loc[routing["cluster_name"] == hyp_name, "fwd"],
                routing.loc[routing["cluster_name"] == hyp_name, "bare_qid"],
            )
        )

        cat1 = total_queries_per_replica - len(post_spinup_pairs)
        cat3 = len(to_new_pairs)
        cat2 = len(post_spinup_pairs) - cat3

        slo_viol = cf_by_rpu.loc[rpu, "slo_violation"] if rpu in cf_by_rpu.index else float("nan")
        cost = cf_by_rpu.loc[rpu, "cost"] if rpu in cf_by_rpu.index else float("nan")

        results.append(
            {
                "candidate_rpu": rpu,
                "hypothetical_cluster": hyp_name,
                "total_queries_per_replica": total_queries_per_replica,
                "1_before_spinup": cat1,
                "2_after_spinup_to_old_cluster": cat2,
                "3_after_spinup_to_new_cluster": cat3,
                "slo_violation": slo_viol,
                "cost": cost,
            }
        )

    result_df = pd.DataFrame(results).set_index("candidate_rpu")

    console = Console()

    console.print()
    console.print(
        f"[bold]Counterfactual routing breakdown[/bold]  –  [cyan]{log_path}[/cyan]"
    )
    if chosen_rpu is not None:
        console.print(
            f"Old (real) cluster: [green]{old_cluster}[/green]  │  "
            f"Chosen new RPU: [bold yellow]{chosen_rpu}[/bold yellow]"
        )
    else:
        console.print(f"Old (real) cluster: [green]{old_cluster}[/green]")
    console.print(
        f"Total unique (fwd-pass, query) pairs per replica: "
        f"[bold]{total_queries_per_replica}[/bold]"
    )
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Candidate RPU", justify="right")
    table.add_column("Hypothetical cluster", justify="left", no_wrap=True)
    table.add_column("Total queries", justify="right")
    table.add_column("① Before spinup", justify="right")
    table.add_column("② After spinup → old", justify="right")
    table.add_column("③ After spinup → new", justify="right")
    table.add_column("SLO violation", justify="right", no_wrap=True)
    table.add_column("Cost", justify="right", no_wrap=True)

    for rpu, row in result_df.iterrows():
        is_chosen = rpu == chosen_rpu
        style = "bold yellow" if is_chosen else ""
        label = lambda v: f"[bold yellow]{v}[/bold yellow]" if is_chosen else str(v)  # noqa: E731
        table.add_row(
            label(rpu),
            label(row["hypothetical_cluster"]),
            label(row["total_queries_per_replica"]),
            label(row["1_before_spinup"]),
            label(row["2_after_spinup_to_old_cluster"]),
            label(row["3_after_spinup_to_new_cluster"]),
            label(f"{row['slo_violation']:.4f}"),
            label(f"{row['cost']:.4f}"),
            style=style,
        )

    console.print(table)
    console.print()

    # Sanity check: cat1 + cat2 + cat3 == total
    assert (
        result_df["1_before_spinup"]
        + result_df["2_after_spinup_to_old_cluster"]
        + result_df["3_after_spinup_to_new_cluster"]
        == total_queries_per_replica
    ).all(), "Category counts do not sum to total!"
    console.print("[green]✓[/green] Sanity check passed: categories sum to total for all candidate sizes.")
    console.print()


if __name__ == "__main__":
    main()
