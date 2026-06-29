from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from autoslo.filesystem.structured_events import EventType
from autoslo.filesystem.structured_log import StructuredLog


@dataclass(frozen=True)
class CandidateDiagnostics:
    rpu: Optional[int]
    slo_violation: Optional[float]
    projected_cost: Optional[float]

    arrivals_pre_spinup: int
    completions_pre_spinup: int
    completions_of_simulated_pre_spinup: int
    arrivals_post_spinup: int
    completions_post_spinup: int
    completions_of_simulated_post_spinup: int

    fraction_routed_to_hypothetical: float


@dataclass(frozen=True)
class DecisionDiagnostics:
    decision_rel_time_s: float
    autoscaling_policy: str
    selected_rpu: Optional[int]
    reason: str
    candidates: list[CandidateDiagnostics]
    num_queries_in_observation_window: int


def process_log(structured_log_path: Path) -> list[DecisionDiagnostics]:
    """
    Collect statistics about each autoscaler spin-up decision in a structured
    log, along with reconstructed candidate metrics that informed the decision.
    """
    # Read in the log, filter spinup-related events and split into blocks.
    full_df = StructuredLog.load(structured_log_path).df
    spinup_df = full_df[
        full_df["source"].isin({"Autoscaler", "Autoscaler.QueryRouter"})
        & (full_df["event_type"] != "tear_down_decision")
    ].copy()
    is_spin_up_decision = (spinup_df["source"] == "Autoscaler") & (
        spinup_df["event_type"] == EventType.SPIN_UP_DECISION.value
    )
    spinup_df["decision_block_id"] = (
        is_spin_up_decision.shift().fillna(0).cumsum()
    )

    # Map events to originating phase/RPU
    spinup_df["phase"] = spinup_df.apply(
        lambda row: (
            row["details"].get("phase", None)
            if isinstance(row["details"], dict)
            else None
        ),
        axis=1,
    )
    spinup_df["candidate_rpu"] = spinup_df.apply(
        lambda row: (
            int(candidate_rpu)
            if isinstance(row["details"], dict)
            and (candidate_rpu := row["details"].get("candidate_rpu"))
            is not None
            else None
        ),
        axis=1,
    )

    # Save the raw (pre-ffill) candidate_rpu so baseline SIM events
    # (phase="post_spinup", candidate_rpu=NaN) can be isolated later.
    spinup_df["raw_candidate_rpu"] = spinup_df["candidate_rpu"].copy()

    # For "routing_score" and "routing" events, fill in the most recent phase
    # and candidate RPU from the same decision block.
    spinup_df[["phase", "candidate_rpu"]] = spinup_df.groupby(
        "decision_block_id"
    )[["phase", "candidate_rpu"]].ffill()
    spinup_df.loc[
        spinup_df["event_type"].isin({"spin_up_decision", "rpu_selection"}),
        ["phase", "candidate_rpu"],
    ] = None

    # Process each decision block separately and aggregate diagnostics.
    all_decision_diagnostics: list[DecisionDiagnostics] = []
    for block_id, block in spinup_df.groupby("decision_block_id"):
        if not (
            (block["source"] == "Autoscaler")
            & (block["event_type"] == EventType.SPIN_UP_DECISION.value)
        ).any():
            continue
        decision_diagnostics = process_block(block)
        all_decision_diagnostics.append(decision_diagnostics)

    return all_decision_diagnostics


def process_block(block: pd.DataFrame) -> DecisionDiagnostics:
    """
    Process a single decision block of autoscaler events to extract diagnostics
    about the spin-up decision and its associated candidate metrics.
    """
    block = block.copy()

    # Count arrivals in pre-spinup and in each post-spinup candidate phase.
    arrivals_per_phase = {}
    arrivals_per_phase["pre_spinup"] = block[
        (block["event_type"] == EventType.SIM_QUERY_ARRIVAL.value)
        & (block["phase"] == "pre_spinup")
    ].shape[0]

    for candidate_rpu in block["candidate_rpu"].dropna().unique():
        arrivals_per_phase[candidate_rpu] = block[
            (block["event_type"] == EventType.SIM_QUERY_ARRIVAL.value)
            & (block["candidate_rpu"] == candidate_rpu)
            & (block["phase"] == "post_spinup")
        ].shape[0]

    # Count completions in pre-spinup and in each post-spinup candidate phase.
    completions_per_phase = {}
    completions_of_simulated_per_phase = {}
    completions_per_phase["pre_spinup"] = block[
        (block["event_type"] == EventType.SIM_QUERY_COMPLETION.value)
        & (block["phase"] == "pre_spinup")
    ].shape[0]
    completions_of_simulated_per_phase["pre_spinup"] = block[
        (block["event_type"] == EventType.SIM_QUERY_COMPLETION.value)
        & (block["phase"] == "pre_spinup")
        & (block["query_id"].str.startswith("fwd-"))
    ].shape[0]
    for candidate_rpu in block["candidate_rpu"].dropna().unique():
        completions_per_phase[candidate_rpu] = block[
            (block["event_type"] == EventType.SIM_QUERY_COMPLETION.value)
            & (block["candidate_rpu"] == candidate_rpu)
            & (block["phase"] == "post_spinup")
        ].shape[0]
        completions_of_simulated_per_phase[candidate_rpu] = block[
            (block["event_type"] == EventType.SIM_QUERY_COMPLETION.value)
            & (block["candidate_rpu"] == candidate_rpu)
            & (block["phase"] == "post_spinup")
            & (block["query_id"].str.startswith("fwd-"))
        ].shape[0]

    # Compute routing fractions post-spinup per candidate RPU.
    fraction_routed_to_hypothetical_per_rpu = {}
    for candidate_rpu in block["candidate_rpu"].dropna().unique():
        routing_events = block[
            (block["event_type"] == EventType.ROUTING.value)
            & (block["candidate_rpu"] == candidate_rpu)
            & (block["phase"] == "post_spinup")
        ]
        total_routed = routing_events.shape[0]
        if total_routed > 0:
            routed_to_hypothetical = routing_events[
                routing_events["cluster_name"].str.endswith("-hypothetical")
            ].shape[0]
            fraction_routed_to_hypothetical_per_rpu[candidate_rpu] = (
                routed_to_hypothetical / total_routed
            )
    # Extract candidate metrics from "rpu_counterfactual" events in the block.
    slo_violation_per_rpu = {}
    cost_per_rpu = {}
    baseline_slo_violation: Optional[float] = None
    baseline_cost: Optional[float] = None
    cf_rows = block[block["event_type"] == "rpu_counterfactual"]
    for _, row in cf_rows.iterrows():
        if not isinstance(row["details"], dict):
            continue
        rpu = row["details"].get("rpu")
        if rpu is None:
            # no-spinup baseline entry
            v = row["details"].get("slo_violation")
            c = row["details"].get("cost")
            baseline_slo_violation = float(v) if v is not None else None
            baseline_cost = float(c) if c is not None else None
        else:
            slo_violation_per_rpu[rpu] = row["details"].get("slo_violation")
            cost_per_rpu[rpu] = row["details"].get("cost")

    candidate_diagnostics: list[CandidateDiagnostics] = []
    for candidate_rpu in block["candidate_rpu"].dropna().unique():
        candidate_diagnostics.append(
            CandidateDiagnostics(
                rpu=int(candidate_rpu),
                slo_violation=(
                    float(slo_violation_per_rpu[candidate_rpu])
                    if candidate_rpu in slo_violation_per_rpu
                    else None
                ),
                projected_cost=(
                    float(cost_per_rpu[candidate_rpu])
                    if candidate_rpu in cost_per_rpu
                    else None
                ),
                arrivals_pre_spinup=arrivals_per_phase.get("pre_spinup", 0),
                completions_pre_spinup=completions_per_phase.get(
                    "pre_spinup", 0
                ),
                completions_of_simulated_pre_spinup=(
                    completions_of_simulated_per_phase.get("pre_spinup", 0)
                ),
                arrivals_post_spinup=arrivals_per_phase.get(candidate_rpu, 0),
                completions_post_spinup=completions_per_phase.get(
                    candidate_rpu, 0
                ),
                completions_of_simulated_post_spinup=(
                    completions_of_simulated_per_phase.get(candidate_rpu, 0)
                ),
                fraction_routed_to_hypothetical=fraction_routed_to_hypothetical_per_rpu.get(
                    candidate_rpu, 0
                ),
            )
        )

    # Add baseline row if a no-spinup baseline was run.
    # Baseline SIM events are identified by phase="post_spinup" and
    # raw_candidate_rpu=NaN (before the ffill propagated an RPU value into them).
    if baseline_slo_violation is not None or baseline_cost is not None:
        is_baseline_sim = block["raw_candidate_rpu"].isna() & (
            block["phase"] == "post_spinup"
        )
        baseline_arrivals = block[
            (block["event_type"] == EventType.SIM_QUERY_ARRIVAL.value)
            & is_baseline_sim
        ].shape[0]
        baseline_completions = block[
            (block["event_type"] == EventType.SIM_QUERY_COMPLETION.value)
            & is_baseline_sim
        ].shape[0]
        baseline_completions_simulated = block[
            (block["event_type"] == EventType.SIM_QUERY_COMPLETION.value)
            & is_baseline_sim
            & block["query_id"].str.startswith("fwd-")
        ].shape[0]
        candidate_diagnostics.append(
            CandidateDiagnostics(
                rpu=None,
                slo_violation=baseline_slo_violation,
                projected_cost=baseline_cost,
                arrivals_pre_spinup=arrivals_per_phase.get("pre_spinup", 0),
                completions_pre_spinup=completions_per_phase.get("pre_spinup", 0),
                completions_of_simulated_pre_spinup=(
                    completions_of_simulated_per_phase.get("pre_spinup", 0)
                ),
                arrivals_post_spinup=baseline_arrivals,
                completions_post_spinup=baseline_completions,
                completions_of_simulated_post_spinup=baseline_completions_simulated,
                fraction_routed_to_hypothetical=0.0,
            )
        )

    # Build return value.
    decision_row = block[
        (block["source"] == "Autoscaler")
        & (block["event_type"] == EventType.SPIN_UP_DECISION.value)
    ].iloc[0]
    decision_time = decision_row["rel_time_s"]
    autoscaling_policy = decision_row["details"].get(
        "autoscaling_policy", "unknown"
    )
    selected_rpu = decision_row["details"].get("rpu", None)
    reason = decision_row["details"].get("reason", "")
    return DecisionDiagnostics(
        decision_rel_time_s=decision_time,
        autoscaling_policy=autoscaling_policy,
        selected_rpu=int(selected_rpu) if selected_rpu is not None else None,
        reason=reason,
        candidates=candidate_diagnostics,
        num_queries_in_observation_window=block[
            (block["event_type"] == EventType.SIM_QUERY_ARRIVAL.value)
            & (block["query_id"].str.startswith("fwd-0:"))
        ].shape[0],
    )


def _print_summary(
    decisions: list[DecisionDiagnostics], console: Console
) -> None:
    if not decisions:
        console.print("No spin-up decisions found in structured log.")
        return

    for idx, decision in enumerate(decisions, start=1):
        header = (
            f"Decision {idx} @ t={decision.decision_rel_time_s:.3f}s "
            f"(policy={decision.autoscaling_policy or 'unknown'}, "
            f"selected_rpu={decision.selected_rpu})"
        )
        console.rule(header)

        if decision.reason:
            console.print(f"Reason: {decision.reason}")
            console.print(
                f"Num queries in observation window: "
                f"{decision.num_queries_in_observation_window}"
            )

        table = Table(show_header=True, header_style="bold")
        table.add_column("RPU", justify="right")
        table.add_column("SLO Violation", justify="right")
        table.add_column("Projected Cost", justify="right")
        table.add_column("Pre Arrivals", justify="right")
        table.add_column("Post Arrivals", justify="right")
        table.add_column("Pre Completions", justify="right")
        table.add_column("Post Completions", justify="right")
        table.add_column("Fraction Routed to Hypothetical", justify="right")

        for c in sorted(
            decision.candidates,
            key=lambda x: (x.rpu is None, x.rpu if x.rpu is not None else 0),
        ):
            mark = "*" if decision.selected_rpu == c.rpu else ""
            rpu_label = "baseline" if c.rpu is None else str(c.rpu)
            table.add_row(
                f"{rpu_label}{mark}",
                (
                    f"{c.slo_violation:.6f}"
                    if c.slo_violation is not None
                    else "-"
                ),
                (
                    f"{c.projected_cost:.6f}"
                    if c.projected_cost is not None
                    else "-"
                ),
                str(c.arrivals_pre_spinup),
                str(c.arrivals_post_spinup),
                str(c.completions_pre_spinup),
                str(c.completions_post_spinup),
                f"{c.fraction_routed_to_hypothetical:.0%}",
            )

        console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize autoscaler spin-up decision diagnostics from a "
            "structured log and write reconstructed artifacts to the same "
            "directory."
        )
    )
    parser.add_argument(
        "structured_log_path",
        type=str,
        help=(
            "Path to structured_log.parquet (or its containing directory). "
            "Output artifacts are written next to the log."
        ),
    )
    args = parser.parse_args()
    resolved = Path(args.structured_log_path)
    if resolved.is_dir():
        resolved = resolved / "structured_log.parquet"
    if not resolved.exists():
        raise FileNotFoundError(f"No structured log found at {resolved}")

    decisions = process_log(resolved)

    # Print to console.
    console = Console()
    _print_summary(decisions, console)

    # Print to file.
    artifact_path = resolved.parent / "autoscaler_spin_up_diagnostics.json"
    payload = {
        "decisions": [asdict(d) for d in decisions],
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"Wrote diagnostics artifact to {artifact_path}")


if __name__ == "__main__":
    main()
