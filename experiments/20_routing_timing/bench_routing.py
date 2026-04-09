"""
bench_routing.py
----------------
Timing micro-experiment for ModelPolicy.score_counterfactual.

Varies:
  - C  = number of eligible clusters       [1, 2, 4, 8, 16]
  - Ac = active queries per cluster         [0, 1, 2, 4, 8, 16, 32]

Fixed:
  - RPU per cluster = 16
  - 10 repetitions per grid point
  - Queries sourced from a real run (data/runs/<run_id>)

Outputs:
  - results/<tag>/data.csv
  - results/<tag>/heatmap.png
  - results/<tag>/lineplot.png
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from autoslo.utils.colors import Palette

# Heatmap colormap: white → light_yellow → dark_orange → dark_red
_TIMING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "timing",
    [
        Palette.white,
        Palette.light_yellow,
        Palette.dark_orange,
        Palette.dark_red,
    ],
)

# ── autoslo imports ──────────────────────────────────────────────────
from autoslo.models.iconq_model import IconqModel
from autoslo.slo.slo_resolver import SloResolver
from autoslo.routing.model_policy import ModelPolicy
from autoslo.routing.managed_cluster_pool import ClusterSnapshot
from autoslo.workload_definition.query import Query, QueryTextId
from autoslo.slo.slo_objective import SloMetric
from autoslo.workload_execution.trace import Trace
from autoslo.blueprints.cluster import Cluster

# ── defaults ─────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "1771539369"
DEFAULT_RUN_ID = "1773030337"
DEFAULT_RPU = 16
DEFAULT_REPS = 10
DEFAULT_SLO_S = 10.0

C_VALUES = [1, 2, 4, 8, 16]
AC_VALUES = [0, 1, 2, 4, 8, 16, 32]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ── helpers ──────────────────────────────────────────────────────────
def _prepare_queries(
    model: IconqModel,
    trace: Trace,
    rpu: int,
    n_queries: int,
) -> list[Query]:
    """Build a pool of fully-featurised Query objects from a real run."""
    qtids = trace.query_text_ids  # Series[query_id → QueryTextId]

    queries: list[Query] = []
    for i, (qid, qtid) in enumerate(qtids.items()):
        if len(queries) >= n_queries:
            break

        feat = model.iconq_query_featurizer.featurize_from_query_text_id(qtid)
        stage_pred = model.stage_model.predict_from_query_text_id(
            {qid: qtid}, cluster_rpu=rpu
        )[qid].overall_mean_s()

        queries.append(
            Query(
                query_id=f"bench_{i:04d}",
                query_text_id=qtid,
                rel_start_time_s=float(i),
                featurization=feat,
                stage_predictions_per_rpu={rpu: stage_pred},
            )
        )

    return queries


def _build_snapshots(
    active_queries: list[Query],
    n_clusters: int,
    rpu: int,
) -> tuple[dict[str, ClusterSnapshot], dict[str, int]]:
    """Build C cluster snapshots, each with Ac queries (round-robin)."""
    snapshots: dict[str, ClusterSnapshot] = {}
    rpus: dict[str, int] = {}

    # Distribute active queries round-robin across clusters.
    per_cluster: dict[str, list[Query]] = defaultdict(list)
    for i, q in enumerate(active_queries):
        cn = f"cluster_{i % n_clusters}"
        per_cluster[cn].append(q)

    cost_per_s = Cluster.cost_per_second_for_rpu(rpu)
    for c in range(n_clusters):
        cn = f"cluster_{c}"
        snapshots[cn] = ClusterSnapshot(
            cluster_name=cn,
            cost_per_second=cost_per_s,
            active_queries=per_cluster.get(cn, []),
            billing_window_start_s=0.0,
        )
        rpus[cn] = rpu

    return snapshots, rpus


def _current_latencies(
    snapshots: dict[str, ClusterSnapshot],
) -> dict[str, float]:
    """Build a current-latencies dict (dummy: stage prediction for each query)."""
    lats: dict[str, float] = {}
    for snap in snapshots.values():
        for q in snap.active_queries:
            # Use the stage prediction as "current" latency estimate.
            for pred in q.stage_predictions_per_rpu.values():
                lats[q.query_id] = pred
                break
    return lats


# ── benchmark ────────────────────────────────────────────────────────
def run_benchmark(
    model_id: str,
    run_id: str,
    rpu: int,
    reps: int,
    slo_s: float,
) -> pd.DataFrame:
    """Run the 2D grid benchmark and return timing results."""
    print(f"Loading model {model_id}...")
    model = IconqModel.load(model_id)

    print(f"Loading trace {run_id}...")
    trace = Trace(run_id)

    # We need at most max(AC_VALUES) * max(C_VALUES) active queries + 1 incoming.
    max_needed = max(AC_VALUES) * max(C_VALUES) + 1
    print(f"Pre-featurising up to {max_needed} queries...")
    query_pool = _prepare_queries(model, trace, rpu, n_queries=max_needed)
    print(f"  → prepared {len(query_pool)} queries")

    policy = ModelPolicy(
        iconq_model_id=model_id,
        default_slo_s=slo_s,
    )

    rows: list[dict] = []
    grid = list(itertools.product(C_VALUES, AC_VALUES))
    total = len(grid) * reps
    done = 0

    for C, Ac in grid:
        # Total active queries = C * Ac, plus 1 incoming.
        n_active = C * Ac
        if n_active + 1 > len(query_pool):
            print(
                f"  SKIP C={C}, Ac={Ac}: need {n_active + 1} queries, have {len(query_pool)}"
            )
            continue

        # The incoming query is the one after all active queries.
        active_qs = query_pool[:n_active]
        incoming = query_pool[n_active]

        snapshots, rpus = _build_snapshots(active_qs, C, rpu)
        lats = _current_latencies(snapshots)

        for rep in range(reps):
            t0 = time.perf_counter()
            policy.score_counterfactual(
                query=incoming,
                arrival_time_s=incoming.rel_start_time_s,
                snapshots=snapshots,
                cluster_rpus=rpus,
                current_latencies=lats,
            )
            elapsed = time.perf_counter() - t0

            rows.append(
                {
                    "C": C,
                    "Ac": Ac,
                    "rep": rep,
                    "time_s": elapsed,
                }
            )

            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"  [{done}/{total}] C={C}, Ac={Ac}, rep={rep}, {elapsed:.4f}s"
                )

    return pd.DataFrame(rows)


# ── plotting ─────────────────────────────────────────────────────────
def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate heatmap + line plots from timing data."""
    agg = (
        df.groupby(["C", "Ac"])["time_s"].agg(["median", "mean"]).reset_index()
    )
    pivot = agg.pivot(index="C", columns="Ac", values="median")

    # ── Heatmap ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(Palette.white)
    ax.set_facecolor(Palette.white)
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap=_TIMING_CMAP,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), color=Palette.black)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), color=Palette.black)
    ax.set_xlabel("Queries per Cluster (Ac)", color=Palette.black)
    ax.set_ylabel("Number of Clusters (C)", color=Palette.black)
    ax.set_title("Routing Time (median, seconds)", color=Palette.black)
    ax.tick_params(colors=Palette.black)

    # Annotate cells: dark text on light cells, white text on dark cells.
    valid = pivot.values[~np.isnan(pivot.values)]
    midpoint = (valid.min() + valid.max()) / 2
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            text_color = Palette.white if val > midpoint else Palette.black
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax, label="seconds")
    cbar.ax.yaxis.label.set_color(Palette.black)
    cbar.ax.tick_params(colors=Palette.black)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'heatmap.png'}")

    # ── Line plots (one line per C, x = Ac) ─────────────────────────
    palette_cycle = Palette.as_list()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(Palette.white)
    ax.set_facecolor(Palette.white)
    q25 = (
        df.groupby(["C", "Ac"])["time_s"]
        .quantile(0.25)
        .reset_index()
        .rename(columns={"time_s": "q25"})
    )
    q75 = (
        df.groupby(["C", "Ac"])["time_s"]
        .quantile(0.75)
        .reset_index()
        .rename(columns={"time_s": "q75"})
    )
    merged = agg.merge(q25, on=["C", "Ac"]).merge(q75, on=["C", "Ac"])

    for idx, (c_val, grp) in enumerate(merged.groupby("C")):
        color = palette_cycle[idx % len(palette_cycle)]
        grp = grp.sort_values("Ac")
        ax.plot(grp["Ac"], grp["median"], "o-", color=color, label=f"C={c_val}")
        ax.fill_between(
            grp["Ac"], grp["q25"], grp["q75"], color=color, alpha=0.15
        )

    ax.set_xlabel("Queries per Cluster (Ac)", color=Palette.black)
    ax.set_ylabel("Time (s)", color=Palette.black)
    ax.set_title(
        "Routing Time vs. Concurrency (by cluster count)", color=Palette.black
    )
    ax.tick_params(colors=Palette.black)
    ax.spines[:].set_color(Palette.gray)
    legend = ax.legend(
        title="Clusters", facecolor=Palette.white, edgecolor=Palette.gray
    )
    legend.get_title().set_color(Palette.black)
    for text in legend.get_texts():
        text.set_color(Palette.black)
    ax.grid(True, color=Palette.light_gray, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "lineplot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'lineplot.png'}")


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Timing micro-experiment for ModelPolicy routing.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"IconQ model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"Source run ID for real queries (default: {DEFAULT_RUN_ID})",
    )
    parser.add_argument(
        "--rpu",
        type=int,
        default=DEFAULT_RPU,
        help=f"RPU size for all clusters (default: {DEFAULT_RPU})",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"Repetitions per grid point (default: {DEFAULT_REPS})",
    )
    parser.add_argument(
        "--slo-s",
        type=float,
        default=DEFAULT_SLO_S,
        help=f"Default SLO in seconds (default: {DEFAULT_SLO_S})",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Output subdirectory name (default: timestamp)",
    )
    parser.add_argument(
        "--load",
        default=None,
        metavar="CSV",
        help="Load existing CSV instead of re-running the benchmark.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from --load CSV (no benchmarking).",
    )
    args = parser.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or run ──────────────────────────────────────────────────
    if args.load or args.plot_only:
        csv_path = Path(args.load) if args.load else out_dir / "data.csv"
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = run_benchmark(
            model_id=args.model_id,
            run_id=args.run_id,
            rpu=args.rpu,
            reps=args.reps,
            slo_s=args.slo_s,
        )
        csv_path = out_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved data → {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    make_plots(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
