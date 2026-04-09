"""
bench_headroom.py
-----------------
Timing micro-experiment for HeadroomPolicy._select_rpu.

Varies:
  - R  = number of candidate RPU sizes        [1, 2, 4, 8]
  - W  = routing-window length (queries)       [1, 2, 4, 8, 16, 32]

Fixed:
  - C  = 4 clusters (each at RPU 16)
  - Ac = 4 active queries per cluster
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
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from autoslo.blueprints.cluster import Cluster
from autoslo.capacity.headroom_policy import HeadroomPolicy

# ── autoslo imports ──────────────────────────────────────────────────
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.model_policy import ModelPolicy
from autoslo.routing.managed_cluster_pool import ClusterSnapshot
from autoslo.slo.slo_objective import SloMetric
from autoslo.slo.slo_resolver import SloResolver
from autoslo.workload_definition.query import Query, QueryTextId
from autoslo.workload_execution.trace import Trace

# ── defaults ─────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "1771539369"
DEFAULT_RUN_ID = "1773030337"
DEFAULT_RPU = 16
DEFAULT_REPS = 10
DEFAULT_SLO_S = 10.0
DEFAULT_C = 4
DEFAULT_AC = 4

# All RPU sizes our system allows. R candidates are drawn from this list.
ALL_RPUS = [4, 8, 16, 32]

R_VALUES = [1, 2, 4, 8]
W_VALUES = [1, 2, 4, 8, 16, 32]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


# ── helpers ──────────────────────────────────────────────────────────
def _prepare_queries(
    model: IconqModel,
    trace: Trace,
    rpu_sizes: list[int],
    n_queries: int,
) -> list[Query]:
    """Build fully-featurised Query objects with stage predictions for
    *all* RPU sizes in ``rpu_sizes``."""
    qtids = trace.query_text_ids

    queries: list[Query] = []
    for i, (qid, qtid) in enumerate(qtids.items()):
        if len(queries) >= n_queries:
            break

        feat = model.iconq_query_featurizer.featurize_from_query_text_id(qtid)
        stage_preds: dict[int, float] = {}
        for rpu in rpu_sizes:
            stage_preds[rpu] = model.stage_model.predict_from_query_text_id(
                {qid: qtid}, cluster_rpu=rpu
            )[qid].overall_mean_s()

        queries.append(
            Query(
                query_id=f"bench_{i:04d}",
                query_text_id=qtid,
                rel_start_time_s=float(i),
                featurization=feat,
                stage_predictions_per_rpu=stage_preds,
            )
        )

    return queries


def _build_snapshots(
    active_queries: list[Query],
    n_clusters: int,
    rpu: int,
) -> dict[str, ClusterSnapshot]:
    """Build C cluster snapshots, distributing Ac queries round-robin."""
    per_cluster: dict[str, list[Query]] = defaultdict(list)
    for i, q in enumerate(active_queries):
        cn = f"cluster_{i % n_clusters}"
        per_cluster[cn].append(q)

    cost_per_s = Cluster.cost_per_second_for_rpu(rpu)
    snapshots: dict[str, ClusterSnapshot] = {}
    for c in range(n_clusters):
        cn = f"cluster_{c}"
        snapshots[cn] = ClusterSnapshot(
            cluster_name=cn,
            cost_per_second=cost_per_s,
            active_queries=per_cluster.get(cn, []),
            billing_window_start_s=0.0,
        )
    return snapshots


def _build_initial_latencies(
    snapshots: dict[str, ClusterSnapshot],
    rpu: int,
) -> dict[str, float]:
    """Return predicted latencies for every active query in every snapshot."""
    lats: dict[str, float] = {}
    for snap in snapshots.values():
        for q in snap.active_queries:
            lats[q.query_id] = q.stage_predictions_per_rpu.get(rpu, 5.0)
    return lats


def _make_mock_pool(snapshots: dict[str, ClusterSnapshot], rpu: int):
    """Create a minimal mock pool that satisfies _counterfactual_replay's
    self._pool.get_rpu(cn) calls."""
    pool = MagicMock()
    pool.get_rpu.side_effect = lambda cn: rpu
    return pool


def _build_routing_window(
    window_queries: list[Query],
    snapshots: dict[str, ClusterSnapshot],
    rpu: int,
) -> deque:
    """Build the _routing_window deque that HeadroomPolicy expects.

    Each entry: (query, predicted_latency_s, routed_at_s, snapshot | None)
    """
    window = deque()
    for q in window_queries:
        pred_lat = q.stage_predictions_per_rpu.get(rpu, 5.0)
        arrival = q.rel_start_time_s
        window.append((q, pred_lat, arrival, None))
    return window


def _candidate_rpus_for_r(r: int) -> list[int]:
    """Return *r* candidate RPU sizes from ALL_RPUS.
    We always include the full list prefix so smaller RPUs are tried first."""
    return ALL_RPUS[:r]


# ── benchmark ────────────────────────────────────────────────────────
def run_benchmark(
    model_id: str,
    run_id: str,
    rpu: int,
    reps: int,
    slo_s: float,
    n_clusters: int,
    ac: int,
) -> pd.DataFrame:
    """Run the 2D grid benchmark (R × W) and return timing results."""
    print(f"Loading model {model_id}...")
    model = IconqModel.load(model_id)

    print(f"Loading trace {run_id}...")
    trace = Trace(run_id)

    # We need max(W) window queries + C*Ac active queries.
    max_window = max(W_VALUES)
    max_active = n_clusters * ac
    max_needed = max_window + max_active
    print(f"Pre-featurising up to {max_needed} queries (all RPU sizes)...")
    query_pool = _prepare_queries(model, trace, ALL_RPUS, n_queries=max_needed)
    print(f"  → prepared {len(query_pool)} queries")

    # Build cluster snapshots (fixed across all grid points).
    active_qs = query_pool[:max_active]
    snapshots = _build_snapshots(active_qs, n_clusters, rpu)
    initial_lats = _build_initial_latencies(snapshots, rpu)

    # Remaining queries are used as the window.
    window_pool = query_pool[max_active:]

    # Routing policy for counterfactual scoring.
    routing_policy = ModelPolicy(
        iconq_model_id=model_id,
        default_slo_s=slo_s,
    )

    resolver = SloResolver.from_dict(default_slo_s=slo_s, slo_dict={})
    mock_pool = _make_mock_pool(snapshots, rpu)

    rows: list[dict] = []
    grid = list(itertools.product(R_VALUES, W_VALUES))
    total = len(grid) * reps
    done = 0

    for R, W in grid:
        if W > len(window_pool):
            print(
                f"  SKIP R={R}, W={W}: need {W} window queries, have {len(window_pool)}"
            )
            continue

        candidate_rpus = _candidate_rpus_for_r(R)
        window_qs = window_pool[:W]

        for rep in range(reps):
            # Construct a fresh HeadroomPolicy for each rep to reset state.
            hp = HeadroomPolicy(
                slo_resolver=resolver,
                slo_metric=SloMetric.RELATIVE,
                allowed_rpu_sizes=candidate_rpus,
                iconq_model=model,
                routing_policy=routing_policy,
                slo_threshold=0.0,
            )

            # Inject internal state to enable counterfactual replay.
            hp._pool = mock_pool
            hp._routing_window = _build_routing_window(
                window_qs, snapshots, rpu
            )
            hp._window_initial_snapshots = dict(snapshots)
            hp._window_initial_latencies = dict(initial_lats)

            current_time_s = float(W + max_active + 10)

            t0 = time.perf_counter()
            hp._select_rpu(current_time_s)
            elapsed = time.perf_counter() - t0

            rows.append(
                {
                    "R": R,
                    "W": W,
                    "rep": rep,
                    "time_s": elapsed,
                }
            )

            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"  [{done}/{total}] R={R}, W={W}, rep={rep}, {elapsed:.4f}s"
                )

    return pd.DataFrame(rows)


# ── plotting ─────────────────────────────────────────────────────────
def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate heatmap + line plots from timing data."""
    agg = df.groupby(["R", "W"])["time_s"].agg(["median", "mean"]).reset_index()
    pivot = agg.pivot(index="R", columns="W", values="median")

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
    ax.set_xlabel("Window Length (W)", color=Palette.black)
    ax.set_ylabel("Candidate RPU Sizes (R)", color=Palette.black)
    ax.set_title("_select_rpu Time (median, seconds)", color=Palette.black)
    ax.tick_params(colors=Palette.black)

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

    # ── Line plots (one line per R, x = W) ──────────────────────────
    palette_cycle = Palette.as_list()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(Palette.white)
    ax.set_facecolor(Palette.white)
    q25 = (
        df.groupby(["R", "W"])["time_s"]
        .quantile(0.25)
        .reset_index()
        .rename(columns={"time_s": "q25"})
    )
    q75 = (
        df.groupby(["R", "W"])["time_s"]
        .quantile(0.75)
        .reset_index()
        .rename(columns={"time_s": "q75"})
    )
    merged = agg.merge(q25, on=["R", "W"]).merge(q75, on=["R", "W"])

    for idx, (r_val, grp) in enumerate(merged.groupby("R")):
        color = palette_cycle[idx % len(palette_cycle)]
        grp = grp.sort_values("W")
        ax.plot(grp["W"], grp["median"], "o-", color=color, label=f"R={r_val}")
        ax.fill_between(
            grp["W"], grp["q25"], grp["q75"], color=color, alpha=0.15
        )

    ax.set_xlabel("Window Length (W)", color=Palette.black)
    ax.set_ylabel("Time (s)", color=Palette.black)
    ax.set_title(
        "_select_rpu Time vs. Window Length (by # candidate RPUs)",
        color=Palette.black,
    )
    ax.tick_params(colors=Palette.black)
    ax.spines[:].set_color(Palette.gray)
    legend = ax.legend(
        title="# RPU candidates",
        facecolor=Palette.white,
        edgecolor=Palette.gray,
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
        description="Timing micro-experiment for HeadroomPolicy._select_rpu.",
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
        help=f"RPU for fixed clusters (default: {DEFAULT_RPU})",
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
        "--clusters",
        type=int,
        default=DEFAULT_C,
        help=f"Number of clusters (fixed, default: {DEFAULT_C})",
    )
    parser.add_argument(
        "--ac",
        type=int,
        default=DEFAULT_AC,
        help=f"Active queries per cluster (fixed, default: {DEFAULT_AC})",
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
            n_clusters=args.clusters,
            ac=args.ac,
        )
        csv_path = out_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved data → {csv_path}")

    print("\nGenerating plots...")
    make_plots(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
