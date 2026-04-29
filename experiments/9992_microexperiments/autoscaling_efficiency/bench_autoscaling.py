"""
bench_autoscaling.py
--------------------
Timing micro-experiment for ``Autoscaler._select_rpu``.

Varies:
  - R = number of candidate RPU sizes the autoscaler considers
        (drawn as a prefix of ``[4, 8, 16, 32]``)         [1, 2, 3, 4]
  - W = routing-window length (queries replayed)           [1, 2, 4, 8, 16, 32]

Fixed:
  - C  = 4 ready clusters, all at RPU 16
  - Ac = 4 active queries per cluster
  - 10 repetitions per grid point
  - Routing policy = USE_ICONQ_MODEL
  - Queries sourced from a standard workload parquet under
    ``data/workloads/<schema_name>/<workload_name>.parquet`` (same
    pre-population path used by the runner / simulator / policy tuner)

The benchmark seeds the autoscaler's internal window state directly so that a
single ``_select_rpu`` call performs the full counterfactual replay (one
hypothetical cluster per candidate RPU; W routing decisions per replay).

Outputs:
  - ``results/<tag>/data.csv``
  - ``results/<tag>/heatmap.png``
  - ``results/<tag>/lineplot.png``
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
)

from autoslo.clusters.autoscaler import Autoscaler
from autoslo.clusters.cluster import Cluster, ClusterState, ClusterView
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouterPolicy
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.visualizations.colors import Palette
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.workload import Workload

# ── defaults ─────────────────────────────────────────────────
DEFAULT_MODEL_ID = "1771539369"
DEFAULT_SCHEMA_NAME = "ext_tpcds1000"
DEFAULT_WORKLOAD_NAME = "redbench_provisioned_157_0"
DEFAULT_RPU = 16
DEFAULT_REPS = 10
DEFAULT_SLO_S = 10.0
DEFAULT_C = 4
DEFAULT_AC = 4

ALL_RPUS = Cluster.UP_TO_32_RPU_SIZES  # [4, 8, 16, 32]

R_VALUES = [2, 3, 4]
W_VALUES = [1, 2, 4, 8, 16, 32]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

_TIMING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "timing",
    [
        Palette.white,
        Palette.light_yellow,
        Palette.dark_orange,
        Palette.dark_red,
    ],
)


# ── helpers ──────────────────────────────────────────────────────────
def _prepare_queries(
    model: IconqModel,
    schema_name: str,
    workload_name: str,
    rpu_sizes: list[int],
    n_queries: int,
) -> list[Query]:
    """Build :class:`Query` objects from a standard
    ``workloads/<schema>/<name>.parquet`` file via
    :meth:`Workload.populate_featurizations_and_isolated_predictions` so
    each query carries stage predictions for every RPU in ``rpu_sizes`` —
    necessary because the autoscaler may route them to a hypothetical
    cluster of any candidate size during counterfactual replay.
    """
    workload = Workload(workload_name=workload_name, schema_name=schema_name)
    workload.populate_featurizations_and_isolated_predictions(
        iconq_model=model, allowed_rpu_sizes=rpu_sizes
    )
    all_queries = workload.queries()
    pool: list[Query] = []
    for i, q in enumerate(all_queries[:n_queries]):
        pool.append(
            Query(
                query_id=f"bench_{i:04d}",
                query_text_id=q.query_text_id,
                rel_start_time_s=float(i),
                featurization=q.featurization,
                stage_predictions_per_rpu=q.stage_predictions_per_rpu,
            )
        )
    return pool


def _build_snapshot(
    active_queries: list[Query],
    n_clusters: int,
    rpu: int,
    creation_time_s: float = 0.0,
) -> dict[str, ClusterView]:
    """Build C ready clusters, distributing active queries round-robin."""
    per_cluster: dict[str, list[Query]] = defaultdict(list)
    for i, q in enumerate(active_queries):
        cn = f"autoslo-{rpu}-bench-{i % n_clusters}"
        per_cluster[cn].append(q)

    snapshot: dict[str, ClusterView] = {}
    for c_idx in range(n_clusters):
        cn = f"autoslo-{rpu}-bench-{c_idx}"
        cluster = Cluster(
            creation_time_s=creation_time_s,
            rpu=rpu,
            name=cn,
            state=ClusterState.PENDING,
        )
        cluster.update_state(ClusterState.READY)
        accumulated: dict[str, float] = {}
        for q in per_cluster.get(cn, []):
            accumulated[q.query_id] = q.stage_predictions_per_rpu.get(rpu, 5.0)
            cluster.add_query(q, accumulated)
        snapshot[cn] = ClusterView(cluster)
    return snapshot


# ── benchmark ────────────────────────────────────────────────────────
def run_benchmark(
    model_id: str,
    schema_name: str,
    workload_name: str,
    rpu: int,
    reps: int,
    slo_s: float,
    n_clusters: int,
    ac: int,
) -> pd.DataFrame:
    print(f"Loading IconqModel {model_id}...")
    model = IconqModel.load(model_id)

    print(f"Loading Workload {schema_name}/{workload_name}...")
    max_window = max(W_VALUES)
    n_active = n_clusters * ac
    max_needed = max_window + n_active
    print(
        f"Pre-featurising up to {max_needed} queries "
        f"(stage predictions for RPUs {ALL_RPUS})..."
    )
    pool = _prepare_queries(
        model,
        schema_name=schema_name,
        workload_name=workload_name,
        rpu_sizes=ALL_RPUS,
        n_queries=max_needed,
    )
    print(f"  -> prepared {len(pool)} queries")

    active_qs = pool[:n_active]
    window_pool = pool[n_active:]
    snapshot = _build_snapshot(active_qs, n_clusters, rpu)

    resolver = SloResolver.from_dict(default_slo_s=slo_s, slo_dict={})
    objective = SloObjective(slo_metric=SloMetric.RELATIVE, slo_threshold=0.0)

    rows: list[dict] = []
    grid = list(itertools.product(R_VALUES, W_VALUES))
    total = len(grid) * reps
    done = 0

    # All window queries arrive after the pre-existing active queries.
    # The "current time" at which _select_rpu runs is just past the last
    # window query, ensuring no replays terminate the loop early.
    base_time_s = float(n_active)

    for R, W in grid:
        if W > len(window_pool):
            print(f"  SKIP R={R}, W={W}: need {W} window queries")
            continue
        candidate_rpus = ALL_RPUS[:R]
        # Re-time window queries so they all fall inside the observation
        # window starting at base_time_s.
        window_qs = [
            Query(
                query_id=q.query_id,
                query_text_id=q.query_text_id,
                rel_start_time_s=base_time_s + j * 0.001,
                featurization=q.featurization,
                stage_predictions_per_rpu=q.stage_predictions_per_rpu,
            )
            for j, q in enumerate(window_pool[:W])
        ]
        rel_time_s = base_time_s + W * 0.001 + 0.001

        for rep in range(reps):
            autoscaler = Autoscaler(
                slo_resolver=resolver,
                slo_objective=objective,
                allowed_rpu_sizes=candidate_rpus,
                iconq_model=model,
                routing_policy=QueryRouterPolicy.USE_ICONQ_MODEL,
            )
            # Seed internal window state directly: there is no public API
            # for injecting a pre-built window, so this is the only way to
            # measure _select_rpu in isolation.
            autoscaler._window_start_time_s = base_time_s
            autoscaler._snapshot_at_window_start = snapshot
            autoscaler._window_queries = list(window_qs)
            autoscaler._latest_rel_time_s = rel_time_s

            t0 = time.perf_counter()
            autoscaler._select_rpu(rel_time_s)
            elapsed = time.perf_counter() - t0
            rows.append({"R": R, "W": W, "rep": rep, "time_s": elapsed})

            done += 1
            if done % 5 == 0 or done == total:
                print(
                    f"  [{done}/{total}] R={R}, W={W}, rep={rep}, "
                    f"{elapsed:.4f}s"
                )

    return pd.DataFrame(rows)


# ── plotting ─────────────────────────────────────────────────────────
def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        df.groupby(["R", "W"])["time_s"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    pivot_mean = agg.pivot(index="R", columns="W", values="mean")
    pivot_min = agg.pivot(index="R", columns="W", values="min")
    pivot_max = agg.pivot(index="R", columns="W", values="max")

    # Heatmap — colour-encoded by mean; each cell shows mean then min…max
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(Palette.white)
    ax.set_facecolor(Palette.white)
    im = ax.imshow(
        pivot_mean.values, aspect="auto", origin="lower", cmap=_TIMING_CMAP
    )
    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_xticklabels(pivot_mean.columns.astype(int), color=Palette.black)
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_yticklabels(pivot_mean.index.astype(int), color=Palette.black)
    ax.set_xlabel("Window Length (W)", color=Palette.black)
    ax.set_ylabel("# Candidate RPU Sizes (R)", color=Palette.black)
    ax.set_title("_select_rpu Time (mean, seconds)", color=Palette.black)
    ax.tick_params(colors=Palette.black)
    for spine in ax.spines.values():
        spine.set_edgecolor(Palette.gray)

    valid = pivot_mean.values[~np.isnan(pivot_mean.values)]
    midpoint = (valid.min() + valid.max()) / 2 if valid.size else 0
    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            mean_val = pivot_mean.values[i, j]
            if np.isnan(mean_val):
                continue
            text_color = Palette.white if mean_val > midpoint else Palette.black
            range_color = (
                Palette.light_gray if mean_val > midpoint else Palette.gray
            )
            lo = pivot_min.values[i, j]
            hi = pivot_max.values[i, j]
            ax.text(
                j,
                i + 0.15,
                f"{mean_val:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold",
            )
            ax.text(
                j,
                i - 0.18,
                f"{lo:.3f} \u2026 {hi:.3f}",
                ha="center",
                va="center",
                color=range_color,
                fontsize=6,
            )
    cbar = fig.colorbar(im, ax=ax, label="seconds")
    cbar.ax.yaxis.label.set_color(Palette.black)
    cbar.ax.tick_params(colors=Palette.black)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'heatmap.png'}")

    # Line plot
    palette_cycle = Palette.as_list()
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

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(Palette.white)
    ax.set_facecolor(Palette.white)
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
    for spine in ax.spines.values():
        spine.set_edgecolor(Palette.gray)
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
        description="Timing micro-experiment for Autoscaler._select_rpu.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--schema-name", default=DEFAULT_SCHEMA_NAME)
    parser.add_argument("--workload-name", default=DEFAULT_WORKLOAD_NAME)
    parser.add_argument("--rpu", type=int, default=DEFAULT_RPU)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--slo-s", type=float, default=DEFAULT_SLO_S)
    parser.add_argument("--clusters", type=int, default=DEFAULT_C)
    parser.add_argument("--ac", type=int, default=DEFAULT_AC)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--load", default=None, metavar="CSV")
    parser.add_argument("--plot-only", action="store_true")
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
            schema_name=args.schema_name,
            workload_name=args.workload_name,
            rpu=args.rpu,
            reps=args.reps,
            slo_s=args.slo_s,
            n_clusters=args.clusters,
            ac=args.ac,
        )
        csv_path = out_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved data -> {csv_path}")

    print("\nGenerating plots...")
    make_plots(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
