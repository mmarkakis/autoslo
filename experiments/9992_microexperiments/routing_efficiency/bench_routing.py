"""
bench_routing.py
----------------
Timing micro-experiment for ``QueryRouter.route_query``.

Varies:
  - C  = number of eligible clusters         [1, 2, 4, 8, 16]
  - Ac = active queries per cluster          [0, 1, 2, 4, 8, 16, 32]

Fixed:
  - RPU per cluster = 16
  - 10 repetitions per grid point
  - Routing policy = USE_ICONQ_MODEL (the production path; the
    other ``QueryRouterPolicy`` values share the same heavy prediction
    code path and only differ in the cheap ``select_best`` step)
  - Queries sourced from a standard workload parquet under
    ``data/workloads/<schema_name>/<workload_name>.parquet`` (same
    pre-population path used by the runner / simulator / policy tuner)

Outputs:
  - ``results/<tag>/data.csv``   — raw per-rep timings
  - ``results/<tag>/heatmap.png`` — median time per (C, Ac) cell
  - ``results/<tag>/lineplot.png`` — median +/- IQR vs. Ac, one line per C
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from autoslo.clusters.cluster import Cluster, ClusterState, ClusterView
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter, QueryRouterPolicy
from autoslo.slo.slo_metric import SloMetric
from autoslo.slo.slo_objective import SloObjective
from autoslo.slo.slo_resolver import SloResolver
from autoslo.utils.colors import Palette
from autoslo.workload_definition.query import Query
from autoslo.workload_definition.workload import Workload


# ── defaults ─────────────────────────────────────────────────
DEFAULT_MODEL_ID = "1771539369"
DEFAULT_SCHEMA_NAME = "ext_tpcds1000"
DEFAULT_WORKLOAD_NAME = "redbench_provisioned_157_0"
DEFAULT_RPU = 16
DEFAULT_REPS = 10
DEFAULT_SLO_S = 10.0

C_VALUES = [1, 2, 4, 8, 16]
AC_VALUES = [0, 1, 2, 4, 8, 16, 32]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

_TIMING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "timing",
    [Palette.white, Palette.light_yellow, Palette.dark_orange, Palette.dark_red],
)


# ── helpers ──────────────────────────────────────────────────────────
def _prepare_queries(
    model: IconqModel,
    schema_name: str,
    workload_name: str,
    rpu_sizes: list[int],
    n_queries: int,
) -> list[Query]:
    """Build a pool of fully-featurised :class:`Query` objects from a
    standard ``workloads/<schema>/<name>.parquet`` file via
    :meth:`Workload.populate_featurizations_and_isolated_predictions` —
    the same pre-population path used by the runner / simulator /
    policy tuner.
    """
    workload = Workload(
        workload_name=workload_name, schema_name=schema_name
    )
    workload.populate_featurizations_and_isolated_predictions(
        iconq_model=model, allowed_rpu_sizes=rpu_sizes
    )
    all_queries = workload.queries()
    # Re-stamp identity and start times so the benchmark is independent of
    # workload timing artefacts and so each query gets a unique query_id.
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
    """Build C ready clusters with active queries distributed round-robin."""
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
) -> pd.DataFrame:
    print(f"Loading IconqModel {model_id}...")
    model = IconqModel.load(model_id)

    print(f"Loading Workload {schema_name}/{workload_name}...")
    max_needed = max(AC_VALUES) * max(C_VALUES) + 1
    print(f"Pre-featurising up to {max_needed} queries...")
    pool = _prepare_queries(
        model,
        schema_name=schema_name,
        workload_name=workload_name,
        rpu_sizes=[rpu],
        n_queries=max_needed,
    )
    print(f"  -> prepared {len(pool)} queries")

    resolver = SloResolver.from_dict(default_slo_s=slo_s, slo_dict={})
    objective = SloObjective(
        slo_metric=SloMetric.RELATIVE, slo_threshold=0.0
    )

    rows: list[dict] = []
    grid = list(itertools.product(C_VALUES, AC_VALUES))
    total = len(grid) * reps
    done = 0

    for C, Ac in grid:
        n_active = C * Ac
        if n_active + 1 > len(pool):
            print(f"  SKIP C={C}, Ac={Ac}: need {n_active + 1} queries")
            continue

        active_qs = pool[:n_active]
        incoming = pool[n_active]
        snapshot = _build_snapshot(active_qs, C, rpu)

        for rep in range(reps):
            # Fresh router per rep so that round-robin / random policies
            # would not carry state across reps (harmless for ICONQ).
            router = QueryRouter(
                slo_resolver=resolver,
                slo_objective=objective,
                routing_policy=QueryRouterPolicy.USE_ICONQ_MODEL,
            )
            t0 = time.perf_counter()
            router.route_query(
                query=incoming,
                snapshot=snapshot,
                iconq_model=model,
                rel_time_s=incoming.rel_start_time_s,
            )
            elapsed = time.perf_counter() - t0
            rows.append({"C": C, "Ac": Ac, "rep": rep, "time_s": elapsed})

            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"  [{done}/{total}] C={C}, Ac={Ac}, rep={rep}, "
                    f"{elapsed:.4f}s"
                )

    return pd.DataFrame(rows)


# ── plotting ─────────────────────────────────────────────────────────
def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    agg = (
        df.groupby(["C", "Ac"])["time_s"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    pivot_mean = agg.pivot(index="C", columns="Ac", values="mean")
    pivot_min = agg.pivot(index="C", columns="Ac", values="min")
    pivot_max = agg.pivot(index="C", columns="Ac", values="max")

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
    ax.set_xlabel("Queries per Cluster (Ac)", color=Palette.black)
    ax.set_ylabel("Number of Clusters (C)", color=Palette.black)
    ax.set_title("route_query Time (mean, seconds)", color=Palette.black)
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
            range_color = Palette.light_gray if mean_val > midpoint else Palette.gray
            lo = pivot_min.values[i, j]
            hi = pivot_max.values[i, j]
            ax.text(
                j, i + 0.15, f"{mean_val:.3f}",
                ha="center", va="center",
                color=text_color, fontsize=8, fontweight="bold",
            )
            ax.text(
                j, i - 0.18, f"{lo:.3f} \u2026 {hi:.3f}",
                ha="center", va="center",
                color=range_color, fontsize=6,
            )

    cbar = fig.colorbar(im, ax=ax, label="seconds")
    cbar.ax.yaxis.label.set_color(Palette.black)
    cbar.ax.tick_params(colors=Palette.black)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'heatmap.png'}")

    # Line plot — median ± IQR, one line per C
    palette_cycle = Palette.as_list()
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

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(Palette.white)
    ax.set_facecolor(Palette.white)
    for idx, (c_val, grp) in enumerate(merged.groupby("C")):
        color = palette_cycle[idx % len(palette_cycle)]
        grp = grp.sort_values("Ac")
        ax.plot(grp["Ac"], grp["median"], "o-", color=color, label=f"C={c_val}")
        ax.fill_between(grp["Ac"], grp["q25"], grp["q75"], color=color, alpha=0.15)
    ax.set_xlabel("Queries per Cluster (Ac)", color=Palette.black)
    ax.set_ylabel("Time (s)", color=Palette.black)
    ax.set_title(
        "route_query Time vs. Concurrency (by cluster count)",
        color=Palette.black,
    )
    ax.tick_params(colors=Palette.black)
    for spine in ax.spines.values():
        spine.set_edgecolor(Palette.gray)
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
        description="Timing micro-experiment for QueryRouter.route_query.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--schema-name", default=DEFAULT_SCHEMA_NAME)
    parser.add_argument("--workload-name", default=DEFAULT_WORKLOAD_NAME)
    parser.add_argument("--rpu", type=int, default=DEFAULT_RPU)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--slo-s", type=float, default=DEFAULT_SLO_S)
    parser.add_argument("--tag", default=None,
                        help="Output subdirectory name (default: timestamp).")
    parser.add_argument("--load", default=None, metavar="CSV",
                        help="Load existing CSV instead of re-running.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only regenerate plots from --load CSV.")
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
        )
        csv_path = out_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved data -> {csv_path}")

    print("\nGenerating plots...")
    make_plots(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
