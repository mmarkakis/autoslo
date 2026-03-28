# Experiment 23 — History-window depth

**Question:** How much historical data does the policy tuner need to produce
a good policy for a target day?

## Setup

- **Target day:** May 27 2024 (Monday, ~1 392 queries)
- **Trace file:** `data/__workloads/ext_tpcds1000/redbench_provisioned_157_0.parquet`
  (March 1 – May 29 2024, ~100 K queries)

Three scenarios, varying the history depth used to build the reservoir:

| Scenario     | History window         | Approx. queries | Forecast policy  |
|--------------|------------------------|-----------------|------------------|
| `prev_day`   | May 26                 | ~1 236          | uniform          |
| `prev_week`  | May 20 – May 26        | ~7 333          | recency_weighted |
| `prev_month` | Apr 27 – May 26        | ~32 516         | recency_weighted |

Each run also performs a **holdout evaluation** — simulating the tuned config
on the *real* May 27 workload (extracted from the same trace).

## How to run

```bash
# Full experiment (all three scenarios sequentially):
python experiments/23_history_window/run_experiment.py \
    --config data/__run_configs/test.yml \
    --trace data/__workloads/ext_tpcds1000/redbench_provisioned_157_0.parquet

# Aggregate results into a comparison table + plots:
python experiments/23_history_window/aggregate_results.py \
    --run-dir data/tuner_runs/history_exp
```

## Output

- Per-scenario tuner artifacts: `data/tuner_runs/history_exp/{prev_day,prev_week,prev_month}/`
- Aggregated comparison: `experiments/23_history_window/results/`
  - `comparison.csv` — one row per scenario
  - `holdout_comparison.png` — bar chart of holdout violation + cost
