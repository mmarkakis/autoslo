# Legacy Microexperiments (Archival)

This directory is retained for historical reference only.

The scripts here predate the current API and experiment layout and are not
considered the authoritative microbenchmark workflow.

## Current workflow

Use the dedicated microbenchmark manifests under:

- `data/manifests/microbench/`

Run via:

```bash
python src/autoslo/entry_points/microbench.py \
  --microbench_manifest_path routing_efficiency
```

Examples:

```bash
python src/autoslo/entry_points/microbench.py \
  --microbench_manifest_path autoscaling_efficiency

python src/autoslo/entry_points/microbench.py \
  --microbench_manifest_path tuner_efficiency
```

Outputs are written to:

- `data/microbenchmark_runs/<microbenchmark>/<run_tag>/`
- `data/plots/microbench_<microbenchmark>/<run_tag>/`
