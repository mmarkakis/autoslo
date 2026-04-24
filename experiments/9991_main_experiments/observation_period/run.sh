#!/usr/bin/env bash


set -euo pipefail

SPEC="experiments/9991_main_experiments/observation_period/trial_spec.yml"

echo "=== Step 1: Generate per-scenario configs ==="
python src/autoslo/experiments/generate_trial_configs.py "$SPEC"

echo ""
echo "=== Step 2: Run PolicyTuner for each scenario ==="
python src/autoslo/experiments/run.py "$SPEC"

