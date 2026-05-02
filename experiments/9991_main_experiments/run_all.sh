#!/usr/bin/env bash
# Run every experiment under 9991_main_experiments sequentially.
# Each subdirectory that contains a trial_spec.yml is executed in sorted order.
# The script aborts immediately if any experiment fails (set -e).
#
# Usage (from repo root):
#   bash experiments/9991_main_experiments/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for spec in $(find "$SCRIPT_DIR" -mindepth 2 -maxdepth 2 -name "trial_spec.yml" | sort); do
    exp_spec_path="$(dirname "$spec")"
    exp_name="$(basename "$exp_spec_path")"

    echo ""
    echo "================================================================="
    echo "  Experiment: $exp_name"
    echo "================================================================="

    python src/autoslo/experiments/run_tuning.py "$exp_spec_path"
done

echo ""
echo "================================================================="
echo "  All experiments completed successfully."
echo "================================================================="
