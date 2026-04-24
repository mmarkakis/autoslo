#!/usr/bin/env bash
# Run every experiment under 9991_main_experiments sequentially.
# Each subdirectory that contains a run.sh is executed in sorted order.
# The script aborts immediately if any experiment fails (set -e).
#
# Usage (from repo root):
#   bash experiments/9991_main_experiments/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for run_sh in $(find "$SCRIPT_DIR" -mindepth 2 -maxdepth 2 -name "run.sh" | sort); do
    exp_dir="$(dirname "$run_sh")"
    exp_name="$(basename "$exp_dir")"

    echo ""
    echo "================================================================="
    echo "  Experiment: $exp_name"
    echo "================================================================="

    bash "$run_sh"
done

echo ""
echo "================================================================="
echo "  All experiments completed successfully."
echo "================================================================="
