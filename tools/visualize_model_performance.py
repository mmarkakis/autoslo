"""
Generate performance visualisation plots for one or more trained IconqModels.

Usage
-----
  python tools/visualize_model_performance.py MODEL_ID [MODEL_ID ...]
"""

import argparse

from autoslo.visualizations.iconq_model_performance import plot_all

parser = argparse.ArgumentParser(
    description="Generate performance plots for trained IconqModel(s).",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "model_ids",
    nargs="+",
    metavar="MODEL_ID",
    help="One or more IconqModel IDs.",
)

args = parser.parse_args()

for model_id in args.model_ids:
    print(f"\n=== {model_id} ===")
    plot_all(model_id)
