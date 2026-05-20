import argparse

from autoslo.models.iconq_model import IconqModel

parser = argparse.ArgumentParser(
    description="Print performance tables for one or more trained IconqModels."
)
parser.add_argument(
    "model_ids",
    nargs="+",
    metavar="MODEL_ID",
    help="One or more IconqModel IDs (subdirectory names under data/iconq_models/).",
)

args = parser.parse_args()
IconqModel.print_performance_tables(args.model_ids)
