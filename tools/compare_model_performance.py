import argparse

from autoslo.models.iconq_model import IconqModel, print_errors_table, DataSplit

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

for model_id in args.model_ids:
    model = IconqModel.load(model_id)
    _, train_errors = model.eval_on_split(split=DataSplit.TRAIN)
    _, val_errors = model.eval_on_split(split=DataSplit.VAL)
    _, test_errors = model.eval_on_split(split=DataSplit.TEST)

    print_errors_table(
        title=f"[bold]{model_id}[/]",
        sets=[
            ("train", train_errors),
            ("val", val_errors),
            ("test", test_errors),
        ],
    )
