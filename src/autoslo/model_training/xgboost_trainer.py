import itertools
import multiprocessing as mp
import os
from datetime import datetime, timezone
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import xgboost as xgb
from tqdm.auto import tqdm

import autoslo.utils.paralellism as plu
import autoslo.utils.paths as pu

import argparse

import yaml

from typing import Union

from autoslo.featurization.featurizer import Featurizer


class XGBoostTrainer:

    def run(
        self,
        params,
    ):
        self.model = None
        self.params = params
        self.run_id = params["run_id"]
        self.output_dir = os.path.join(pu.get_models_dir(), self.run_id)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.featurizer = Featurizer.from_name(
            self.params["featurization_params"]["featurizer_name"],
            **self.params["featurization_params"],
        )
        self.feature_names = self.featurizer.feature_names
        self.label_name = self.featurizer.label_name
        self.label_linearizer = (
            np.expm1 if self.featurizer.is_label_in_log_space else lambda x: x
        )

        # Write out the params.
        with open(
            os.path.join(self.output_dir, f"training_params.yml"),
            "w",
        ) as f:
            yaml.dump(self.params, f, sort_keys=False)

        # Process obejective parameters
        objective_params = self.params["model_params"]["objective_params"]
        if objective_params["objective_type"] == "custom":
            self.params["model_params"]["objective_params"]["objective"] = (
                getattr(self, objective_params["objective"])
            )
        self.heavy_side = objective_params.get("heavy_side", 10.0)
        self.light_side = objective_params.get("light_side", 0.5)
        T_low_bare = objective_params.get("low_threshold", 10.0)
        self.T_low = (
            np.log1p(T_low_bare)
            if self.featurizer.is_label_in_log_space
            else T_low_bare
        )
        self.low_w = objective_params.get("low_value_weight", 0.2)
        self.monotonic_rpu = self.params["model_params"].get(
            "monotonic_rpu", False
        )
        self.monotone_constraints = tuple(
            [
                -1 if ((feature_name == "rpu") and self.monotonic_rpu) else 0
                for feature_name in self.featurizer.feature_names
            ]
        )

        # Remove these parameters, if they exist, since they are not valid
        # XGBoost parameters.
        for key in [
            "objective_type",
            "heavy_side",
            "light_side",
            "low_threshold",
            "low_value_weight",
        ]:
            if key in objective_params:
                del self.params["model_params"]["objective_params"][key]

        # Read the data and train the model.
        self.split_dfs = self.read_and_featurize()
        self.train()

        # Generate plots and evaluation metrics.
        self.plot_scatter()
        self.plot_feature_importance()
        self.evaluate()

    def read_and_featurize(self) -> dict:
        """
        Read in the training, validation and test data and featurize them.

        Returns:
            A dictionary mapping split names to featurized DataFrames.
        """
        d = {}
        if self.params["data_params"]["dataset"] == "redset":
            for split_type in ["train", "validation", "test"]:
                split_name = self.params["data_params"][
                    f"{split_type}_split_name"
                ]
                bin_granularity = self.params["data_params"]["bin_granularity"]
                full_split_name = f"{split_name}_{bin_granularity}"

                d[split_type] = self.featurizer.featurize_redset(
                    full_split_name
                )
        else:
            raise ValueError(
                f"Unsupported dataset: {self.params['data_params']['dataset']}"
            )
        return d

    def asymmetric_mse(self, y_true, y_pred):
        """
        Custom asymmetric mean squared error loss function.
        Heavier penalty for underestimation than overestimation.

        Parameters:
            y_true: array-like of true values
            y_pred: array-like of predicted values
        """
        residual = (y_true - y_pred).astype("float")

        grad = np.where(
            residual > 0,
            -2 * residual * self.heavy_side,
            -2 * residual * self.light_side,
        )
        hess = np.where(
            residual > 0,
            2 * self.heavy_side,
            2 * self.light_side,
        )
        return grad, hess

    def asymmetric_thresholded_mse(self, y_true, y_pred):
        """
        Custom asymmetric mean squared error loss function with thresholding.
        Heavier penalty for underestimation than overestimation. For very small
        values

        Parameters:
            y_true: array-like of true values
            y_pred: array-like of predicted values
        """
        base_grad, base_hess = self.asymmetric_mse(y_true, y_pred)

        w_small = self.low_w + (1.0 - self.low_w) * np.minimum(
            1.0, y_true / self.T_low
        )

        grad = w_small * base_grad
        hess = w_small * base_hess
        return grad, hess

    def train(self):
        """
        Train the XGBoost model.
        """

        # Set up the model
        self.model = xgb.XGBRegressor(
            max_depth=self.params["model_params"]["max_depth"],
            n_estimators=self.params["model_params"]["num_boost_rounds"],
            **self.params["model_params"]["objective_params"],
            early_stopping_rounds=10,
            n_jobs=4,
            monotone_constraints=self.monotone_constraints,
        )

        # Train the model
        self.model.fit(
            self.split_dfs["train"][self.feature_names],
            self.split_dfs["train"][self.label_name],
            eval_set=[
                (
                    self.split_dfs["validation"][self.feature_names],
                    self.split_dfs["validation"][self.label_name],
                )
            ],
            verbose=False,
        )

        # Save the model
        self.model.save_model(os.path.join(self.output_dir, f"model.json"))

    def plot_scatter(self):
        """
        Plot scatter plots of true vs predicted values for train, validation and test sets.
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        for i, (split_name, split_df) in enumerate(self.split_dfs.items()):
            ax = axs[i]
            ax.scatter(
                self.label_linearizer(split_df[self.label_name]),
                self.label_linearizer(
                    self.model.predict(split_df[self.feature_names])
                ),
                alpha=0.5,
            )
            ax.set_title(f"{split_name} - {self.run_id}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(
                [0, max(split_df[self.label_name])],
                [0, max(split_df[self.label_name])],
                "r--",
            )

        plt.savefig(
            os.path.join(self.output_dir, f"scatter.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def plot_feature_importance(self):
        """
        Plot feature importance based on gain.
        """
        xgb.plot_importance(self.model, importance_type="gain")
        plt.title(f"Feature Importance - {self.run_id}")
        plt.savefig(
            os.path.join(self.output_dir, f"feature_importance.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def evaluate(self):
        """
        Evaluate the model on train, validation and test sets and save metrics.
        """
        d = {}

        thresholds = [1, 5, 10, 20, 30, 60, 120, 300, 600]

        for split_name, split_df in self.split_dfs.items():

            y_true = np.asarray(
                self.label_linearizer(split_df[self.label_name]), float
            )
            y_pred = np.asarray(
                self.label_linearizer(
                    self.model.predict(split_df[self.feature_names])
                ),
                float,
            )

            # 1) Pinball @ 95 and 99
            diff = y_true - y_pred
            pinball_95 = float(
                np.mean(np.where(diff >= 0, 0.95 * diff, (1 - 0.95) * (-diff)))
            )
            pinball_99 = float(
                np.mean(np.where(diff >= 0, 0.99 * diff, (1 - 0.99) * (-diff)))
            )

            # 2) Coverage
            coverage = float(np.mean(y_true <= y_pred))

            # 3) Miss Depth (multiplicative, only underestimates)
            miss_mask = y_true > y_pred
            miss_depth = (
                float(np.mean((np.log1p(y_true) - np.log1p(y_pred))[miss_mask]))
                if np.any(miss_mask)
                else 0.0
            )

            # 4) MALE (overall closeness, multiplicative)
            male = float(np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred))))

            # 5) Accuracy at over/under prediction, at various thresholds
            threshaccs = {}
            for threshold in thresholds:
                # Count 1 if both prediction and truth are on the same side of
                # the threshold
                acc = float(
                    np.mean(
                        ((y_true >= threshold) & (y_pred >= threshold))
                        | ((y_true < threshold) & (y_pred < threshold))
                    )
                )
                threshaccs[threshold] = acc

                d = d | {
                    f"{split_name}_threshacc@{threshold}": acc,
                }

            # 6) Mean threshold accuracy
            mean_threshacc = float(np.mean(list(threshaccs.values())))

            # 7) MSE
            mse = float(np.mean((y_true - y_pred) ** 2))

            # 8) MAE
            mae = float(np.mean(np.abs(y_true - y_pred)))

            # 9) Asymmetric MSE (10, 0.5)
            asym_mse = float(
                np.mean(
                    np.where(
                        y_true - y_pred > 0,
                        10.0 * (y_true - y_pred) ** 2,
                        0.5 * (y_true - y_pred) ** 2,
                    )
                )
            )

            # 10) Asymmetric Thresholded MSE (3, 1, low threshold 10, low weight 0.2)
            asym_thresh_mse = float(
                np.mean(
                    np.where(
                        y_true - y_pred > 0,
                        3.0 * (y_true - y_pred) ** 2,
                        1.0 * (y_true - y_pred) ** 2,
                    )
                    * np.where(
                        y_true < 10.0,
                        0.2 + 0.8 * (y_true / 10.0),
                        1.0,
                    )
                )
            )

            # 11) Q-error mean, median, p90, p95, p99
            eps = 1e-6
            q_errors = np.where(
                y_true >= y_pred,
                y_true / (y_pred + eps),
                y_pred / (y_true + eps),
            )

            d = d | {
                f"{split_name}_pinball@95": pinball_95,
                f"{split_name}_pinball@99": pinball_99,
                f"{split_name}_coverage": coverage,
                f"{split_name}_miss_depth_log_under": miss_depth,
                f"{split_name}_MALE": male,
                f"{split_name}_mean_threshacc": mean_threshacc,
                f"{split_name}_MSE": mse,
                f"{split_name}_MAE": mae,
                f"{split_name}_asym_mse": asym_mse,
                f"{split_name}_asym_thresh_mse": asym_thresh_mse,
                f"{split_name}_qerror_mean": float(np.mean(q_errors)),
                f"{split_name}_qerror_median": float(np.median(q_errors)),
                f"{split_name}_qerror_p90": float(np.percentile(q_errors, 90)),
                f"{split_name}_qerror_p95": float(np.percentile(q_errors, 95)),
                f"{split_name}_qerror_p99": float(np.percentile(q_errors, 99)),
            }

        # Save metrics
        with open(os.path.join(self.output_dir, f"metrics.yml"), "w") as f:
            yaml.dump(d, f, sort_keys=False)

        return d


def expand_params_node(node: Union[dict, list, str, int, float]) -> list:
    """
    Recursively expand a node of the config tree.

    Returns: list of possibilities for this node.
      - for scalars: [scalar]
      - for dicts:   [dict, dict, ...]
      - for lists:   depends:
          * list of scalars -> [scalar, scalar, ...]
          * list of dicts   -> [dict, dict, ...] (each fully expanded)
    """
    # Case 1: leaf scalar
    if not isinstance(node, (dict, list)):
        return [node]

    # Case 2: dict -> cross product over children
    if isinstance(node, dict):
        keys = list(node.keys())
        expanded_children = {k: expand_params_node(node[k]) for k in keys}

        configs = []
        for combo in itertools.product(*(expanded_children[k] for k in keys)):
            cfg = {}
            for k, v in zip(keys, combo):
                cfg[k] = v
            configs.append(cfg)
        return configs

    # Case 3: list -> alternatives
    if isinstance(node, list):
        if not node:
            return []

        # If it's a list of dicts, treat each dict as a variant subtree
        if all(isinstance(x, dict) for x in node):
            variants = []
            for option in node:
                # each option itself may have sweeps inside
                variants.extend(expand_params_node(option))
            return variants

        # Otherwise, assume list of scalars: direct alternatives
        # (we assume params are never genuinely list-valued)
        return list(node)


def main(config_name: str, force: bool):
    """
    High-level function to train XGBoost models with various parameters.

    Parameters:
        config_name: Name of the YAML configuration file with training
            parameters. Given relative to the top-level config/ directory.
        force: If True, forces re-training of models even if they already exist.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """

    # Determine parameter combinations.
    config_path = os.path.join(pu.get_config_dir(), config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config_params = yaml.safe_load(f)

    # Go through the nested structure to get all combinations of training
    # parameters. Whenever a list is encountered at any level of the nesting,
    # it indicates multiple options for that parameter.
    all_param_combinations = expand_params_node(config_params)
    print(f"Total parameter combinations: {len(all_param_combinations)}")

    # Check which parameter combinations have already been run
    param_combinations = []
    if force:
        print(
            f"The --force flag is set. "
            f"Re-training all {len(all_param_combinations)} models."
        )
        param_combinations = all_param_combinations
    else:
        for training_params in all_param_combinations:
            if not pu.ModelLocator.run_exists(training_params):
                param_combinations.append(training_params)
        print(
            "The --force flag is not set. "
            f"{len(param_combinations)} out of {len(all_param_combinations)} "
            "models will be trained."
        )

    # Create pairs of (run_id, params)
    base_run_id = int(datetime.now(tz=timezone.utc).timestamp())
    for i, param_combination in enumerate(param_combinations):
        param_combination["run_id"] = f"{base_run_id}_{i}"

    # Run in parallel using multiprocessing
    with mp.Pool(plu.deg_of_paralellism()) as pool:
        _ = list(
            tqdm(
                pool.imap_unordered(XGBoostTrainer().run, param_combinations),
                total=len(param_combinations),
            )
        )

    # Trigger computation of summary table
    pu.ModelLocator.get_runs_df()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost models with various parameters."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="xgboost_trainer_config.yml",
        help=(
            "Path to the YAML configuration file with training parameters. "
            "Given relative to the top-level config/ directory."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-training of models even if they already exist.",
    )
    args = parser.parse_args()
    main(args.config_name, args.force)
