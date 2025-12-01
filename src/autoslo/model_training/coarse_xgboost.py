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

import yaml

TRAIN_PARAMS = {
    "bin_granularity": ["daily", "hourly"],
    "label": ["duration_s_p95", "duration_s_p99"],
    "num_boost_rounds": [100, 200, 500, 1000],
    "max_depth": [5, 15, 10, 20, 30],
    "objective_params": [
        {"objective": "reg:quantileerror", "quantile_alpha": 0.95},
        {"objective": "reg:quantileerror", "quantile_alpha": 0.99},
        {
            "objective": "custom:asymmetric_mse",
            "heavy_side": 10,
            "light_side": 0.5,
        },
        {
            "objective": "custom:asymmetric_thresholded_mse",
            "heavy_side": 3,
            "light_side": 1,
            "low_threshold": 10.0,
            "low_value_weight": 0.2,
        },
        {
            "objective": "custom:asymmetric_thresholded_mse",
            "heavy_side": 2,
            "light_side": 1,
            "low_threshold": 10.0,
            "low_value_weight": 0.2,
        },
        {
            "objective": "custom:asymmetric_thresholded_mse",
            "heavy_side": 3,
            "light_side": 1,
            "low_threshold": 10.0,
            "low_value_weight": 0.1,
        },
    ],
    "feature_set": {
        "all": [
            "num_queries",
            "nan_cluster_size_num_queries",
            "was_aborted_mean",
            "was_cached_mean",
            "num_permanent_tables_accessed_mean",
            "num_external_tables_accessed_mean",
            "num_system_tables_accessed_mean",
            "mbytes_scanned_mean",
            "mbytes_scanned_p95",
            "mbytes_scanned_p99",
            "num_joins_mean",
            "num_joins_p95",
            "num_joins_p99",
            "num_scans_mean",
            "num_scans_p95",
            "num_scans_p99",
            "num_aggregations_mean",
            "num_aggregations_p95",
            "num_aggregations_p99",
            "query_type_analyze",
            "query_type_copy",
            "query_type_ctas",
            "query_type_delete",
            "query_type_insert",
            "query_type_other",
            "query_type_select",
            "query_type_unload",
            "query_type_update",
            "query_type_vacuum",
            "rpu",
        ],
        "minimal_mean": [
            "num_queries",
            "mbytes_scanned_mean",
            "num_joins_mean",
            "num_scans_mean",
            "num_aggregations_mean",
            "rpu",
        ],
        "minimal_p95": [
            "num_queries",
            "mbytes_scanned_p95",
            "num_joins_p95",
            "num_scans_p95",
            "num_aggregations_p95",
            "rpu",
        ],
        "minimal_p99": [
            "num_queries",
            "mbytes_scanned_p99",
            "num_joins_p99",
            "num_scans_p99",
            "num_aggregations_p99",
            "rpu",
        ],
        "inter_mean": [
            "interarrival_time_s_p1",
            "mbytes_scanned_mean",
            "num_joins_mean",
            "num_scans_mean",
            "num_aggregations_mean",
            "rpu",
        ],
        "inter_p95": [
            "interarrival_time_s_p1",
            "mbytes_scanned_p95",
            "num_joins_p95",
            "num_scans_p95",
            "num_aggregations_p95",
            "rpu",
        ],
        "inter_p99": [
            "interarrival_time_s_p1",
            "mbytes_scanned_p99",
            "num_joins_p99",
            "num_scans_p99",
            "num_aggregations_p99",
            "rpu",
        ],
        "all_nonq": [
            "interarrival_time_s_p1",
            "interarrival_time_s_p5",
            "interarrival_time_s_mean",
            "was_aborted_mean",
            "was_cached_mean",
            "num_permanent_tables_accessed_mean",
            "num_external_tables_accessed_mean",
            "num_system_tables_accessed_mean",
            "mbytes_scanned_mean",
            "mbytes_scanned_p95",
            "mbytes_scanned_p99",
            "num_joins_mean",
            "num_joins_p95",
            "num_joins_p99",
            "num_scans_mean",
            "num_scans_p95",
            "num_scans_p99",
            "num_aggregations_mean",
            "num_aggregations_p95",
            "num_aggregations_p99",
            "query_type_analyze",
            "query_type_copy",
            "query_type_ctas",
            "query_type_delete",
            "query_type_insert",
            "query_type_other",
            "query_type_select",
            "query_type_unload",
            "query_type_update",
            "query_type_vacuum",
            "rpu",
        ],
    },
}


class ModelTrainer:

    def run(
        self,
        training_params,
        run_dir,
    ):

        self.training_params = training_params
        self.bin_granularity = training_params["bin_granularity"]
        self.label = training_params["label"]
        self.num_boost_rounds = training_params["num_boost_rounds"]
        self.max_depth = training_params["max_depth"]
        self.objective_params = training_params["objective_params"]
        self.feature_set_name = training_params["feature_set"]
        self.feature_set = TRAIN_PARAMS["feature_set"][self.feature_set_name]
        self.run_dir = run_dir
        self.model_id = training_params["model_id"]
        self.output_dir = os.path.join(run_dir, self.model_id)
        self.model = None
        self.split_dfs = {}
        self.log_labels = {}

        self.splits_path = os.path.join(
            pu.get_data_path(),
            "redset_byproducts",
            "provisioned",
            f"{self.bin_granularity}_splits",
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Write out the training parameters
        with open(
            os.path.join(
                self.output_dir, f"{self.model_id}_training_params.yml"
            ),
            "w",
        ) as f:
            yaml.dump(training_params, f)

        self.read_and_preprocess()
        self.train()
        self.plot_scatter()
        self.plot_feature_importance()
        eval_metrics = self.evaluate()

        return self.training_params | eval_metrics

    def read_and_preprocess(self):
        """
        Read in the training, validation and test data and preprocess them.
        """
        for split_name in ["train", "validation", "test"]:
            file_path = os.path.join(
                self.splits_path, f"{split_name}_{self.bin_granularity}.parquet"
            )
            pa.set_cpu_count(plu.inner_level_num_cpus())
            df = pd.read_parquet(
                file_path,
                columns=[x for x in self.feature_set if x != "rpu"]
                + [
                    self.label,
                    "unique_cluster_sizes",
                    "unique_cluster_size_count",
                ],
                engine="pyarrow",
            )
            df = df[df["unique_cluster_size_count"] == 1]
            df["cluster_size"] = df["unique_cluster_sizes"].apply(
                lambda x: x[0]
            )
            df["rpu"] = df["cluster_size"] * 8
            self.split_dfs[split_name] = df

            self.log_labels[split_name] = np.log1p(df[self.label])

    def asymmetric_mse(self, y_true, y_pred):
        """
        Custom asymmetric mean squared error loss function.
        Heavier penalty for underestimation than overestimation.

        Parameters:
            y_true: array-like of true values
            y_pred: array-like of predicted values
        """
        residual = (y_true - y_pred).astype("float")

        heavy = self.objective_params.get("heavy_side", 10)
        light = self.objective_params.get("light_side", 0.5)

        grad = np.where(
            residual > 0,
            -2 * residual * heavy,
            -2 * residual * light,
        )
        hess = np.where(
            residual > 0,
            2 * heavy,
            2 * light,
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

        T_low = self.objective_params.get("low_threshold", 10.0)
        T_low_logspace = np.log1p(T_low)
        low_w = self.objective_params.get("low_value_weight", 0.2)

        w_small = low_w + (1.0 - low_w) * np.minimum(
            1.0, y_true / T_low_logspace
        )

        grad = w_small * base_grad
        hess = w_small * base_hess
        return grad, hess

    def train(self):
        """
        Train the XGBoost model.
        """

        # Set up the model
        obj_kwargs = {}
        if self.objective_params["objective"].startswith("custom:"):
            obj_name = self.objective_params["objective"].split(":")[1]
            obj_kwargs["objective"] = getattr(self, obj_name)
        else:
            obj_kwargs = self.objective_params
        self.model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            n_estimators=self.num_boost_rounds,
            **obj_kwargs,
            early_stopping_rounds=10,
            n_jobs=4,
        )

        # Train the model
        self.model.fit(
            self.split_dfs["train"][self.feature_set],
            self.log_labels["train"],
            eval_set=[
                (
                    self.split_dfs["validation"][self.feature_set],
                    self.log_labels["validation"],
                )
            ],
            verbose=False,
        )

        # Save the model
        self.model.save_model(
            os.path.join(self.output_dir, f"{self.model_id}_model.json")
        )

    def plot_scatter(self):
        """
        Plot scatter plots of true vs predicted values for train, validation and test sets.
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        for i, (split_name, split_df) in enumerate(self.split_dfs.items()):
            ax = axs[i]
            ax.scatter(
                np.expm1(self.log_labels[split_name]),
                np.expm1(self.model.predict(split_df[self.feature_set])),
                alpha=0.5,
            )
            ax.set_title(f"{split_name} - {self.model_id}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.plot(
                [0, max(split_df[self.label])],
                [0, max(split_df[self.label])],
                "r--",
            )

        plt.savefig(
            os.path.join(self.output_dir, f"{self.model_id}_scatter.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def plot_feature_importance(self):
        """
        Plot feature importance based on gain.
        """
        xgb.plot_importance(self.model, importance_type="gain")
        plt.title(f"Feature Importance - {self.model_id}")
        plt.savefig(
            os.path.join(
                self.output_dir, f"{self.model_id}_feature_importance.png"
            ),
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
            y_true_log = self.log_labels[split_name]
            y_pred_log = self.model.predict(split_df[self.feature_set])

            # Convert back to original scale
            y_true = np.asarray(np.expm1(y_true_log), float)
            y_pred = np.asarray(np.expm1(y_pred_log), float)
            eps = 1e-12

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
                float(np.mean((y_true_log - y_pred_log)[miss_mask]))
                if np.any(miss_mask)
                else 0.0
            )

            # 4) MALE (overall closeness, multiplicative)
            male = float(np.mean(np.abs(y_true_log - y_pred_log)))

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

            d = d | {
                f"{split_name}_pinball@95": pinball_95,
                f"{split_name}_pinball@99": pinball_99,
                f"{split_name}_coverage": coverage,
                f"{split_name}_miss_depth_log_under": miss_depth,
                f"{split_name}_MALE": male,
                f"{split_name}_mean_threshacc": mean_threshacc,
            }

        # Save metrics
        with open(
            os.path.join(self.output_dir, f"{self.model_id}_metrics.yml"), "w"
        ) as f:
            yaml.dump(d, f)

        return d


if __name__ == "__main__":

    run_id = int(datetime.now(tz=timezone.utc).timestamp())
    run_dir = os.path.join(pu.get_data_path(), "models", f"{run_id}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # Determine parameter combinations.
    keys = list(TRAIN_PARAMS.keys())
    values = list(TRAIN_PARAMS.values())

    param_combinations = [
        dict(zip(keys, combo)) for combo in itertools.product(*values)
    ]

    # Create pairs of (model_id, training_params)
    for i, param_combination in enumerate(param_combinations):
        param_combination["model_id"] = f"model_{i}"

    # Run in parallel using multiprocessing
    run_one = partial(
        ModelTrainer().run,
        run_dir=run_dir,
    )

    with mp.Pool(plu.deg_of_paralellism()) as pool:
        metrics = list(
            tqdm(
                pool.imap_unordered(run_one, param_combinations),
                total=len(param_combinations),
            )
        )

    # Save the metrics as a dataframe to a parquet file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_parquet(
        os.path.join(run_dir, "all_model_metrics.parquet"),
        index=False,
    )
