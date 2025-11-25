import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import autoslo.utils.colors as cu
import autoslo.utils.paths as pu

BIN_OPTIONS = {
    "hourly": {
        "suffix": "hourly",
        "bin_col_name": "hour_bin",
        "bin_size": timedelta(hours=1),
    },
    "daily": {
        "suffix": "daily",
        "bin_col_name": "day_bin",
        "bin_size": timedelta(days=1),
    },
}


class Splitter:

    FRAC_QUERIES_WITH_NAN_THRESHOLD = 0.5
    FRAC_DURATION_WITH_NAN_THRESHOLD = 0.1
    MIN_UNIQUE_DAYS = 28
    RANDOM_SEED = 42
    RESERVED_FRACTION = 0.2
    TEST_TRAILING_DAYS = 7
    VALIDATION_TRAILING_DAYS = 7

    def __init__(self, dir: str):
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

        self.helper_dir = os.path.join(self.dir, "split_info")
        os.makedirs(self.helper_dir, exist_ok=True)

        self.instances_with_many_nan_sizes: set[int] = set()
        self.short_lifetime_instances: set[int] = set()
        self.reserved_test_instances: set[int] = set()

        self.dfs: dict[str, pd.DataFrame] = {}
        for bin_key, bin_option in BIN_OPTIONS.items():
            suffix = bin_option["suffix"]
            df_path = os.path.join(self.dir, f"all_{suffix}.parquet")
            self.dfs[bin_key] = pd.read_parquet(df_path)

    def run(self):

        self.determine_instances_with_many_nan_sizes()
        self._remove_instances_in_list(
            self.instances_with_many_nan_sizes, "instances_with_many_nan_sizes"
        )

        self.determine_short_lifetime_instances()
        self._remove_instances_in_list(
            self.short_lifetime_instances, "short_lifetime_instances"
        )

        self.determine_reserved_test_instances()
        self._remove_instances_in_list(
            self.reserved_test_instances, "reserved_test_instances"
        )

        self.determine_splits()

    def _remove_instances_in_list(
        self, instance_id_list: list[int], label: str
    ):
        # Write the instance ids out to a text file.
        out_path = os.path.join(self.helper_dir, f"{label}.txt")
        with open(out_path, "w") as f:
            f.writelines(f"{instance_id}\n" for instance_id in instance_id_list)

        # Write out the data for these instances and remove these instances from
        # the main dataframes.
        for df_name, df in self.dfs.items():
            # Write out the data for these instances.
            out_path = os.path.join(
                self.helper_dir,
                f'{label}_{BIN_OPTIONS[df_name]["suffix"]}.parquet',
            )
            df_banned = df[df["instance_id"].isin(instance_id_list)]
            df_banned.to_parquet(out_path)

            # Remove these instances from the main dataframe.
            unique_instance_ids_before = set(
                df["instance_id"].unique().tolist()
            )
            self.dfs[df_name] = df[~df["instance_id"].isin(instance_id_list)]
            unique_instance_ids_after = set(
                self.dfs[df_name]["instance_id"].unique().tolist()
            )
            removed_instance_ids = (
                unique_instance_ids_before - unique_instance_ids_after
            )
            print(
                f"\tRemoved {len(removed_instance_ids)} instances from "
                f"{df_name} ({len(unique_instance_ids_before)} -> "
                f"{len(unique_instance_ids_after)})"
            )

    def determine_instances_with_many_nan_sizes(self):

        # Per instance, aggregate statistics about the queries where
        # cluster_size is nan.
        nan_stats_df = (
            self.dfs["daily"]
            .groupby("instance_id")
            .agg(
                num_queries=("num_queries", "sum"),
                duration_s_sum=("duration_s_sum", "sum"),
                nan_cluster_size_num_queries=(
                    "nan_cluster_size_num_queries",
                    "sum",
                ),
                nan_cluster_size_duration_s_sum=(
                    "nan_cluster_size_duration_s_sum",
                    "sum",
                ),
            )
        )
        nan_stats_df["frac_queries_with_nan"] = (
            nan_stats_df["nan_cluster_size_num_queries"]
            / nan_stats_df["num_queries"]
        )
        nan_stats_df["frac_duration_with_nan"] = (
            nan_stats_df["nan_cluster_size_duration_s_sum"]
            / nan_stats_df["duration_s_sum"]
        )

        # Determine the instances where more than half the queries have nan 
        # cluster size.
        condition = (
            nan_stats_df["frac_queries_with_nan"]
            > self.FRAC_QUERIES_WITH_NAN_THRESHOLD
        ) | (
            nan_stats_df["frac_duration_with_nan"]
            > self.FRAC_DURATION_WITH_NAN_THRESHOLD
        )
        self.instances_with_many_nan_sizes = set(
            nan_stats_df[condition].index.tolist()
        )
        print(
            f"Determined {len(self.instances_with_many_nan_sizes)} instances "
            "with many missing cluster sizes."
        )

        # Plot the instances with many nan sizes.
        plt.figure(figsize=(6, 6))
        N_removed = len(self.instances_with_many_nan_sizes)
        N_allowed = len(nan_stats_df) - N_removed
        sns.scatterplot(
            x="frac_queries_with_nan",
            y="frac_duration_with_nan",
            data=nan_stats_df[~condition],
            color=cu.Palette.dark_blue,
            label=f"Allowed Instances (N={N_allowed})",
            alpha=0.5,
        )
        # Highlight banned instances in red
        sns.scatterplot(
            x="frac_queries_with_nan",
            y="frac_duration_with_nan",
            data=nan_stats_df[condition],
            color=cu.Palette.dark_red,
            label=f"Removed Instances (N={N_removed})",
            alpha=0.5,
        )
        plt.xlabel("Fraction of Instance Queries where `cluster_size` is NaN")
        plt.ylabel(
            "Fraction of Total Duration of Instance Queries\nTaken by Queries "
            "where `cluster_size` is NaN"
        )
        plt.title(
            "We exclude instances with many missing `cluster_size` values"
        )

        # Shade the region where instances are not banned
        plt.fill_betweenx(
            y=[0, 0.1], x1=0, x2=0.5, color=cu.Palette.light_gray, alpha=0.5
        )

        plt.grid()
        out_path = os.path.join(self.helper_dir, "banned_instances.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    def determine_short_lifetime_instances(self):
        # Are there instances where the unique days with query submissions are 
        # fewer than MIN_UNIQUE_DAYS
        lifetime_stats_df = (
            self.dfs["daily"]
            .groupby("instance_id")
            .agg(
                first_bin=(BIN_OPTIONS["daily"]["bin_col_name"], "min"),
                last_bin=(BIN_OPTIONS["daily"]["bin_col_name"], "max"),
            )
        )
        lifetime_stats_df["num_unique_days"] = (
            lifetime_stats_df["last_bin"] - lifetime_stats_df["first_bin"]
        ).dt.days + 1

        lifetime_stats_df = lifetime_stats_df.sort_values("num_unique_days")
        self.short_lifetime_instances = set(
            lifetime_stats_df[
                lifetime_stats_df["num_unique_days"] < self.MIN_UNIQUE_DAYS
            ].index.tolist()
        )
        print(
            f"Determined {len(self.short_lifetime_instances)} instances with "
            f"short lifetimes (under {self.MIN_UNIQUE_DAYS} days)."
        )

        # Plot the instances with short lifetimes.
        plt.figure(figsize=(6, 6))
        instance_lifetime_cdf = (
            lifetime_stats_df["num_unique_days"]
            .sort_values()
            .reset_index(drop=True)
        )
        min_above = instance_lifetime_cdf[
            instance_lifetime_cdf >= self.MIN_UNIQUE_DAYS
        ].min()

        N_removed = len(self.short_lifetime_instances)
        N_allowed = len(lifetime_stats_df) - N_removed
        plt.plot(
            instance_lifetime_cdf[
                instance_lifetime_cdf >= self.MIN_UNIQUE_DAYS
            ],
            np.linspace(0, 1, len(instance_lifetime_cdf))[
                instance_lifetime_cdf >= self.MIN_UNIQUE_DAYS
            ],
            linestyle="-",
            linewidth=2,
            color=cu.Palette.dark_blue,
            alpha=0.5,
            label=f"Allowed Instances (N={N_allowed})",
        )

        # Plot the points before MIN_UNIQUE_DAYS in red
        plt.plot(
            instance_lifetime_cdf[instance_lifetime_cdf <= min_above],
            np.linspace(0, 1, len(instance_lifetime_cdf))[
                instance_lifetime_cdf <= min_above
            ],
            linestyle="-",
            linewidth=2,
            color=cu.Palette.dark_red,
            alpha=0.5,
            label=f"Removed Instances (N={N_removed})",
        )

        plt.xlabel("Number of Unique Days with Queries")
        plt.ylabel("Fraction of Instances")
        plt.title("CDF of Instance Lifetimes (in days)")

        # Shade the area from 29 days onwards.
        plt.fill_betweenx(
            y=[0, 1],
            x1=29,
            x2=instance_lifetime_cdf.max(),
            color=cu.Palette.light_gray,
            alpha=0.5,
        )

        plt.legend()

        plt.grid()
        out_path = os.path.join(self.helper_dir, "short_lifetime_instances.png")
        plt.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def determine_reserved_test_instances(self):
        # Reserve some instances for testing. W
        unique_instance_ids = self.dfs["daily"]["instance_id"].unique()
        np.random.seed(self.RANDOM_SEED)
        self.reserved_test_instances = set(
            np.random.choice(
                unique_instance_ids,
                size=int(self.RESERVED_FRACTION * len(unique_instance_ids)),
                replace=False,
            ).tolist()
        )
        print(
            f"Determined {len(self.reserved_test_instances)} reserved test "
            f"instances ({self.RESERVED_FRACTION * 100:.1f}% of total)."
        )

    def determine_splits(self):
        val_start_points = {}
        test_start_points = {}

        # For each instance, determine the start of the validation and test 
        # periods.
        for instance_id, df_instance in self.dfs["daily"].groupby(
            "instance_id"
        ):
            max_bin = df_instance[BIN_OPTIONS["daily"]["bin_col_name"]].max()
            test_start_points[instance_id] = (
                max_bin.floor("D")
                - timedelta(days=self.TEST_TRAILING_DAYS)
                + timedelta(days=1)
            )
            val_start_points[instance_id] = test_start_points[
                instance_id
            ] - timedelta(days=self.VALIDATION_TRAILING_DAYS)

        # Now, for each dataframe, assign splits based on the above start 
        # points.
        for df_name, df in self.dfs.items():
            bin_col_name = BIN_OPTIONS[df_name]["bin_col_name"]

            def assign_split(row):
                bin_day = row[bin_col_name].floor("D")
                instance_id = row["instance_id"]
                if bin_day >= test_start_points[instance_id]:
                    return "test"
                elif bin_day >= val_start_points[instance_id]:
                    return "validation"
                else:
                    return "train"

            df["split"] = df.apply(assign_split, axis=1)

            # Write out each split separately.
            out_dir = os.path.join(
                self.dir,
                f"{df_name}_splits",
            )
            os.makedirs(out_dir, exist_ok=True)
            for split in ["train", "validation", "test"]:
                out_path = os.path.join(
                    out_dir,
                    f"{split}_{BIN_OPTIONS[df_name]['suffix']}.parquet",
                )
                df_split = df[df["split"] == split]
                df_split.to_parquet(out_path, index=False)
                print(
                    f"Wrote out {len(df_split)} rows for {df_name} {split} "
                    f"to {out_path}."
                )


if __name__ == "__main__":
    dir = os.path.join(pu.get_data_path(), "redset_byproducts", "provisioned")
    splitter = Splitter(dir)
    splitter.run()
