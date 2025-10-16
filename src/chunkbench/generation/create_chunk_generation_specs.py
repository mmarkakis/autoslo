import os

import yaml

import chunkbench.path_utils as pu

NUM_TEMPLATES_OPTIONS = [99]#[33, 66, 99]
PCT_HEAVY_OPTIONS = [0, 10, 25, 50]#[0, 25, 50, 75]
MEAN_INTERARRIVAL_TIME_S_OPTIONS = [10, 30, 60, 120]#[1, 5, 10, 30]
S_IN_HOUR = 3600


def create_chunk_generation_specs():
    out_dir = os.path.join(pu.DATA_PATH, "chunk_generation_specs")
    os.makedirs(out_dir, exist_ok=True)

    for num_templates in NUM_TEMPLATES_OPTIONS:
        for pct_heavy in PCT_HEAVY_OPTIONS:
            for mean_interarrival_time_s in MEAN_INTERARRIVAL_TIME_S_OPTIONS:
                chunk_id = f"tpcds_{num_templates}templates_{pct_heavy:02d}pctheavy_{mean_interarrival_time_s:02d}meaninterarrivals"
                config = {
                    "schema": "tpcds",
                    "chunk_id": chunk_id,
                    "chunk_duration_s": S_IN_HOUR,
                    "random_seed": 42,
                    "num_templates": num_templates,
                    "num_queries_per_template": 3,
                    "pct_heavy": pct_heavy,
                    "mean_interarrival_time_s": mean_interarrival_time_s,
                    "stddev_interarrival_time_s": mean_interarrival_time_s / 2,
                }
                out_path = os.path.join(
                    out_dir,
                    f"{chunk_id}.yml",
                )
                with open(out_path, "w") as f:
                    yaml.dump(config, f, sort_keys=False)


if __name__ == "__main__":
    create_chunk_generation_specs()
