import os

import numpy as np
import pandas as pd
import autoslo.utils.paths as pu


def read_in_query_texts(
    template_id: int,
    num_queries_per_template: int,
) -> dict[int, str]:
    template_str = f"query{template_id:03d}"
    query_texts: dict[int, str] = {}
    template_dir = os.path.join(pu.QUERIES_PATH, template_str)
    if not os.path.exists(template_dir):
        print(f"Template directory {template_dir} missing. Skipping.")
        return query_texts
    for query_num in range(1, num_queries_per_template + 1):
        with open(
            os.path.join(template_dir, f"{template_str}_{query_num:03d}.sql"),
            "r",
        ) as f:
            query_text = f.read()
        query_texts[query_num] = query_text
    return query_texts


def generate_partial_workload(
    query_texts: dict[int, dict[int, str]],
    workload_name: str,
    workload_duration_s: float,
    workload_template_ids: list[int],
    workload_num_queries_per_template: int,
    workload_mean_interarrival_s: float,
    workload_stddev_interarrival_s: float,
    relative_start_time_s: float,
    random_seed: int = 42
) -> list[dict]:
    workload = []
    current_time_s = relative_start_time_s
    np.random.seed(random_seed)
    while current_time_s < relative_start_time_s + workload_duration_s:
        
        template_id = np.random.choice(workload_template_ids)
        query_num_within_template = np.random.randint(
            1, workload_num_queries_per_template + 1
        )
        query_text = query_texts[template_id][query_num_within_template]

        workload.append(
            {
                "workload_name": workload_name,
                "rel_start_time_s": current_time_s,
                "query_template": template_id,
                "query_num_within_template": query_num_within_template,
                "query_text": query_text,
            }
        )

        # Update current time.
        interarrival_time_s = max(
            0.0,
            np.random.normal(
                loc=workload_mean_interarrival_s,
                scale=workload_stddev_interarrival_s,
            ),
        )
        current_time_s += interarrival_time_s

    return workload


def synthesize_workload(
    workload_name: str,
    regular_workload_duration_s: float,
    regular_workload_template_ids: list[int],
    regular_workload_num_queries_per_template: int,
    regular_workload_mean_interarrival_s: float,
    regular_workload_stddev_interarrival_s: float,
    interference_workload_relative_start_time_s: float,
    interference_workload_duration_s: float,
    interference_workload_template_ids: list[int],
    interference_workload_num_queries_per_template: int,
    interference_workload_mean_interarrival_s: float,
    interference_workload_stddev_interarrival_s: float,
    random_seed: int = 42
) -> None:

    # Retrieve the query texts for the specified templates.
    all_template_ids = set(
        regular_workload_template_ids + interference_workload_template_ids
    )

    max_queries_per_template = max(
        regular_workload_num_queries_per_template,
        interference_workload_num_queries_per_template,
    )
    query_texts: dict[int, dict[int, str]] = {
        template_id: read_in_query_texts(template_id, max_queries_per_template)
        for template_id in all_template_ids
    }

    
    # Generate the workloads and compile them together.
    regular_workload = generate_partial_workload(
        query_texts=query_texts,
        workload_name=workload_name,
        workload_duration_s=regular_workload_duration_s,
        workload_template_ids=regular_workload_template_ids,
        workload_num_queries_per_template=(
            regular_workload_num_queries_per_template
        ),
        workload_mean_interarrival_s=(
            regular_workload_mean_interarrival_s
        ),
        workload_stddev_interarrival_s=(
            regular_workload_stddev_interarrival_s
        ),
        relative_start_time_s=0.0,
        random_seed=random_seed,
    )
    interference_workload = generate_partial_workload(
        query_texts=query_texts,
        workload_name=workload_name,
        workload_duration_s=interference_workload_duration_s,
        workload_template_ids=interference_workload_template_ids,
        workload_num_queries_per_template=(
            interference_workload_num_queries_per_template
        ),
        workload_mean_interarrival_s=(
            interference_workload_mean_interarrival_s
        ),
        workload_stddev_interarrival_s=(
            interference_workload_stddev_interarrival_s
        ),
        relative_start_time_s=interference_workload_relative_start_time_s,
        random_seed=random_seed,
    )
    full_workload = regular_workload + interference_workload

    # Write the workload out as a DataFrame.
    workload_df = pd.DataFrame(full_workload)
    workload_df = workload_df.sort_values(by="rel_start_time_s").reset_index(
        drop=True
    )
    workload_df['query_id'] = workload_df.index
    out_path = os.path.join(
        pu.get_data_path(), 
        "interference_workloads",
        f"{workload_name}.parquet",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    workload_df.to_parquet(out_path, index=False)


#
LIGHT_COMPLIANT_TEMPLATE_IDS = [42,52,55,98]
LIGHT_DISRUPTIVE_TEMPLATE_IDS = [41,8,61]
LIGHT_DISRUPTIVE_TEMPLATE_IDS_V2 = [21, 32]
HEAVY_COMPLIANT_TEMPLATE_IDS = [47,67]
HEAVY_DISRUPTIVE_TEMPLATE_IDS = [64,72,78]


# Create the 4 workloads we planned, each 30 minutes with a 10 minute interference
# starting at the 10 minute mark.
TEN_MINUTES_S = 10 * 60.0
synthesize_workload(
    workload_name="interference_light_compliant",
    regular_workload_duration_s=3 * TEN_MINUTES_S,
    regular_workload_template_ids=LIGHT_COMPLIANT_TEMPLATE_IDS,
    regular_workload_num_queries_per_template=3,
    regular_workload_mean_interarrival_s=0.5,
    regular_workload_stddev_interarrival_s=0.25,
    interference_workload_relative_start_time_s=TEN_MINUTES_S,
    interference_workload_duration_s=TEN_MINUTES_S,
    interference_workload_template_ids=LIGHT_COMPLIANT_TEMPLATE_IDS,
    interference_workload_num_queries_per_template=3,
    interference_workload_mean_interarrival_s=0.5,
    interference_workload_stddev_interarrival_s=0.25,
)

synthesize_workload(
    workload_name="interference_light_disruptive",
    regular_workload_duration_s=3 * TEN_MINUTES_S,
    regular_workload_template_ids=LIGHT_COMPLIANT_TEMPLATE_IDS,
    regular_workload_num_queries_per_template=3,
    regular_workload_mean_interarrival_s=0.5,
    regular_workload_stddev_interarrival_s=0.25,
    interference_workload_relative_start_time_s=TEN_MINUTES_S,
    interference_workload_duration_s=TEN_MINUTES_S,
    interference_workload_template_ids=LIGHT_DISRUPTIVE_TEMPLATE_IDS,
    interference_workload_num_queries_per_template=3,
    interference_workload_mean_interarrival_s=0.5,
    interference_workload_stddev_interarrival_s=0.25,
)

synthesize_workload(
    workload_name="interference_heavy_compliant",
    regular_workload_duration_s=3 * TEN_MINUTES_S,
    regular_workload_template_ids=LIGHT_COMPLIANT_TEMPLATE_IDS,
    regular_workload_num_queries_per_template=3,
    regular_workload_mean_interarrival_s=0.5,
    regular_workload_stddev_interarrival_s=0.25,
    interference_workload_relative_start_time_s=TEN_MINUTES_S,
    interference_workload_duration_s=TEN_MINUTES_S,
    interference_workload_template_ids=HEAVY_COMPLIANT_TEMPLATE_IDS,
    interference_workload_num_queries_per_template=3,
    interference_workload_mean_interarrival_s=90.0,
    interference_workload_stddev_interarrival_s=30.0,
)

synthesize_workload(
    workload_name="interference_heavy_disruptive",
    regular_workload_duration_s=3 * TEN_MINUTES_S,
    regular_workload_template_ids=LIGHT_COMPLIANT_TEMPLATE_IDS,
    regular_workload_num_queries_per_template=3,
    regular_workload_mean_interarrival_s=0.5,
    regular_workload_stddev_interarrival_s=0.25,
    interference_workload_relative_start_time_s=TEN_MINUTES_S,
    interference_workload_duration_s=TEN_MINUTES_S,
    interference_workload_template_ids=HEAVY_DISRUPTIVE_TEMPLATE_IDS,
    interference_workload_num_queries_per_template=3,
    interference_workload_mean_interarrival_s=90.0,
    interference_workload_stddev_interarrival_s=30.0,
)

synthesize_workload(
    workload_name="interference_light_disruptive_v2",
    regular_workload_duration_s=3 * TEN_MINUTES_S,
    regular_workload_template_ids=LIGHT_COMPLIANT_TEMPLATE_IDS,
    regular_workload_num_queries_per_template=3,
    regular_workload_mean_interarrival_s=0.5,
    regular_workload_stddev_interarrival_s=0.25,
    interference_workload_relative_start_time_s=TEN_MINUTES_S,
    interference_workload_duration_s=TEN_MINUTES_S,
    interference_workload_template_ids=LIGHT_DISRUPTIVE_TEMPLATE_IDS_V2,
    interference_workload_num_queries_per_template=3,
    interference_workload_mean_interarrival_s=0.5,
    interference_workload_stddev_interarrival_s=0.25,
)