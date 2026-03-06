import os
from datetime import datetime, timedelta

import pandas as pd
import autoslo.utils.paths as pu

from autoslo.workload_definition.workload import Workload

records = []
gen_start_time = datetime.now()

for i in range(50):
    template = i % 5 + 1
    record = {
        "query_id": f"query_{i}",
        "abs_start_time": gen_start_time + timedelta(seconds=i * 10),
        "query_text_id": f"ext_tpcds1000#{template:03d}#001",
        "repetition_id": "1",
    }
    records.append(record)

df = pd.DataFrame(records)
output_path = os.path.join(
    pu.get_workloads_dir(), "ext_tpcds1000", "test_workload.parquet"
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_parquet(output_path)

# Print a nice summary
Workload.print_summary_from_df(df)