from datetime import datetime, timedelta

import pandas as pd
import autoslo.filesystem.path_utils as pu

from autoslo.workload_definition.workload import Workload

records = []
gen_start_time = datetime.fromisoformat("2026-01-01T00:00:00")

templates = [1,3,5,6,7]
n = len(templates)

for i in range(5 * n):
    template = templates[i % n]
    record = {
        "query_id": f"query_{i}",
        "abs_start_time": gen_start_time + timedelta(seconds=i * 10),
        "query_text_id": f"ext_tpcds1000#{template:03d}#001",
        "repetition_id": "1",
    }
    records.append(record)

df = pd.DataFrame(records)
output_path = (
    pu.get_workloads_dir() / "ext_tpcds1000" / "test_workload.parquet"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path)

