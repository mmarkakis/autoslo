from datetime import datetime, timedelta

import pandas as pd

import autoslo.filesystem.path_utils as pu

records = []
current_time = datetime.fromisoformat("2026-01-01T00:00:00")

REGULAR_TEMPLATES = [53, 63]
l = len(REGULAR_TEMPLATES)
DISRUPTIVE_TEMPLATES = [16, 57]
h = len(DISRUPTIVE_TEMPLATES)


## PHASE 1: Run a bunch of light queries to show that their latency is generally
## good and somewhat bounded.
phase_1_interarrival = timedelta(seconds=0)
for i in range(5 * l):
    template = REGULAR_TEMPLATES[i % l]
    record = {
        "query_id": f"query_{i}",
        "abs_start_time": current_time,
        "query_text_id": f"ext_tpcds1000#{template:03d}#001",
        "repetition_id": "1",
    }
    records.append(record)
    current_time += phase_1_interarrival

# PHASE 2: Run a few heavy queries to show that they have much higher latency and
# that they can cause some of the light queries to have higher latency as well.
phase_2_interarrival = timedelta(seconds=0)
for i in range(2 * h):
    template = DISRUPTIVE_TEMPLATES[i % h]
    record = {
        "query_id": f"query_{5*l + i}",
        "abs_start_time": current_time,
        "query_text_id": f"ext_tpcds1000#{template:03d}#001",
        "repetition_id": "1",
    }
    records.append(record)
    current_time += phase_2_interarrival

# PHASE 3: Go back to running light queries to show that the first few queries
# are impacted negatively.
phase_3_interarrival = timedelta(seconds=0)
for i in range(5 * l):
    template = REGULAR_TEMPLATES[i % l]
    record = {
        "query_id": f"query_{5*l + 5*h + i}",
        "abs_start_time": current_time,
        "query_text_id": f"ext_tpcds1000#{template:03d}#001",
        "repetition_id": "1",
    }
    records.append(record)
    current_time += phase_3_interarrival

df = pd.DataFrame(records)
output_path = (
    pu.get_workloads_dir()
    / "ext_tpcds1000"
    / "motivation_workload_tables.parquet"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path)
