from chunkbench.building_blocks.chunk import Chunk
import os
from tqdm.auto import tqdm
import pandas as pd

import chunkbench.path_utils as pu

NUM_TEMPLATES = [99]
PCT_HEAVY = [0, 10, 25, 50]
MEAN_INTERARRIVAL_S = [10.0, 30.0, 60.0, 120]

num_elements = len(NUM_TEMPLATES) * len(PCT_HEAVY) * len(MEAN_INTERARRIVAL_S)

bar = tqdm(total=num_elements, desc="Creating chunk workloads")
for num_templates in NUM_TEMPLATES:
    for pct_heavy in PCT_HEAVY:
        for mean_interarrival_s in MEAN_INTERARRIVAL_S:
            chunk = Chunk(
                H=pct_heavy,
                T=mean_interarrival_s,
                num_templates=num_templates,
            )
            chunk.save()
            chunk.synthesize_chunk_workload()

            bar.update(1)
bar.close()
