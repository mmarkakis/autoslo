import numpy as np
import pandas as pd

from autoslo.workload_definition.tpcds_sampler import TPCDSSampler


def test_tpcds_sampler_simple():
    dist = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    index_dict = {0: 0, 1: 10, 2: 20}
    column_dict = {0: "A", 1: "B", 2: "C"}
    sampler = TPCDSSampler(dist, index_dict, column_dict)

    latencies_s = pd.Series([0, 2, 10, 12, 20, 22])
    samples = sampler.sample(latencies_s, seed=42)
    assert samples.tolist() == ["A", "A", "B", "B", "C", "C"]


def test_tpcds_sampler_prob_dist():
    dist = np.array([[0.5, 0.5], [0.2, 0.8]])
    index_dict = {0: 0, 1: 10}
    column_dict = {0: "A", 1: "B"}
    sampler = TPCDSSampler(dist, index_dict, column_dict)

    # Create very long series to compute stats with low variance.
    latencies_s = pd.Series([0] * 10000 + [10] * 10000)
    samples = sampler.sample(latencies_s, seed=42)
    samples_0 = samples[:10000]
    samples_10 = samples[10000:]
    prop_A_0 = (samples_0 == "A").mean()
    prop_B_0 = (samples_0 == "B").mean()
    prop_A_10 = (samples_10 == "A").mean()
    prop_B_10 = (samples_10 == "B").mean()
    assert abs(prop_A_0 - 0.5) < 0.05
    assert abs(prop_B_0 - 0.5) < 0.05
    assert abs(prop_A_10 - 0.2) < 0.05
    assert abs(prop_B_10 - 0.8) < 0.05
