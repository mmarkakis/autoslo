"""
This module contains test functions for the module `process_finder.py` located in
`src/autoslo/forecasting/`. The tests focus on verifying the functionality of
the process identification algorithms used to analyze query arrival data.
"""

import pandas as pd
import numpy as np
from autoslo.forecasting.windowed_template_detector import ProcessFinder
import matplotlib.pyplot as plt



def test_suggest_periods_from_fft():
    """
    Test the suggest_periods_from_fft function to ensure it correctly identifies
    candidate periods from synthetic query arrival data.
    """
   
    # Create synthetic data with known periodicities.
    np.random.seed(42)
    time_col = "rel_start_time_s"
    total_time = 3600  # 1 hour
   

    # For every minute, decide how many queries arrive based on two periodic processes.
    t = np.arange(0, total_time, 60)  # 1-minute resolution
    arrivals = (
        5 * (np.sin(2 * np.pi * t / 70) > 0).astype(int) +  # Process with 5 queries every 5 minutes
        3 * (np.sin(2 * np.pi * t / 600 + np.pi/4) > 0).astype(int) +  # Process with 3 queries every 10 minutes
        np.random.poisson(1, size=len(t))  # Random noise
    )

    # Plot the synthetic data for visual inspection.
    plt.figure(figsize=(10, 4))
    plt.plot(t, arrivals)
    plt.title("Synthetic Query Arrival Times")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Queries")
    plt.savefig("synthetic_query_arrivals.png")

    # Now, for each minute, uniformly distribute the arrivals within that minute.
    query_times = []
    for minute, count in zip(t, arrivals):
        if count > 0:
            query_times.extend(
                minute + np.random.uniform(0, 60, size=count)
            )
    df = pd.DataFrame({time_col: query_times})

    # Also plot that for visual inspection.
    plt.figure(figsize=(10, 4))
    plt.hist(df[time_col], bins=60, alpha=0.7)
    plt.title("Histogram of Query Arrivals")
    plt.xlabel("Time (s)")
    plt.ylabel("Number of Queries")
    plt.savefig("histogram_query_arrivals.png")


    # # Run the period suggestion function.
    pf = ProcessFinder.suggest_periods_from_fft(
        df, time_col, dt=1, top_k=2, min_period=10, max_period=700
    )
    print(f"Suggested periods: {pf}")
  

    # # Check if the known periods are in the suggested periods.
    assert any(
        abs(period - 70) < 20 or abs(period - 600) < 20 for period in pf
    ), f"Suggested periods {pf} do not include expected periods."