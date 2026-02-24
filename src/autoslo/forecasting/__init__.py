"""
autoslo.forecasting
===================
Workload forecasting via time-inhomogeneous HMM phase detection.

Public API
----------
PhasePipeline       End-to-end pipeline (bin → fit → sample).
PipelineConfig      Configuration dataclass for PhasePipeline.
PhaseHMM            Low-level HMM model (fit, viterbi, predict_proba).
PhaseHMMConfig      Configuration dataclass for PhaseHMM.
WorkloadSampler     Ancestral sampler from a fitted PhaseHMM.
ForecastResult      Aggregate forecast statistics over M draws.
WorkloadDraw        A single sampled workload trajectory.
plot_phase_assignment   Visualise inferred phases on training data.
plot_forecast_bands     Visualise per-class forecast uncertainty bands.
plot_state_occupancy    Plot time spent in each phase.
plot_phase_duration_hist Plot phase duration histogram.
plot_transition_heatmap Plot mean transition matrix.
plot_class_mix_over_time Plot per-class mix over time.
"""

from autoslo.forecasting.phase_hmm import PhaseHMM, PhaseHMMConfig, PhaseHMMResult
from autoslo.forecasting.phase_pipeline import (
    PhasePipeline,
    PipelineConfig,
    bin_queries,
    plot_forecast_bands,
    plot_phase_assignment,
    plot_state_occupancy,
    plot_phase_duration_hist,
    plot_transition_heatmap,
    plot_class_mix_over_time,
)
from autoslo.forecasting.workload_sampler import (
    ForecastResult,
    WorkloadDraw,
    WorkloadSampler,
    add_absolute_timestamps,
)

__all__ = [
    "PhaseHMM",
    "PhaseHMMConfig",
    "PhaseHMMResult",
    "PhasePipeline",
    "PipelineConfig",
    "bin_queries",
    "WorkloadSampler",
    "ForecastResult",
    "WorkloadDraw",
    "add_absolute_timestamps",
    "plot_phase_assignment",
    "plot_forecast_bands",
    "plot_state_occupancy",
    "plot_phase_duration_hist",
    "plot_transition_heatmap",
    "plot_class_mix_over_time",
]
