# Repository Reference

This page dives deeper into the file structure of this repository. It contains 
the following sections:

- [Paper-to-Code Mapping](#paper-to-code-mapping): Includes source code 
references for each technical section and/or experiment in the paper. 


---

## Paper-to-Code Mapping


### Section 4: Latency Predictor (Iconq+)

| Concept | Source file(s) |
|---------|-------------|
| Iconq+ model wrapper and inference | `src/autoslo/models/iconq_model.py`, `src/autoslo/models/iconq_model_config.py` |
| Stage model (concurrency-unaware latency proxy) | `src/autoslo/models/stage_model.py` |
| Query / interaction featurization (Section 4.2) | `src/autoslo/featurization/iconq_query_featurizer.py`, `src/autoslo/featurization/iconq_interaction_featurizer.py`, `src/autoslo/nn/concurrent_query_dataset.py` |
| Training, censored observations (Section 4.3) | `src/autoslo/model_training/train.py`, `src/autoslo/model_training/collect_model_training_data.py`, `src/autoslo/nn/loss_functions.py` |
| LSTM network with incremental inference (Section 4.4) | `src/autoslo/nn/runtime_net.py` |

### Section 5: Query Router 

| Concept | Source file(s) |
|---------|-------------|
| Query Router | `src/autoslo/routing/query_router.py` |
| Routing policies | `src/autoslo/routing/query_router_policy.py` |

### Section 6: Autoscaler

| Concept | Source file(s) |
|---------|-------------|
| Autoscaler | `src/autoslo/clusters/autoscaler.py`, `src/autoslo/clusters/autoscaling_trigger_policy.py` |
| Cluster provisioning | `src/autoslo/clusters/cluster.py`, `src/autoslo/clusters/cluster_provisioner.py`, `src/autoslo/clusters/managed_cluster_pool.py` |

### Section 7: Policy Tuner

| Concept | Source file(s) |
|---------|-------------|
| Policy Tuner  | `src/autoslo/tuner/policy_tuner.py` |
| Workload Reservoir (Section 7.1) | `src/autoslo/tuner/reservoir.py` |
| Workload Forecaster (Section 7.2) | `src/autoslo/forecasting/forecaster.py`, `src/autoslo/forecasting/forecast_policy.py` |
| Batch Simulator (Section 7.3) | `src/autoslo/tuner/scenario_evaluator.py`, `src/autoslo/workload_execution/workload_simulator.py` |
| Spinup Scheduler (Section 7.4) | `src/autoslo/tuner/spinup_optimizer.py`, `src/autoslo/clusters/scheduled_spinup.py` |
| Configuration Tuner (Section 7.5) | `src/autoslo/tuner/param_sweep.py` |

### Section 8: Evaluation

| Evaluation subsection | Experiment results |
|----------------------|--------------------|
| End-to-end Effectiveness (Section 8.2) | `data/plots/main_eval_v8/` |
| Latency Predictor (Section 8.3) | `data/plots/iconq_comparison/` |
| Query Router(Section 8.4) | `data/plots/query_router_eval_v8/` |
| Autoscaler: Spinup Size Selector (Section 8.5) | `data/plots/autoscaler_eval_v8/` |
| Autoscaler: Spinup Trigger (Section 8.6) | `data/plots/trigger_eval_v8/` |
| Policy Tuner (Section 8.7) | `data/plots/tuner_eval_v8/` |
| Efficiency (Section 8.8) | `data/plots/routing_efficiency/`, `data/plots/autoscaling_efficiency/`, `data/plots/scenario_evaluator_efficiency/`, `data/plots/spinup_optimizer_efficiency/` |

