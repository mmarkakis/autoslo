# Repository Reference

This page dives deeper into the file structure of this repository. It contains 
the following sections:

- [Paper-to-Code Mapping](#paper-to-code-mapping): Includes source code 
references for each technical section and/or experiment in the paper. 
- [Source Code Reference](#source-code-reference-srcautoslo): Describes the
 source code at `src/autoslo/` by subdirectory.
- [Data Directory Reference](#data-directory-reference-data): Describes the 
contents of the data directory `/data` by subdirectory.

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

---

## Source Code Reference: `src/autoslo/`

### `clusters/` — Cluster management and autoscaling

| Module | Purpose |
|--------|---------|
| `actions.py` | `SpinUpAction`, `TearDownAction` — immutable scaling action dataclasses emitted by the autoscaler and capacity checkpoints |
| `autoscaler.py` | `Autoscaler` — observation windows, scaling triggers, and counterfactual spinup evaluation (Section 6) |
| `autoscaling_policy.py` | `AutoscalingPolicy` enum — selects the spinup RPU selection strategy (e.g. `ADD_SINGLE_BEST_FORWARD`) |
| `autoscaling_trigger_policy.py` | `AutoscalingTriggerPolicy` enum — selects the spinup trigger logic (queue depth, predicted violations, combined) |
| `billing.py` | `Billing` — models Redshift Serverless usage-based billing with pause/resume timers |
| `cluster.py` | `Cluster`, `ClusterView` — mutable cluster state and its frozen snapshot used by the router |
| `cluster_conn_info.py` | `ClusterConnInfo` — Redshift workgroup connection parameters |
| `cluster_provisioner.py` | `ClusterProvisioner` (abstract) + `SimulatedClusterProvisioner` — decouples provisioning from simulation/live execution |
| `managed_cluster_pool.py` | `ManagedClusterPool` — thread-safe registry; enforces `SpinUpBudget`; routes actions to the provisioner |
| `redshift_provisioner.py` | `RedshiftProvisioner` — live implementation of `ClusterProvisioner` using boto3 |
| `redshift_run_stats_collector.py` | Collects query execution stats from a live Redshift workgroup during a run |
| `scheduled_spinup.py` | `ScheduledSpinUp` — pre-scheduled spinup placed by `SpinupOptimizer` and fired by the event loop |
| `spin_up_budget.py` | `SpinUpBudget` — enforces the `max_clusters` cap on the pool |

### `config/` — Execution configuration

| Module | Purpose |
|--------|---------|
| `component_configs.py` | Pydantic-style dataclasses for every component's config block (autoscaler, router, SLO objective, etc.) |
| `execution_config.py` | `ExecutionConfig` — top-level YAML wrapper linking workload, scheduled spinups, and component configs |
| `utils.py` | Utilities for loading, copying, and applying parameter overrides to config files |

### `entry_points/` — CLI scripts

Each file is a runnable script; none are imported by other modules.

| Module | Purpose |
|--------|---------|
| `execute.py` | Run execution manifests against the simulator or live Redshift |
| `microbench.py` | Run efficiency microbenchmarks by manifest name |
| `plot.py` | Plot SLO performance from simulation or live execution results |
| `tune.py` | Run the Policy Tuner from a tuning manifest |

### `featurization/` — Query featurization

| Module | Purpose |
|--------|---------|
| `db_stats_collector.py` | Standalone CLI: collects schema statistics from a live Redshift cluster and writes to `data/db_stats/` |
| `iconq_interaction_featurizer.py` | `IconqInteractionFeaturizer` — builds (query, neighbor) interaction feature vectors for ICONQ+ inference |
| `iconq_query_featurizer.py` | `IconqQueryFeaturizer` — embeds each query as a vector of operator, table, and RPU features |

### `filesystem/` — File I/O utilities

| Module | Purpose |
|--------|---------|
| `config_resolver.py` | Resolves execution config paths and references across the data directory |
| `logos_export.py` | Exports a `StructuredLog` to a Logos-ready DataFrame (standalone utility; not imported by other modules) |
| `path_utils.py` | Project-wide path constants (`DATA_DIR`, etc.) and helpers for constructing run/output paths |
| `structured_events.py` | `EventType` enum defining all structured log event kinds |
| `structured_log.py` | `StructuredLog` — reads and writes per-run structured YAML/Parquet event logs |
| `yaml_helpers.py` | YAML load/dump utilities with parameter substitution support |

### `forecasting/` — Workload forecasting

| Module | Purpose |
|--------|---------|
| `forecast_policy.py` | `ForecastPolicy` enum — selects the workload sampling strategy (e.g. `WEIGHTED_EXPONENTIAL`) |
| `forecaster.py` | `Forecaster` — samples synthetic workloads from `QueryReservoir` via `forecast_n_scenarios` |

### `microbenchmarks/` — Efficiency microbenchmarks (Section 8.8)

| Module | Purpose |
|--------|---------|
| `microbenchmark_runner.py` | `MicrobenchmarkRunner` — base class for all microbenchmark implementations |
| `autoscaling_efficiency.py` | Benchmarks `Autoscaler.consider_spin_up` latency vs. candidate cluster sizes and arrival rates |
| `routing_efficiency.py` | Benchmarks `QueryRouter.route_query` latency vs. cluster count and concurrency |
| `scenario_evaluator_efficiency.py` | Benchmarks `ScenarioEvaluator` / `SimulateBatch` throughput vs. workload size and parallelism |
| `spinup_optimizer_efficiency.py` | Benchmarks `SpinupOptimizer.optimize` latency vs. configuration |

### `model_training/` — ICONQ+ training pipeline

| Module | Purpose |
|--------|---------|
| `collect_model_training_data.py` | Standalone CLI: executes training workloads against live Redshift and records labelled run data |
| `iconq_model_training_checkpoint.py` | `IconqModelTrainingCheckpoint` — saves and resumes training state mid-run |
| `train.py` | Top-level training loop: featurization, dataset construction, loss, checkpointing, and model export |

### `models/` — ML model wrappers

| Module | Purpose |
|--------|---------|
| `cache_model.py` | `CacheModel` — predicts cache-hit probability for query templates; used in cache-aware routing |
| `iconq_model.py` | `IconqModel` — loads `RuntimeNet` weights and exposes `predict_from_dataset` for batched latency inference |
| `iconq_model_config.py` | `IconqModelInitConfig`, `IconqModelTrainConfig` — model hyperparameter dataclasses |
| `model_prediction.py` | `ModelPrediction` — output wrapper with `overall_mean_s()` and distribution accessors |
| `stage_model.py` | `StageModel` — fast XGBoost-based concurrency-unaware latency predictor used as a feature in ICONQ+ |
| `xgboost_model.py` | `XGBoostModel` — generic XGBoost wrapper used for distillation and Stage |

### `nn/` — Neural network modules

| Module | Purpose |
|--------|---------|
| `bayesian_linear.py` | `BayesianLinear` — linear layer with optional weight uncertainty |
| `bayesian_lstm_cell.py` | `BayesianLSTMCell` — LSTM cell with Bayesian weight treatment |
| `bayesian_mlp.py` | `BayesianMlp` — Bayesian MLP (defined but not currently used) |
| `bayesian_pinch_lstm.py` | `BayesianPinchLSTM` — LSTM stack used inside `RuntimeNet` |
| `concurrent_query_dataset.py` | `ConcurrentQueryDataset` — PyTorch `Dataset` building (base query, neighbors) batches for ICONQ+ inference |
| `iconq_runtime_net.py` | Older `IconqRuntimeNet` — superseded by `runtime_net.py`; not imported anywhere |
| `loss_functions.py` | Q-error and NLL loss functions used during ICONQ+ training |
| `lstm_state.py` | `LSTMState` — frozen snapshot of LSTM hidden/cell state enabling incremental inference |
| `runtime_net.py` | `RuntimeNet` — the main LSTM + interaction attention + MDN head used by `IconqModel` |

### `query_plans/` — Query plan parsing

| Module | Purpose |
|--------|---------|
| `parse_plan.py` | Parses Redshift `EXPLAIN` JSON output into a `PlanOperator` tree; writes to `data/parsed_query_plans/` |
| `plan_operator.py` | `PlanOperator` — tree node with operator type, cost estimates, and child list |

### `routing/` — Query router

| Module | Purpose |
|--------|---------|
| `query_router.py` | `QueryRouter`, `QueryRouterState` — implements Algorithm 1; batches ICONQ+ inference across all candidate clusters |
| `query_router_policy.py` | `QueryRouterPolicy` enum — `USE_ICONQ_MODEL`, `USE_STAGE_MODEL`, `CACHE_AWARE`, `ROUND_ROBIN`, etc. |
| `wrapper.py` | Thin adapter wiring `QueryRouter` into `WorkloadRunner` for live execution |

### `slo/` — SLO primitives

| Module | Purpose |
|--------|---------|
| `slo_metric.py` | `SloMetric` enum — `BINARY`, `ABSOLUTE_S`, `RELATIVE` violation metrics |
| `slo_objective.py` | `SloObjective`, `ViolationCost` — bundles metric + threshold; `idx_of_best` implements threshold-aware Pareto selection |
| `slo_resolver.py` | `SloResolver` — maps `query_text_id` to per-query SLO thresholds; supports `tightened(factor)` |
| `slo_table.py` | Standalone CLI: generates per-(percentile, multiplier) SLO YAML files from a run's structured log |

### `tuner/` — Policy tuner (Section 7)

| Module | Purpose |
|--------|---------|
| `parallelism.py` | Centralises process pool size and multiprocessing context settings |
| `param_sweep.py` | `ParamSweep` — Cartesian grid search with Pareto-front selection on train/val workload sets |
| `policy_tuner.py` | `PolicyTuner` — orchestrates Phases 2–7: workload sampling, baseline eval, spinup scheduling, parameter sweeps |
| `policy_tuner_timer.py` | `PolicyTunerTimer` — records wall-clock time for each tuning phase |
| `reservoir.py` | `QueryReservoir` — in-memory store of historical queries indexed by date, hour, and template |
| `scenario_evaluator.py` | `ScenarioEvaluator` — spawns a process pool and evaluates a (configs × workloads) matrix in parallel |
| `spinup_optimizer.py` | `SpinupOptimizer` — greedy scheduled-spinup search (Alg. 4 & 5); also the autoscaler's spinup size selector (Alg. 3) |
| `tuner_console.py` | Module-level Rich `console` singleton; activates file mirroring once the run directory is known |

### `visualizations/` — Plot utilities

| Module | Purpose |
|--------|---------|
| `bar_charts.py` | Grouped and stacked bar chart helpers |
| `colors.py` | Shared color palette and per-series style constants |
| `iconq_model_comparison.py` | ICONQ+ model variant comparison plots (used in experiment notebooks; not imported from `src/`) |
| `iconq_model_performance.py` | ICONQ+ prediction accuracy and calibration plots |
| `prediction_error_cdf.py` | CDF of Q-error / absolute prediction error |
| `prediction_error_scatter.py` | Scatter plots of predicted vs. actual latency (used in experiment notebooks; not imported from `src/`) |
| `render_log_viewer.py` | Interactive structured-log viewer rendered via Rich |
| `scatter_plots.py` | Generic scatter plot utilities for SLO cost/violation trade-offs |
| `spinup_timelines.py` | Timeline visualization of cluster spinup/teardown events |

### `workload_definition/` — Workload and query types

| Module | Purpose |
|--------|---------|
| `create_motivation_workload_tables.py` | Standalone script: generates workload tables for motivation experiments |
| `create_test_workload.py` | Standalone script: creates small deterministic test workloads |
| `generate_redset_workload.py` | Standalone script: generates executable workloads from Redset/Redbench traces |
| `poisson_workload_creator.py` | `PoissonWorkloadCreator` — samples Poisson-arrival workloads from a template distribution |
| `query.py` | `Query` — core dataclass with arrival time, template, SLO, plan, and feature vector |
| `query_plan_registry.py` | `QueryPlanRegistry` — lazy-loading registry mapping query text IDs to parsed plan trees |
| `query_text_registry.py` | `QueryTextRegistry` — maps query text IDs to SQL strings |
| `redbench_to_executable.py` | Converts Redbench join-matching output into an executable `Workload` |
| `schema.py` | `Schema` — database schema definition (table names, operator vocabulary) loaded from YAML |
| `workload.py` | `Workload` — ordered sequence of `Query` objects with metadata |

### `workload_execution/` — Simulator and live runner

| Module | Purpose |
|--------|---------|
| `aggregated_execution_results.py` | `AggregatedSimulationResults` — aggregates per-query results across a workload into SLO and cost summaries |
| `conn_utils.py` | `ConnWithSetup` — psycopg2 connection subclass that applies Redshift session settings on connect |
| `execution_result.py` | `ExecutionResult` — per-query result record (latency, SLO, cluster, timestamps) |
| `run_stats_collector.py` | Standalone CLI: collects per-query statistics from a completed live run |
| `simulator_event.py` | `SimulatorEvent`, `SimulatorEventType` — typed event dataclass for the min-heap event loop |
| `trace.py` | Writes structured per-query trace records during execution (Parquet + YAML) |
| `workload_runner.py` | `WorkloadRunner` — live execution against Redshift; issues queries asynchronously and records results |
| `workload_simulator.py` | `WorkloadSimulator` — discrete-event engine; processes `QUERY_ARRIVAL`, `QUERY_COMPLETION`, `SCHEDULED_SPINUP`, `CLUSTER_READY` events |

---

## Data Directory Reference: `data/`

Files and directories generated at runtime are not committed; this directory
will be largely empty on a fresh clone. Directories prefixed with `__` are
derived caches built automatically from other inputs.

### Input Artifacts

| Path | Contents |
|------|----------|
| `benchmarking_workloads/` | Pre-built TPC-DS workloads used for controlled experiments |
| `cache_models/` | Compiled cache-hit prediction models, keyed by schema hash |
| `chunks/` | Redbench query "chunks" used to assemble composite workloads |
| `composite_workloads/` | Workloads assembled from chunks (cached sub-files excluded) |
| `config/` | Runtime configuration; AWS credentials (`config/aws.yml`) excluded |
| `db_stats/` | Database statistics files consumed by the ICONQ+ featurizer |
| `execution_configs/` | Execution configuration YAMLs; `handmade/` and `tuned_submission/` contain paper-evaluation configs |
| `generation_parameters/` | Parameters controlling workload generation |
| `iconq_models/` | Trained ICONQ+ latency predictor checkpoints (large `dataset.pkl` files excluded) |
| `iconq_query_featurizations/` | Pre-computed ICONQ+ query featurizations for model training |
| `interference_workloads/` | Workloads designed to exercise concurrency and interference |
| `manifests/` | YAML manifests for the entry points: `execution/`, `tuning/`, `plotting/`, `microbench/` |
| `model_training_configs/` | Configuration files for training ICONQ+ and XGBoost models |
| `parsed_query_plans/` | Parsed JSON query plans |
| `redset_executable_workloads/` | Executable Redbench workloads from Redset traces; per-workload files excluded |
| `schemas/` | Database schema YAML definitions |
| `slos/` | SLO definition YAML files |
| `stage_models/` | Trained Stage (concurrency-unaware latency) model checkpoints |
| `test_workloads/` | Small workloads for the test suite (parquet files excluded) |
| `tpcds_normalization_stats/` | Normalisation statistics for TPC-DS query features |
| `tpcds_probability_distributions/` | Per-template query arrival probability distributions |
| `tuner_configs/` | Policy Tuner configuration YAML files |
| `workloads/` | Workload definition YAML files |
| `xgboost_models/` | Trained XGBoost distillation model checkpoints |
| `heavy_templates.txt` | List of "heavy" TPC-DS query templates |
| `light_templates.txt` | List of "light" TPC-DS query templates |
| `tpcds_heavy_templates.txt` | Alternative heavy-template list for TPC-DS experiments |
| `processed.parquet` | Processed query history derived from Redset traces |
| `query_history_processed.parquet` | Processed query history with additional feature columns |

### Generated Outputs

| Path | Populated by | Contents |
|------|-------------|----------|
| `execution_configs/tuned/` | `tune.py` | Execution configs produced by the Policy Tuner |
| `microbenchmark_runs/` | `microbench.py` | Structured logs from microbenchmark runs |
| `plots/` | `plot.py`, `microbench.py` | PNG output plots and associated data |
| `runs/` | `execute.py` (live mode) | Results of runs against live Redshift clusters |
| `simulator_runs/` | `execute.py` (simulator mode) | Results of runs against the workload simulator |

### Internal Caches

| Path | Contents |
|------|----------|
| `__query_featurizations/` | Cached featurizations derived from `iconq_query_featurizations/` |
| `__query_plans/` | Cached parsed query plan objects derived from `parsed_query_plans/` |

### Input Artifacts

| Path | Contents |
|------|----------|
| `benchmarking_workloads/` | Pre-built TPC-DS workloads used for controlled experiments |
| `cache_models/` | Compiled cache-hit prediction models, keyed by schema hash |
| `chunks/` | Redbench query "chunks" used to assemble composite workloads |
| `composite_workloads/` | Workloads assembled from chunks (cached sub-files excluded from version control) |
| `config/` | Runtime configuration; AWS credentials (`config/aws.yml`) are excluded from version control |
| `db_stats/` | Database statistics files consumed by the ICONQ+ featurizer |
| `execution_configs/` | Execution configuration YAML files; the `handmade/` and `tuned_submission/` sub-directories contain configs used in the paper evaluation |
| `generation_parameters/` | Parameters controlling workload generation |
| `iconq_models/` | Trained ICONQ+ latency predictor checkpoints (large `dataset.pkl` files excluded) |
| `iconq_query_featurizations/` | Pre-computed ICONQ+ query featurizations used for model training |
| `interference_workloads/` | Workloads designed to exercise concurrency and interference |
| `manifests/` | YAML manifests consumed by the entry points: `execution/`, `tuning/`, `plotting/`, `microbench/` |
| `model_training_configs/` | Configuration files for training ICONQ+ and XGBoost models |
| `parsed_query_plans/` | Parsed JSON query plans |
| `redset_executable_workloads/` | Executable Redbench workloads derived from Redset traces; per-workload query lists and parquet files are excluded |
| `schemas/` | Database schema YAML definitions |
| `slos/` | SLO definition YAML files |
| `stage_models/` | Trained Stage (concurrency-unaware latency) model checkpoints |
| `test_workloads/` | Small workloads used by the test suite (parquet files excluded) |
| `tpcds_normalization_stats/` | Normalisation statistics for TPC-DS query features |
| `tpcds_probability_distributions/` | Per-template query arrival probability distributions |
| `tuner_configs/` | Policy Tuner configuration YAML files |
| `workloads/` | Workload definition YAML files |
| `xgboost_models/` | Trained XGBoost distillation model checkpoints |
| `heavy_templates.txt` | List of "heavy" TPC-DS query templates |
| `light_templates.txt` | List of "light" TPC-DS query templates |
| `tpcds_heavy_templates.txt` | Alternative heavy-template list used in TPC-DS experiments |
| `processed.parquet` | Processed query history (derived from Redset traces) |
| `query_history_processed.parquet` | Processed query history with additional feature columns |

### Generated Outputs

These directories are populated at runtime and their contents are not committed.

| Path | Populated by | Contents |
|------|-------------|----------|
| `execution_configs/tuned/` | `tune.py` | Execution configs produced by the Policy Tuner |
| `microbenchmark_runs/` | `microbench.py` | Structured logs from microbenchmark runs |
| `plots/` | `plot.py`, `microbench.py` | PNG output plots and associated data |
| `runs/` | `execute.py` (live mode) | Results of runs against live Redshift clusters |
| `simulator_runs/` | `execute.py` (simulator mode) | Results of runs against the workload simulator |

### Internal Caches

| Path | Contents |
|------|----------|
| `__query_featurizations/` | Cached featurizations derived from `iconq_query_featurizations/` |
| `__query_plans/` | Cached parsed query plan objects derived from `parsed_query_plans/` |
