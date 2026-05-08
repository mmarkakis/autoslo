# AutoSLO



## Policy Tuning

`src/autoslo/entry_points/tune.py`

**Objective**: Optimize an `execution_config` for a (maybe forecasted) workload.

**Conceptual Inputs**: 
- An initial `execution_config` living in `data/execution_configs/*/`
- A tuner config living in `data/tuner_configs`.
- Values for certain parameters in these two files.

**Concrete Inputs**:
One the command line, we have two options:
- Provide the three inputs above separately.
- Point to a `tuning_manifest` living in `data/manifests/tuning/`, a 
YAML file containing the above inputs.

**Outputs**: 
- An optimized `execution_config`, stored in  `data/execution_configs/tuned/`.
- Detailed intermediate outputs stored in `data/tuner_runs/<TUNER_RUN_NAME>`. 
The `TUNER_RUN_NAME` is deterministically derived from the inputs. 


## Execution

`src/autoslo/entry_points/execute.py`

**Objective**: Execute combinations of (workload, `execution_config`), either
against the simulator or against live Redshift clusters.

**Inputs**:
- An `execution_manifest` specifying the combinations to run, living in 
`data/manifests/execution`.
- A flag for whether to run against the simulator or against live clusters.

**Outputs if running against simulator**:
- A directory with simulation results for each combination, stored at 
`data/simulator_runs/<workload_id>/<execution_config_id>` (derived automatically
from the specification of each combination)

**Outputs if running against Redshift**:
- A directory with run results for each combination, stored at 
`data/runs/<run_timestamp>` (derived automatically), so that we never overwrite
runs even if rerunning the same combination.
- An updated metadata file at `data/runs/map.yml`, collecting tuples of (
    `<run_timestamp>, <workload_id>, <execution_config_id>`).



## Plotting

`src/autoslo/entry_points/plot.py`

**Objective**: Plot the SLO performance of different execution outputs (on the
simulator or on live workload clusters)

**Inputs**: 
- A `plotting_manifest` specifying the points to plot, living in 
`data/manifests/plotting/<plot_name>.yml`.
- A flag for whether to plot based on simulator runs or live cluster runs.

**Outputs**:
- A plot living in `data/plots/<plot_name>.png`
