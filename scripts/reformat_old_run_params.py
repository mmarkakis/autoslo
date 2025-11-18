import autoslo.utils.paths as pu
import os
import yaml

if __name__ == "__main__":

    # iterate over all run directories
    runs_dir = pu.get_runs_path()
    for run_dir in os.listdir(runs_dir):
        full_run_dir = os.path.join(runs_dir, run_dir)
        if not os.path.isdir(full_run_dir):
            continue
        run_params_path = os.path.join(full_run_dir, "run_params.yml")

        # Read the run params
        with open(run_params_path, "r") as f:
            run_params = yaml.safe_load(f)

        # If the format is new, continue. The new format has a "closed_loop" key
        if "closed_loop" in run_params:
            continue

        is_closed_loop = (
            "benchmarking_trace_99_3_3_shuffled_42" in run_params["trace_path"]
        )
        workload_name = (
            run_params["trace_path"].split("/")[-1].split(".")[0]
            if "benchmarking" in run_params["trace_path"]
            else run_params["trace_path"].split("/")[-2]
        )

        # Set the new format
        d = {
            "run_id": run_params["run_id"],
            "workload_name": workload_name,
            "num_queries": run_params["num_queries"],
            "scale_factor": run_params["tpcds_scale_factor"],
            "schema_name": run_params["schema_name"],
            "blueprint_name": f"single_{run_params['endpoint_name']}",
            "query_router_name": (
                "RFixed(fixed_cluster_name='cluster_"
                f"{run_params['endpoint_name']}')"
            ),
            "maxconns": run_params["maxconns"],
            "closed_loop": is_closed_loop,
        }

        # Write the existing params into an "old" file
        old_run_params_path = os.path.join(full_run_dir, "run_params_old.yml")
        with open(old_run_params_path, "w") as f:
            yaml.dump(run_params, f)

        # Write back the updated run params to the original file
        with open(run_params_path, "w") as f:
            yaml.dump(d, f, sort_keys=False)
