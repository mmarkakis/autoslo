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

        # Check if there are no .parquet files, and skip if so
        has_parquet = any(
            fname.endswith(".parquet") for fname in os.listdir(full_run_dir)
        )
        if not has_parquet:
            continue

        # Read the run params and determine the cluster name
        with open(run_params_path, "r") as f:
            run_params = yaml.safe_load(f)
        cluster_name = run_params['query_router_name'].split("'")[1]

        # For eevry parquet file without the + character, rename it to have it
        # and add the cluster name after it before the extension
        for fname in os.listdir(full_run_dir):
            if fname.endswith(".parquet") and "+" not in fname:
                base, ext = os.path.splitext(fname)
                new_fname = f"{base}+{cluster_name}{ext}"
                os.rename(
                    os.path.join(full_run_dir, fname),
                    os.path.join(full_run_dir, new_fname),
                )