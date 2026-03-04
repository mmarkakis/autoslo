import argparse
import asyncio
import os
from datetime import datetime

import yaml

import autoslo.utils.paths as pu
from autoslo.blueprints.blueprint import Blueprint
from autoslo.models.iconq_model import IconqModel
from autoslo.routing.query_router import QueryRouter
from autoslo.routing.r_fixed import RFixed
from autoslo.routing.r_modelbased import RModelBased
from autoslo.routing.r_seqnum import RSeqNum
from autoslo.workload_execution.workload_runner import WorkloadRunner
from autoslo.workload_execution.run_stats_collector import RunStatsCollector


async def run_using_selector(base_args: argparse.Namespace):
    """
    Run selected workload.

    Parameters:
        base_args: Base arguments to pass to each WorkloadRunner instance.
    """

    run_ids = []

    router: QueryRouter

    if not base_args.use_model:
        router = RSeqNum(selector_run_id=base_args.selector_run_id)
    else:

        # Read in the config of the selector run to get the Iconq model info
        config_path = os.path.join(
            pu.get_data_path(), "selector_runs", base_args.selector_run_id, "config.yml" 
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        iconq_model_id = config["iconq_model_id"]

        # Now load the Iconq model to find its featurizer id.
        iconq_model = IconqModel.load(iconq_model_id)
        iconq_query_featurizer_id = iconq_model._iconq_query_featurizer_id

        router = RModelBased(
            selector_run_id=base_args.selector_run_id,
            iconq_query_featurizer_id=iconq_query_featurizer_id,
        )
    
   
    base_args.closed_loop = False  # Open-loop

    # Start by running just the example workload a couple of times
    # to get the workgroups to resume.
    example_workload_name = "benchmarking_workload_1_1_5"
    for cluster in router.blueprint.clusters:
        example_args = argparse.Namespace(
            **vars(base_args), workload_name=example_workload_name, 
            blueprint_name=Blueprint(clusters=[cluster]).name,
            query_router_name=RFixed(cluster.name).name
        )
        print(
            f"{datetime.now()} Running example workload {example_workload_name} on cluster {cluster.name}..."
        )
        run_id = await WorkloadRunner(example_args).run()
        run_ids.append(run_id)

    print(f"{datetime.now()} Sleeping for 30 seconds...")
    await asyncio.sleep(30)

    for cluster in router.blueprint.clusters:
        example_args = argparse.Namespace(
            **vars(base_args), workload_name=example_workload_name, 
            blueprint_name=Blueprint(clusters=[cluster]).name,
            query_router_name=RFixed(cluster.name).name
        )
        print(
            f"{datetime.now()} Running example workload {example_workload_name} again on cluster {cluster.name}..."
        )
        run_id = await WorkloadRunner(example_args).run()
        run_ids.append(run_id)

    

    # Wait for a bit to get clean stats.
    print(f"{datetime.now()} Sleeping for 2 minutes...")
    await asyncio.sleep(2 * 60)

    # Now run the actual workload.
    base_args.workload_name = router.workload_name
    base_args.blueprint_name = router.blueprint.name
    base_args.query_router_name = router.name
    
    print(
        f"{datetime.now()} Running workload {base_args.workload_name}..."
    )
    run_id = await WorkloadRunner(base_args).run()
    run_ids.append(run_id)

    print(f"{datetime.now()} Sleeping for 5 minutes...")
    await asyncio.sleep(5 * 60)

    # Now get the statistics out as well.
    print(f"{datetime.now()} Collecting stats for all runs...")
    stats_collector = RunStatsCollector(run_ids=run_ids)
    await stats_collector.collect_stats(skip_write_on_mismatch=True)
    print(f"{datetime.now()} Done collecting stats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run queries from an interference workload in an open loop."
    )        
    parser.add_argument(
        "--tpcds_scale_factor",
        type=int,
        default=1000,
        help="TPC-DS scale factor to run against.",
    )
    parser.add_argument(
        "--selector_run_id",
        type=str,
        help="Selector run ID to use for the mappings.",
        required=True,
    )
    parser.add_argument(
        "--use_model", 
        action="store_true",
        help="Whether to use a model-based router instead of the seqnum-based router.",
    )
    parser.add_argument(
        "--maxconns",
        type=int,
        default=1000,
        help="Maximum number of connections in the connection pool.",
    )
    base_args = parser.parse_args()
    asyncio.run(run_using_selector(base_args))
