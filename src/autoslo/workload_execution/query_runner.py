import argparse
import asyncio
import logging
import os
import shutil
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Union

import pandas as pd
import yaml
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprints.cluster_pool import ClusterPool
from autoslo.routing.query_router import QueryRouter
from autoslo.workload_definition.workload import Workload
from autoslo.workload_definition.query_text_registry import QueryTextRegistry
from autoslo.workload_definition.schema import Schema


class QueryRunner:
    def __init__(
        self,
        config_path: str | Path,
    ):
        """
        Initialize the QueryRunner from a YAML config file.

        Parameters:
            config_path: Path to a YAML config file inside
                ``data/query_runner_configs/``.  The file must contain the
                fields: ``workload_name``, ``initial_rpus``,
                ``query_router_name``, ``maxconns``, ``closed_loop``.
        """
        self.config_path = Path(config_path)
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Validate workload name and load workload file.
        self.workload_name = cfg["workload_name"]
        if self.workload_name.startswith("benchmarking_workload_"):
            self.workload_path = os.path.join(
                pu.get_data_path(),
                "benchmarking_workloads",
                f"{self.workload_name}.parquet",
            )
        elif self.workload_name.startswith("interference_"):
            self.workload_path = os.path.join(
                pu.get_data_path(),
                "interference_workloads",
                f"{self.workload_name}.parquet",
            )
        else:
            self.workload_path = os.path.join(
                pu.get_data_path(),
                "chunks",
                f"{self.workload_name}",
                "chunk_workload.parquet",
            )
        if not os.path.exists(self.workload_path):
            raise FileNotFoundError(
                f"Workload file {self.workload_path} does not exist."
            )
        self.workload = Workload.load(self.workload_path)
        self.workload.set_rel_start_times_from_zero()
        self.workload_df = self.workload.df
        self.schema = Schema.load(self.workload.schema_name)

        # Build the cluster pool from the configured RPU sizes.
        initial_rpus = cfg["initial_rpus"]
        if not initial_rpus:
            raise ValueError("'initial_rpus' must list at least one RPU size.")
        self.cluster_pool = ClusterPool(initial_rpus=initial_rpus)

        # Validate maxconns and set up connection pool map.
        if cfg["maxconns"] < 1:
            raise ValueError("maxconns must be at least 1.")
        self.maxconns = cfg["maxconns"]
        self.conn_pools = self.cluster_pool.conn_pool_map(
            maxconn=self.maxconns, search_path=self.schema.search_path
        )

        # Set additional parameters.
        self.query_router_name = cfg["query_router_name"]
        self.query_router = QueryRouter.from_name(self.query_router_name)
        self.closed_loop = bool(cfg.get("closed_loop", False))

    def _ts(self, cast_to_int: bool = False) -> Union[int, float]:
        """
        Get the current timestamp.

        Parameters:
            cast_to_int: If True, return the timestamp as an integer (seconds
                since epoch). If False, return as a float (with fractional
                seconds).
        """
        base = datetime.now(tz=timezone.utc).timestamp()
        if cast_to_int:
            return int(base)
        return base

    def _async_ts(self) -> float:
        """
        Get the current timestamp in an async-compatible way.
        """
        return asyncio.get_event_loop().time()

    def _setup_run_directory(self):
        """
        Set up the run directory for storing results and other run metadata.
        """

        # Create a unique run directory based on the current timestamp.
        while True:
            run_id = str(self._ts(cast_to_int=True))
            run_dir = os.path.join(pu.get_runs_path(), f"{run_id}")
            if not os.path.exists(run_dir):
                break
        os.makedirs(run_dir, exist_ok=False)

        # Set up a log file inside the run directory.
        log_file_path = os.path.join(run_dir, "run.log")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        # Remove any existing handlers to avoid duplicate outputs or console handlers.
        for h in list(logger.handlers):
            logger.removeHandler(h)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Prevent propagation to ancestor loggers (which might print to console).
        logger.propagate = False
        logging.info(f"Run directory created at {run_dir}")

        # Dump the parameters of the run into a YAML file.
        d = {
            "run_id": run_id,
            "workload_name": self.workload_name,
            "num_queries": len(self.workload_df),
            "schema_name": self.schema.name,
            "search_path": self.schema.search_path,
            "initial_rpus": [c.rpu for c in self.cluster_pool.clusters],
            "query_router_name": self.query_router_name,
            "maxconns": self.maxconns,
            "closed_loop": self.closed_loop,
        }

        with open(os.path.join(run_dir, "run_params.yml"), "w") as f:
            yaml.dump(d, f, sort_keys=False)
        logging.info(
            f"Run parameters saved to {os.path.join(run_dir, 'run_params.yml')}"
        )

        # Keep a verbatim copy of the config file used for this run.
        shutil.copy2(self.config_path, os.path.join(run_dir, "runner_config.yml"))
        logging.info(f"Config file copied to {os.path.join(run_dir, 'runner_config.yml')}")

        return run_id, run_dir

    def _run_query_sync(
        self, run_id: str, query_id: str, query_text: str, cluster_name: str
    ) -> None:
        """
        Run a single query synchronously.

        Parameters:
            run_id: ID of the current run.
            query_id: ID of the query.
            query_text: SQL text of the query.
            cluster_name: Name of the cluster to run the query on.
        """
        logging.info(f"Starting query {query_id}")
        start_time = self._ts()
        conn = self.conn_pools[cluster_name].getconn()
        try:
            with conn.cursor() as cur:
                edited = f"--{run_id}/{query_id}\n{query_text}"
                cur.execute(edited)
                try:
                    _ = cur.fetchall()
                except Exception as e:
                    pass  # Some queries do not return results.
            conn.commit()
        except Exception as e:
            # Ensure errors don't prevent returning the connection to the pool.
            try:
                conn.rollback()
            except Exception:
                pass
            logging.exception(f"Query {query_id} failed: {e}")
        finally:
            try:
                self.conn_pools[cluster_name].putconn(conn)
            except Exception:
                pass
        end_time = self._ts()
        logging.info(
            f"Query {query_id} finished after t={end_time - start_time:.2f}s"
        )
        self._pbar.update(1)

    async def _run_query_async(
        self,
        run_id: str,
        async_reference_ts: float,
        rel_start_time_s: float,
        query_id: str,
        query_text: str,
        query_text_id: str,
        cluster_name: str,
    ) -> None:
        """
        Run a single query asynchronously, waiting until its scheduled start.

        Parameters:
            run_id: ID of the current run.
            async_reference_ts: Reference timestamp for scheduling.
            rel_start_time_s: Relative start time in seconds from the reference timestamp.
            query_id: ID of the query.
            query_text: SQL text of the query.
            query_text_id: The query_text_id for this query (used for routing).
            cluster_name: Name of the cluster to run the query on.
        """
        now = self._async_ts()
        scheduled_time = async_reference_ts + rel_start_time_s
        delay = scheduled_time - now
        logging.info(
            f"Query {query_id} scheduled to start at t={scheduled_time:.2f}s "
            f"(in {delay:.2f}s)"
        )
        if delay > 0:
            await asyncio.sleep(delay)
        now = self._async_ts()
        self.query_router.on_query_start(
            query_id=query_id,
            cluster_name=cluster_name,
            query_text_id=query_text_id,
            start_time_s=now,
        )
        try:
            fn = partial(
                self._run_query_sync, run_id, query_id, query_text, cluster_name
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, fn)
        finally:
            self.query_router.on_query_finish(
                query_id=query_id,
                cluster_name=cluster_name,
            )

    async def run(self) -> str:
        """
        Run the queries from the workload file.

        Returns:
            The ID of the run.
        """
        run_id, run_dir = self._setup_run_directory()
        print(f"Run started with ID {run_id}.")

        async_reference_ts = self._async_ts()
        logging.info(f"Async reference timestamp: {async_reference_ts:.2f}s")

        tasks = []
        self._pbar = tqdm(total=len(self.workload_df), desc="Queries", unit="q")

        route_info = []

        for _, row in self.workload_df.iterrows():
            query_id = row["query_id"]
            query_text_id = str(row["query_text_id"])
            schema_name = str(row.get("schema_name", ""))
            rel_start_time_s = row["rel_start_time_s"]

            # Resolve the SQL text from the registry.
            query_text = QueryTextRegistry.get(schema_name, query_text_id)
            if query_text is None:
                logging.warning(
                    f"No query text found for schema '{schema_name}', "
                    f"query_text_id '{query_text_id}'. Skipping query {query_id}."
                )
                continue

            route_start_timestamp = self._async_ts()
            cluster_name = self.query_router.route_query(
                query_text=query_text,
                workload_name=self.workload_name,
                seq_num=query_id,
                query_text_id=query_text_id,
            )
            route_end_timestamp = self._async_ts()
            route_info.append(
                {
                    "query_seq_num": query_id,
                    "route_start_timestamp": route_start_timestamp,
                    "route_end_timestamp": route_end_timestamp,
                    "cluster_name": cluster_name,
                }
            )
            if cluster_name not in self.conn_pools:
                print(
                    f"QueryRouter returned unknown cluster name "
                    f"'{cluster_name}' for query {query_id}. Skipping query."
                )
                continue

            if not self.closed_loop:
                task = self._run_query_async(
                    run_id,
                    async_reference_ts,
                    rel_start_time_s,
                    query_id,
                    query_text,
                    query_text_id=query_text_id,
                    cluster_name=cluster_name,
                )
                tasks.append(task)
            else:
                # In closed loop, wait for each query to finish before starting the next.
                # Also ignore rel_start_time_s.
                await self._run_query_async(
                    run_id,
                    async_reference_ts,
                    0,
                    query_id,
                    query_text,
                    query_text_id=query_text_id,
                    cluster_name=cluster_name,
                )

        await asyncio.gather(*tasks)
        self._pbar.close()
        logging.info(f"Run finished at {self._ts()}.")

        # Save query routing timings.
        route_info_df = pd.DataFrame(route_info)
        route_info_df["run_id"] = run_id
        route_info_df["query_router_name"] = self.query_router_name
        route_info_df["routing_time_s"] = (
            route_info_df["route_end_timestamp"]
            - route_info_df["route_start_timestamp"]
        )
        column_order = [
            "run_id",
            "query_router_name",
            "query_seq_num",
            "cluster_name",
            "route_start_timestamp",
            "route_end_timestamp",
            "routing_time_s",
        ]
        route_info_df = route_info_df[column_order]
        route_info_df.to_parquet(
            os.path.join(run_dir, "query_routing_timings.parquet"), index=False
        )

        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run queries from a workload using a YAML config file."
    )
    parser.add_argument(
        "config",
        type=str,
        default=os.path.join(pu.get_query_runner_configs_path(), "default.yml"),
        help="Path to the runner config YAML file (default: "
        "data/query_runner_configs/default.yml).",
    )
    args = parser.parse_args()

    qr = QueryRunner(args.config)
    asyncio.run(qr.run())
