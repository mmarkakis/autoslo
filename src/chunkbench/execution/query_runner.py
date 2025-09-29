## Runs queries from a benchmarking trace in an open loop against some endpoint using psycopg2

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from functools import partial
from typing import Union

import pandas as pd
import yaml
from psycopg2.pool import ThreadedConnectionPool
from tqdm.auto import tqdm

import chunkbench.path_utils as pu
from chunkbench.execution.conn_utils import ConnWithSetup, form_hostname


class QueryRunner:
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        """
        Initialize the QueryRunner.

        Parameters:
            args: Command-line arguments.
        """
        self._process_args(args)

        host = form_hostname(
            self.conn_info["workgroup_name"], self.aws_account_id, self.aws_region
        )

        print(f"Opening connection pool to {host}:{self.conn_info['port']}...")
        self.conn_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=self.maxconns,
            host=host,
            port=self.conn_info["port"],
            user=self.conn_info["user"],
            password=self.conn_info["password"],
            dbname=self.conn_info["dbname"],
            connection_factory=lambda *args, **kwargs: ConnWithSetup(
                *args,
                search_path=self.schema_name,
                **kwargs,
            ),
        )
        print("Connection pool established.")

    def _ts(self, cast_to_int: bool = False) -> Union[int, float]:
        """
        Get the current timestamp.

        Parameters:
            cast_to_int: If True, return the timestamp as an integer (seconds since epoch).
                         If False, return as a float (with fractional seconds).
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

    def _process_args(self, args: argparse.Namespace) -> None:
        """
        Process and validate the command-line arguments.

        Parameters:
            args: Command-line arguments.
        """
        # Validate paths.
        self.trace_path = args.trace_path
        if not os.path.exists(self.trace_path):
            raise FileNotFoundError(f"Trace file {self.trace_path} does not exist.")
        self.trace_df = pd.read_parquet(self.trace_path)

        self.conn_info_path = args.conn_info_path
        if not os.path.exists(self.conn_info_path):
            raise FileNotFoundError(
                f"Connection info file {self.conn_info_path} does not exist."
            )

        # Validate connection info.
        with open(args.conn_info_path, "r") as f:
            all_conn_info = yaml.safe_load(f)
        self.aws_account_id = all_conn_info["aws_account_id"]
        self.aws_region = all_conn_info["aws_region"]

        if args.endpoint_name not in all_conn_info["endpoints"]:
            raise ValueError(
                f"Endpoint name {args.endpoint_name} not found in connection info."
            )
        self.endpoint_name = args.endpoint_name
        self.conn_info = all_conn_info["endpoints"][args.endpoint_name]

        if args.tpcds_scale_factor not in all_conn_info["scale_factors"]:
            raise ValueError(
                f"Scale factor {args.tpcds_scale_factor} not found in connection info."
            )
        self.tpcds_scale_factor = args.tpcds_scale_factor
        self.schema_name = all_conn_info["scale_factors"][args.tpcds_scale_factor]

        # Validate maxconns.
        if args.maxconns < 1:
            raise ValueError("maxconns must be at least 1.")
        self.maxconns = args.maxconns

    def _setup_run_directory(self):
        """
        Set up the run directory for storing results and other run metadata.
        """

        # Create a unique run directory based on the current timestamp.
        while True:
            run_id = str(self._ts(cast_to_int=True))
            run_dir = os.path.join(pu.DATA_PATH, "runs", f"{run_id}")
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
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Prevent propagation to ancestor loggers (which might print to console).
        logger.propagate = False
        logging.info(f"Run directory created at {run_dir}")

        # Dump the parameters of the run into a YAML file.
        d = {
            "trace_path": self.trace_path,
            "conn_info_path": self.conn_info_path,
            "tpcds_scale_factor": self.tpcds_scale_factor,
            "endpoint_name": self.endpoint_name,
            "run_id": run_id,
            "num_queries": len(self.trace_df),
            "maxconns": self.maxconns,
            "run_dir": run_dir,
            "schema_name": self.schema_name,
            "aws_account_id": self.aws_account_id,
            "aws_region": self.aws_region,
        }

        with open(os.path.join(run_dir, "run_params.yml"), "w") as f:
            yaml.dump(d, f)
        logging.info(
            f"Run parameters saved to {os.path.join(run_dir, 'run_params.yml')}"
        )

        return run_id

    def _run_query_sync(self, run_id: str, query_id: str, query_text: str) -> None:
        """
        Run a single query synchronously.

        Parameters:
            run_id: ID of the current run.
            query_id: ID of the query.
            query_text: SQL text of the query.
        """
        logging.info(f"Starting query {query_id}")
        start_time = self._ts()
        conn = self.conn_pool.getconn()
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
                self.conn_pool.putconn(conn)
            except Exception:
                pass
        end_time = self._ts()
        logging.info(f"Query {query_id} finished after t={end_time - start_time:.2f}s")
        self._pbar.update(1)

    async def _run_query_async(
        self,
        run_id: str,
        async_reference_ts: float,
        rel_start_time_s: float,
        query_id: str,
        query_text: str,
    ) -> None:
        """
        Run a single query asynchronously, waiting until its scheduled start time.

        Parameters:
            run_id: ID of the current run.
            async_reference_ts: Reference timestamp for scheduling (from _async_ts()).
            rel_start_time_s: Relative start time in seconds from the reference timestamp.
            query_id: ID of the query.
            query_text: SQL text of the query.
        """
        now = self._async_ts()
        scheduled_time = async_reference_ts + rel_start_time_s
        delay = scheduled_time - now
        logging.info(
            f"Query {query_id} scheduled to start at t={scheduled_time:.2f}s (in {delay:.2f}s)"
        )
        if delay > 0:
            await asyncio.sleep(delay)
        fn = partial(self._run_query_sync, run_id, query_id, query_text)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, fn)

    async def run(self, closed_loop: bool = False) -> None:
        """
        Run the queries from the benchmarking trace.

        Parameters:
            closed_loop: If True, run in closed loop (wait for each query to finish before starting the next).
                         If False, run in open loop (start queries based on their scheduled times
        """
        run_id = self._setup_run_directory()
        print(f"Run started with ID {run_id}.")

        async_reference_ts = self._async_ts()
        logging.info(f"Async reference timestamp: {async_reference_ts:.2f}s")

        tasks = []
        self._pbar = tqdm(total=len(self.trace_df), desc="Queries", unit="q")
        for _, row in self.trace_df.iterrows():
            query_id = row["query_id"]
            query_text = row["query_text"]
            rel_start_time_s = row["rel_start_time_s"]

            if not closed_loop:
                task = self._run_query_async(
                    run_id, async_reference_ts, rel_start_time_s, query_id, query_text
                )
                tasks.append(task)
            else:
                # In closed loop, wait for each query to finish before starting the next.
                # Also ignore rel_start_time_s.
                await self._run_query_async(
                    run_id, async_reference_ts, 0, query_id, query_text
                )

        await asyncio.gather(*tasks)
        self._pbar.close()
        logging.info(f"Run finished at {self._ts()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run queries from a benchmarking trace in an open loop."
    )
    parser.add_argument(
        "--trace_path",
        type=str,
        default=os.path.join(
            pu.DATA_PATH,
            "benchmarking_traces",
            "benchmarking_trace_99_3_3_shuffled_42.parquet",
        ),
        help="Path to the Parquet file containing the benchmarking trace.",
    )
    parser.add_argument(
        "--conn_info_path",
        type=str,
        default=os.path.join(pu.CHUNKBENCH_ROOT, "config", "conn.yml"),
        help="Path to the YAML file containing the connection info for psycopg2.",
    )
    parser.add_argument(
        "--tpcds_scale_factor",
        type=int,
        default=1000,
        help="TPC-DS scale factor to run against.",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        default="16",
        help="Endpoint name to run on.",
    )
    parser.add_argument(
        "--maxconns",
        type=int,
        default=1000,
        help="Maximum number of connections in the connection pool.",
    )
    parser.add_argument(
        "--closed_loop",
        action="store_true",
        help="If set, run in closed loop (wait for each query to finish before starting the next).",
    )
    args = parser.parse_args()

    # Create and run the QueryRunner.
    qr = QueryRunner(args)
    asyncio.run(qr.run(closed_loop=args.closed_loop))
