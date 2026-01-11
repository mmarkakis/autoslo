import logging
import os
from datetime import datetime
from math import isclose

import yaml
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.query_timeline import QueryMove, QueryTimeline
from autoslo.blueprint_selection.query_timeline_visualizer import (
    GanttRecorder,
    render_gantt_scrubber,
)
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_execution.trace import Trace

from typing import Optional


class BlueprintSelector:

    def __init__(
        self,
        workload_name: str,
        slo_s: float,
        slo_violation_rate_threshold: float,
        iconq_model_id: str,
        cluster_name: str,
        init_from_trace: bool = True,
        use_stage_for_isolated_queries: bool = False,
        max_iters: int = 20,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a BlueprintSelector instance.

        Parameters:
            workload_name: The name of the workload to use.
            slo_s: The SLO to meet, in seconds.
            slo_violation_rate_threshold: The threshold for acceptable
                SLO violation rate.
            iconq_model_id: The ID of the IconqModel to use for predictions.
            cluster_name: The name of the default cluster to use.
            init_from_trace: Whether to initialize the timeline from the most
                recent trace of the workload on the specified cluster. If False,
                instead bootstrap from the workload queries.
            use_stage_for_isolated_queries: Whether to use the StageModel for
                isolated queries when bootstrapping the timeline.
            max_iters: The maximum number of iterations for the optimization
                process.
            verbose: Whether to log verbose output.
        """
        self._workload_name = workload_name
        workload = Chunk.load(workload_name)  # FIXME: generalize to workloads.
        self._workload = workload
        self._slo_s = slo_s
        self._slo_violation_rate_threshold = slo_violation_rate_threshold
        self._iconq_model_id = iconq_model_id
        self._iconq_model = IconqModel.load(model_id=iconq_model_id)
        self._default_cluster_name = cluster_name
        self._init_from_trace = init_from_trace
        self._use_stage_for_isolated_queries = use_stage_for_isolated_queries
        self._max_iters = max_iters
        self._verbose = verbose
        self._solve_was_invoked = False

        self._recorder: GanttRecorder
        if init_from_trace:
            workload_name = workload.name
            rpu = Cluster.from_config(cluster_name=cluster_name).rpu
            blueprint_name = Blueprint.one_cluster_with(rpu).name
            schema_name = "ext"  # FIXME: make more robust
            run_ids = pu.RunLocator.get_run_ids(
                workload_name=workload_name,
                blueprint_name=blueprint_name,
                schema_name=schema_name,
            )
            assert (
                len(run_ids) == 1
            ), f"Expected exactly one run ID for workload '{workload_name}', blueprint '{blueprint_name}', schema '{schema_name}', got {run_ids}"
            print(f"Initializing from trace with run ID: {run_ids[0]}")
            trace = Trace(run_id=run_ids[0])

            self._query_timeline = QueryTimeline(
                iconq_model=self._iconq_model,
            )
            self._query_timeline.initialize_from_trace(trace=trace)

        else:
            self._query_timeline = (
                self._bootstrap_query_timeline_from_workload()
            )

        # Setup the outputs directory.
        self._run_id = str(int(datetime.now().timestamp()))
        self._out_dir = os.path.join(
            pu.get_data_path(), "selector_runs", self._run_id
        )
        os.makedirs(self._out_dir, exist_ok=False)
        config_out_path = os.path.join(self._out_dir, "config.yml")
        with open(config_out_path, "w") as f:
            d = {
                "workload_name": self._workload_name,
                "slo_s": self._slo_s,
                "slo_violation_rate_threshold": self._slo_violation_rate_threshold,
                "iconq_model_id": self._iconq_model_id,
                "default_cluster_name": self._default_cluster_name,
                "init_from_trace": self._init_from_trace,
                "use_stage_for_isolated_queries": self._use_stage_for_isolated_queries,
                "max_iters": self._max_iters,
                "verbose": self._verbose,
            }
            yaml.safe_dump(d, f, sort_keys=False)

    def _bootstrap_query_timeline_from_workload(
        self,
    ) -> QueryTimeline:
        """
        Bootstraps a QueryTimeline from the workload using the IconqModel.

        Returns:
            A bootstrapped QueryTimeline instance.
        """

        timeline = QueryTimeline(iconq_model=self._iconq_model)

        # Add the queries.
        for i, query in enumerate(self._workload.queries):
            query_id = query.query_id
            start_time_s = query.start_time_s
            temp_and_q_idx = query.tpcds_temp_and_q_idx

            stage_prediction_overall_mean = (
                self._iconq_model.stage_model.predict_from_tpcds_temp_and_q_idx(
                    {query_id: temp_and_q_idx}, self._default_cluster_name
                )[query_id].overall_mean_s()
            )
            timeline.add_query(
                cluster_name=self._default_cluster_name,
                start_time_s=start_time_s,
                end_time_s=(start_time_s + stage_prediction_overall_mean),
                query_id=query_id,
                seq_num=i,
                tpcds_temp_and_q_idx=temp_and_q_idx,
            )

        # Bootstrap the latencies via iterative prediction.
        for i in tqdm(range(100), desc="Bootstrapping latencies"):

            # Get a dataset of overlapping queries.
            dataset = timeline.get_dataset(
                use_log_runtime=self._iconq_model._trained_on_log_runtime
            )

            predictions = self._iconq_model.predict_from_dataset(
                dataset=dataset,
            )
            num_updated = 0
            for query_id, prediction in predictions.items():
                updated = timeline.update_latency(
                    query_id=query_id,
                    latency_s=prediction.overall_mean_s(),
                )
                if updated:
                    num_updated += 1

        return timeline

    @staticmethod
    def _maybe_log(message: str, verbose: bool):
        if verbose:
            logging.info(message)

    def solve(
        self,
    ) -> dict[int, str]:
        """
        Solve for the optimal blueprint and query assignment. Once per instance
        of BlueprintSelector; afterwards returns the cached solution.
        """

        if self._solve_was_invoked:
            # TODO: process and return from the saved state.
            pass

        self._recorder = GanttRecorder()
        self._recorder.snapshot(
            self._query_timeline, label="Start", slo_s=self._slo_s
        )

        if self._verbose:
            logging.basicConfig(
                filename=os.path.join(self._out_dir, "solve.log"),
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )
            logging.info(
                f"Starting blueprint selection solve for workload "
                f"'{self._workload_name}' with SLO {self._slo_s}s and "
                f"SLO violation rate threshold "
                f"{self._slo_violation_rate_threshold}."
            )

        # Phase I: SLO repair
        for it in range(self._max_iters):
            self._maybe_log(
                f"Starting SLO repair iteration {it}.", self._verbose
            )
            eligible_cluster_names = (
                self._eligible_cluster_names_for_next_move()
            )
            if not self._maybe_apply_best_move_for_slo(
                eligible_cluster_names, it, self._verbose
            ):
                self._maybe_log(
                    f"No more SLO-improving moves found in iteration {it}; ending SLO repair phase.",
                    self._verbose,
                )
                break

        # Phase II: cost reduction
        for it in range(self._max_iters):
            self._maybe_log(
                f"Starting cost reduction iteration {it}.", self._verbose
            )
            eligible_cluster_names = (
                self._eligible_cluster_names_for_next_move()
            )
            if not self._maybe_apply_best_move_for_cost(
                eligible_cluster_names, it, self._verbose
            ):
                self._maybe_log(
                    f"No more cost-reducing moves found in iteration {it}; ending cost reduction phase.",
                    self._verbose,
                )
                break

        self.write_out_timeline_visualization()

        mapping = self._query_timeline.seq_num_to_cluster_name()

        mapping_out_path = os.path.join(self._out_dir, "mapping.yml")
        with open(mapping_out_path, "w") as f:
            yaml.safe_dump(mapping, f, sort_keys=False)

        return mapping

    def write_out_timeline_visualization(self) -> None:
        """
        Write out an HTML visualization of the timeline.
        """
        fig = render_gantt_scrubber(
            self._recorder.snapshots,
            slo_s=self._slo_s,
            constant_layout=True,
            violation_rate_threshold=self._slo_violation_rate_threshold,
            workload_name=self._workload_name,
        )

        out_path = os.path.join(
            self._out_dir, "visualization.html"
        )

        fig.write_html(out_path, auto_play=False)

    def _eligible_cluster_names_for_next_move(self) -> list[str]:
        """
        Get the list of eligible cluster names for the next move. These are the
        clusters that already have at least one query assigned to them, plus at
        most one additional cluster at each size.

        Returns:
            A list of eligible cluster names.
        """
        ordered_cluster_names_per_rpu = Cluster.ordered_cluster_names_per_rpu()
        eligible_cluster_names: list[str] = []
        for rpu, cluster_names in ordered_cluster_names_per_rpu.items():
            if rpu > 32:
                continue  # Limit to clusters up to 32 RPUs for practicality.
            for cluster_name in cluster_names:
                eligible_cluster_names.append(cluster_name)
                if cluster_name not in self._query_timeline.active_clusters:
                    break  # Only add at most one inactive cluster per RPU size.
        return eligible_cluster_names

    def _slo_ok(self) -> tuple[float, bool]:
        """
        Check if the SLO violation rate of the current timeline is within
        the acceptable threshold.

        Returns:
            slo_violation_rate: The current SLO violation rate.
            is_ok: Whether the SLO violation rate is within the acceptable
                threshold.
        """
        slo_violation_rate = self._query_timeline.slo_violation_rate(
            slo_s=self._slo_s
        )
        return (
            slo_violation_rate,
            slo_violation_rate <= self._slo_violation_rate_threshold,
        )

    def _maybe_apply_best_move_for_slo(
        self, eligible_cluster_names: list[str], iteration: int, verbose: bool
    ) -> bool:
        """
        Find and apply the best move to reduce the SLO violation rate.

        Parameters:
            eligible_cluster_names: A list of cluster names eligible for moves.
            iteration: The current iteration number.
            verbose: Whether to log verbose output.

        Returns:
            Whether a move was made.
        """
        initial_slo_violation_rate, slo_ok = self._slo_ok()
        self._maybe_log(
            f"Current SLO violation rate: {initial_slo_violation_rate:.4f}",
            verbose,
        )
        if slo_ok:
            self._maybe_log(
                f"SLO violation rate is within acceptable threshold {self._slo_violation_rate_threshold}; no SLO-improving moves needed.",
                verbose,
            )
            return False

        candidate_moves = self._candidate_moves(
            eligible_cluster_names=eligible_cluster_names,
            look_for_slo_violations=True,
            verbose=verbose,
        )

        if not candidate_moves:
            self._maybe_log(
                "No candidate moves found to improve SLO violation rate.",
                verbose,
            )
            return False

        best_move = None
        best_slo_violation_rate = initial_slo_violation_rate

        for move in candidate_moves:
            self._maybe_log(f"Evaluating move: {move}", verbose)
            inverse_move, old_latencies = self._query_timeline.apply_move(
                move,
                verbose=verbose,
                use_stage_for_isolated_queries=self._use_stage_for_isolated_queries,
            )
            new_slo_violation_rate = self._query_timeline.slo_violation_rate(
                slo_s=self._slo_s
            )
            self._maybe_log(
                f"New SLO violation rate after move: {new_slo_violation_rate:.4f}",
                verbose,
            )

            if new_slo_violation_rate < best_slo_violation_rate:
                self._maybe_log(
                    f"Move improved SLO violation rate from "
                    f"{initial_slo_violation_rate:.4f} to "
                    f"{new_slo_violation_rate:.4f}.",
                    verbose,
                )
                best_slo_violation_rate = new_slo_violation_rate
                best_move = move

            self._maybe_log(f"Applying inverse move: {inverse_move}", verbose)
            self._query_timeline.apply_move(
                inverse_move,
                old_latencies,
                verbose=verbose,
                use_stage_for_isolated_queries=self._use_stage_for_isolated_queries,
            )
            self._maybe_log(
                f"After undoing, SLO violation rate is {self._query_timeline.slo_violation_rate(slo_s=self._slo_s):.4f}",
                verbose,
            )

        if best_move is None:
            self._maybe_log("No SLO-improving moves found.", verbose)
            return False

        self._query_timeline.apply_move(
            best_move,
            verbose=verbose,
            use_stage_for_isolated_queries=self._use_stage_for_isolated_queries,
        )
        self._maybe_log(
            f"SLO iteration {iteration}: "
            f"reduced SLO violation rate from "
            f"{initial_slo_violation_rate:.4f} to "
            f"{best_slo_violation_rate:.4f} "
            f"by applying move {best_move}.",
            verbose,
        )
        self._recorder.snapshot(
            self._query_timeline,
            label=f"SLO iter {iteration}",
            slo_s=self._slo_s,
        )
        return True

    def _maybe_apply_best_move_for_cost(
        self, eligible_cluster_names: list[str], iteration: int, verbose: bool
    ) -> bool:
        """
        Find and apply the best move to reduce cost while maintaining SLO.

        Parameters:
            eligible_cluster_names: A list of cluster names eligible for moves.
            iteration: The current iteration number.
            verbose: Whether to log verbose output.

        Returns:
            Whether a move was made.
        """
        initial_cost = self._query_timeline.total_cost()
        initial_num_active_clusters = len(self._query_timeline.active_clusters)
        self._maybe_log(f"Current total cost: {initial_cost:.4f}", verbose)
        self._maybe_log(
            f"Current number of active clusters: {initial_num_active_clusters}",
            verbose,
        )

        candidate_moves = self._candidate_moves(
            eligible_cluster_names=eligible_cluster_names,
            look_for_slo_violations=False,
        )
        if not candidate_moves:
            self._maybe_log("No candidate moves found to reduce cost.", verbose)
            return False

        best_move = None
        best_cost = initial_cost

        for move in candidate_moves:
            self._maybe_log(f"Evaluating move: {move}", verbose)
            inverse_move, old_latencies = self._query_timeline.apply_move(
                move,
                verbose=verbose,
                use_stage_for_isolated_queries=self._use_stage_for_isolated_queries,
            )
            _, slo_ok = self._slo_ok()
            new_cost = self._query_timeline.total_cost()
            new_num_active_clusters = len(self._query_timeline.active_clusters)

            self._maybe_log(f"SLO OK after move: {slo_ok}", verbose)
            self._maybe_log(
                f"New total cost after move: {new_cost:.4f} (from {initial_cost:.4f})",
                verbose,
            )
            self._maybe_log(
                f"New number of active clusters after move: {new_num_active_clusters} (from {initial_num_active_clusters})",
                verbose,
            )

            if (slo_ok) and (
                (new_cost < best_cost)
                or (
                    isclose(new_cost, best_cost, abs_tol=1e-3)
                    and new_num_active_clusters < initial_num_active_clusters
                )
            ):
                self._maybe_log(
                    f"Move reduced cost from "
                    f"{best_cost:.4f} to "
                    f"{new_cost:.4f} and number of active clusters from {initial_num_active_clusters} to {new_num_active_clusters}.",
                    verbose,
                )
                best_cost = new_cost
                best_move = move
            self._maybe_log(f"Applying inverse move: {inverse_move}", verbose)
            self._query_timeline.apply_move(
                inverse_move,
                old_latencies,
                verbose=verbose,
                use_stage_for_isolated_queries=self._use_stage_for_isolated_queries,
            )

        if best_move is None:
            self._maybe_log(
                "No cost-reducing moves found that maintain SLO.", verbose
            )
            return False

        self._query_timeline.apply_move(
            best_move,
            verbose=verbose,
            use_stage_for_isolated_queries=self._use_stage_for_isolated_queries,
        )
        self._maybe_log(
            f"Cost iteration {iteration}: "
            f"reduced cost from {initial_cost:.4f} to {best_cost:.4f} "
            f"by applying move {best_move}.",
            verbose,
        )
        self._recorder.snapshot(
            self._query_timeline,
            label=f"Cost iter {iteration}",
            slo_s=self._slo_s,
        )
        return True

    def _candidate_moves(
        self,
        eligible_cluster_names: list[str],
        look_for_slo_violations: bool,
        verbose: bool = False,
    ) -> list[QueryMove]:
        """
        Generate a list of candidate query moves to lower cost.

        Parameters:
            eligible_cluster_names: A list of cluster names eligible for moves.
            look_for_slo_violations: Whether to look for intervals with SLO
                violations (True) or SLO slack (False).
            verbose: Whether to log verbose output.

        Returns:
            A list of candidate moves, each represented as a tuple containing
            the query id, the origin cluster name, and the target cluster name.
        """

        intervals = self._query_timeline.find_intervals_by_slo_adherence(
            slo_s=self._slo_s, look_for_slo_violations=look_for_slo_violations
        )
        self._maybe_log(
            f"Using SLO {self._slo_s} and `look_for_slo_violations` = "
            f"{look_for_slo_violations}, found {len(intervals)} intervals "
            f"for candidate moves. They are: {intervals}",
            verbose,
        )

        moves: list[QueryMove] = []

        for origin_cluster_name, interval in intervals:
            self._maybe_log(
                f"Generating moves for interval {interval} on cluster "
                f"'{origin_cluster_name}'.",
                verbose,
            )
            queries = self._query_timeline.queries_in_window(
                cluster_name=origin_cluster_name,
                start_time_s=interval.begin,
                end_time_s=interval.end,
                skip_neighbors=True,
            )
            self._maybe_log(
                f"Found {len(queries)} queries in interval "
                f"{interval}: {queries}",
                verbose,
            )
            for query in queries:
                for target_cluster_name in eligible_cluster_names:
                    if target_cluster_name == origin_cluster_name:
                        continue
                    moves.append(
                        QueryMove(
                            query_id=query.data["query_id"],
                            from_cluster_name=origin_cluster_name,
                            to_cluster_name=target_cluster_name,
                        )
                    )
        self._maybe_log(
            f"Generated {len(moves)} candidate moves: {moves}", verbose
        )
        return moves
