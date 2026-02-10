import logging
import os
from datetime import datetime
from math import isclose
from typing import Optional

import yaml
from tqdm.auto import tqdm

import autoslo.utils.paths as pu
from autoslo.blueprint_selection.query_timeline import QueryMove, QueryTimeline
from autoslo.blueprint_selection.query_timeline_visualizer import (
    GanttRecorder,
    export_gantt_video,
    render_gantt_scrubber,
)
from autoslo.blueprints.blueprint import Blueprint
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.workload_definition.chunk import Chunk
from autoslo.workload_execution.trace import Trace


class BlueprintSelector:

    COST_TOLERANCE_DOLLARS = 1e-3
    SLO_TOLERANCE_S = 1e-3

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
        export_video: bool = False,
        video_frame_duration: float = 1.0,
        selector_run_id: Optional[str] = None,
        optimize_cumulative_slo_violation_time: bool = False,
        slo_violation_amount_threshold_s: float = 0.0,
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
            export_video: Whether to export a video of the timeline
                optimization process.
            video_frame_duration: Duration in seconds for each frame in the
                exported video (default: 1.0).
            selector_run_id: Optional run ID to use for the selector run. If None,
                a new run ID will be generated based on the current timestamp.
            optimize_cumulative_slo_violation_time: Whether to optimize for the
                cumulative SLO violation time instead of the violation rate.
            slo_violation_amount_threshold_s: The threshold for acceptable
                cumulative SLO violation time, in seconds.
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
        self._export_video = export_video
        self._video_frame_duration = video_frame_duration
        self._solve_was_invoked = False
        self._run_id = selector_run_id or str(int(datetime.now().timestamp()))
        self._optimize_cumulative_slo_violation_time = (
            optimize_cumulative_slo_violation_time
        )
        self._slo_violation_amount_threshold_s = (
            slo_violation_amount_threshold_s
        )

        self.log_buffer: list[str] = []

        # Setup the outputs directory.
        self._out_dir = os.path.join(
            pu.get_data_path(), "selector_runs", self._run_id
        )
        os.makedirs(self._out_dir, exist_ok=True)
        config_out_path = os.path.join(self._out_dir, "config.yml")
        with open(config_out_path, "w") as f:
            d = {
                "selector_run_id": self._run_id,
                "workload_name": self._workload_name,
                "slo_s": self._slo_s,
                "slo_violation_rate_threshold": self._slo_violation_rate_threshold,
                "iconq_model_id": self._iconq_model_id,
                "default_cluster_name": self._default_cluster_name,
                "init_from_trace": self._init_from_trace,
                "use_stage_for_isolated_queries": self._use_stage_for_isolated_queries,
                "max_iters": self._max_iters,
                "verbose": self._verbose,
                "export_video": self._export_video,
                "video_frame_duration": self._video_frame_duration,
                "optimize_cumulative_slo_violation_time": (
                    self._optimize_cumulative_slo_violation_time
                ),
                "slo_violation_amount_threshold_s": (
                    self._slo_violation_amount_threshold_s
                ),
            }
            yaml.safe_dump(d, f, sort_keys=False)

        if self._verbose:
            log_filename = os.path.join(self._out_dir, "solve.log")
            print(f"Configuring log at {log_filename}")
            logging.basicConfig(
                filename=log_filename,
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                force=True,
            )
            logging.info(
                f"Starting blueprint selection solve for workload "
                f"'{self._workload_name}' with SLO {self._slo_s}s and "
                f"SLO violation rate threshold "
                f"{self._slo_violation_rate_threshold}."
                f" Optimize cumulative SLO violation time: {self._optimize_cumulative_slo_violation_time}"
                f" SLO violation amount threshold: {self._slo_violation_amount_threshold_s}s"
            )

        self._recorder: GanttRecorder
        if init_from_trace:
            workload_name = workload.name
            rpu = Cluster.rpu_for_cluster_name(cluster_name)
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
                slo_s=self._slo_s,
            )
            self._query_timeline.initialize_from_trace(trace=trace)

        else:
            self._query_timeline = (
                self._bootstrap_query_timeline_from_workload()
            )

    def _bootstrap_query_timeline_from_workload(
        self,
    ) -> QueryTimeline:
        """
        Bootstraps a QueryTimeline from the workload using the IconqModel.

        Returns:
            A bootstrapped QueryTimeline instance.
        """
        self._maybe_log(
            (
                f"Bootstrapping QueryTimeline from workload "
                f"'{self._workload_name}' "
                f"using IconqModel '{self._iconq_model_id}'."
            ),
            self._verbose,
        )
        timeline = QueryTimeline(
            iconq_model=self._iconq_model, slo_s=self._slo_s
        )

        # Add the queries.
        for i, query in enumerate(self._workload.queries):
            query_id = str(query.query_id)
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
        i = 0

        # Start a tqdm bar.
        bar = tqdm(
            total=100, desc="Bootstrapping latencies", disable=not self._verbose
        )

        while i < 100:

            # Get a dataset of overlapping queries.
            dataset = timeline.get_dataset(
                use_log_runtime=self._iconq_model._trained_on_log_runtime,
                use_fixed_window_radius_s=self._iconq_model._init_config.use_fixed_window_radius_s,
                use_fixed_window_max_neighbors_per_side=self._iconq_model._init_config.use_fixed_window_max_neighbors_per_side,
            )

            predictions = self._iconq_model.predict_from_dataset(
                dataset=dataset,
            )
            num_updated = 0
            for query_id, prediction in predictions.items():
                new_latency = prediction.overall_mean_s()
                old_latency = timeline.update_latency(
                    query_id=query_id,
                    latency_s=new_latency,
                    only_update_if_increased=True,
                )
                if self._time_is_lower(old_latency, new_latency):
                    num_updated += 1
            bar.update(1)
            i += 1
            if num_updated == 0:
                break

        print(
            f"Bootstrapping converged after {i} iterations.",
        )

        return timeline

    def _maybe_log(
        self, message: str, verbose: bool, dump_after: int = 0
    ) -> None:
        if verbose:
            self.log_buffer.append(message)

            if len(self.log_buffer) >= dump_after:
                for log_msg in self.log_buffer:
                    logging.info(log_msg)
                self.log_buffer = []

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

        # Phase I: SLO repair
        for it in range(self._max_iters):
            self._maybe_log(
                f"Starting SLO repair iteration {it}.", self._verbose
            )
            eligible_cluster_names = self._eligible_cluster_names_for_next_move(
                include_inactive=True
            )
            if not self._maybe_apply_best_move_for_slo(
                eligible_cluster_names,
                it,
                self._verbose,
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
            eligible_cluster_names = self._eligible_cluster_names_for_next_move(
                include_inactive=True
            )
            if not self._maybe_apply_best_move_for_cost(
                eligible_cluster_names,
                it,
                self._verbose,
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

        # Drain the log buffer.
        self._maybe_log("Done.", self._verbose, dump_after=0)

        return mapping

    def _cost_is_lower(self, cost_a: float, cost_b: float) -> bool:
        return cost_a < (cost_b - self.COST_TOLERANCE_DOLLARS)

    def _time_is_lower(self, time_a: float, time_b: float) -> bool:
        return time_a < (time_b - self.SLO_TOLERANCE_S)

    def _cost_is_equal(self, cost_a: float, cost_b: float) -> bool:
        return isclose(cost_a, cost_b, abs_tol=self.COST_TOLERANCE_DOLLARS)

    def _time_is_equal(self, time_a: float, time_b: float) -> bool:
        return isclose(time_a, time_b, abs_tol=self.SLO_TOLERANCE_S)

    def solve_v2(self) -> dict[int, str]:
        """
        An alternative solve method.
        """

        self._recorder = GanttRecorder()
        self._recorder.snapshot(
            self._query_timeline, label="Start", slo_s=self._slo_s
        )

        #### PHASE 1: SLO OPTIMIZATION ####

        # Initialize data structures.
        sorted_deviations = []
        last_version_checked = {}

        for query_id in self._query_timeline.query_ids:
            version = 0
            dev = self._query_timeline.slo_deviation_for_query_id(query_id)
            sorted_deviations.append((dev, query_id, version))
            last_version_checked[query_id] = -1

        sorted_deviations.sort(reverse=True)
        query_id_to_sorted_pos = {
            query_id: i for i, (_, query_id, _) in enumerate(sorted_deviations)
        }
        self._maybe_log(
            f"Initial sorted deviations: {sorted_deviations}", self._verbose
        )

        # Perform optimization.
        applied_at_least_one_move = True
        next_ver_to_use = 1
        iteration = 0
        while applied_at_least_one_move:
            eligible_cluster_names = self._eligible_cluster_names_for_next_move(
                include_inactive=True
            )
            self._maybe_log(
                f"Starting SLO iteration {iteration} with eligible clusters: {eligible_cluster_names}",
                self._verbose,
            )

            applied_at_least_one_move = False
            for i in range(len(sorted_deviations)):
                dev, query_id, ver = sorted_deviations[i]
                self._maybe_log(
                    f"At index {i}: processing query ID {query_id} with deviation {dev:.4f} (version {ver})",
                    self._verbose,
                )
                if dev <= 0:
                    self._maybe_log(
                        f"Query ID {query_id} has non-positive deviation {dev:.4f}; skipping the rest.",
                        self._verbose,
                    )
                    break

                if ver == last_version_checked[query_id]:
                    self._maybe_log(
                        f"Query ID {query_id} has already been checked for version {ver}; skipping.",
                        self._verbose,
                    )
                    continue

                initial_overall_dev = (
                    self._query_timeline.slo_violation_amount()
                )
                initial_overall_cost = self._query_timeline.total_cost()
                best_overall_dev = initial_overall_dev
                cost_at_best_dev = initial_overall_cost
                best_move = None

                self._maybe_log(
                    f"Initial overall SLO deviation: {initial_overall_dev:.4f}",
                    self._verbose,
                )

                # Try moving the query to other clusters.
                for cluster_name in eligible_cluster_names:
                    current_cluster_name = (
                        self._query_timeline.cluster_name_for_query_id(query_id)
                    )
                    if cluster_name == current_cluster_name:
                        continue

                    query_interval = self._query_timeline.interval_for_query_id(
                        query_id
                    )

                    move = QueryMove(
                        query_id=query_id,
                        seq_num=query_interval.data["seq_num"],
                        from_cluster_name=current_cluster_name,
                        to_cluster_name=cluster_name,
                    )
                    self._maybe_log(
                        f"Evaluating move: {move.pretty_print()}",
                        self._verbose,
                    )
                    inverse_move, old_latencies = (
                        self._query_timeline.apply_move(
                            move,
                            verbose=self._verbose,
                        )
                    )
                    new_overall_dev = (
                        self._query_timeline.slo_violation_amount()
                    )
                    new_overall_cost = self._query_timeline.total_cost()

                    self._maybe_log(
                        f"New SLO deviation {new_overall_dev:.4f} and cost {new_overall_cost:.4f}",
                        self._verbose,
                    )

                    if (
                        self._time_is_lower(new_overall_dev, best_overall_dev)
                    ) or (
                        self._time_is_equal(new_overall_dev, best_overall_dev)
                        and self._cost_is_lower(
                            new_overall_cost, cost_at_best_dev
                        )
                    ):
                        best_overall_dev = new_overall_dev
                        cost_at_best_dev = new_overall_cost
                        best_move = move

                    self._maybe_log(
                        f"Applying inverse move: {inverse_move.pretty_print()}",
                        self._verbose,
                    )
                    self._query_timeline.apply_move(
                        inverse_move,
                        old_latencies,
                        verbose=self._verbose,
                    )

                # Check the case where no move worked. If so, mark as checked.
                if best_move is None:
                    self._maybe_log(
                        f"No improving move found for query ID {query_id}; marking as checked for version {ver}.",
                        self._verbose,
                    )
                    last_version_checked[query_id] = ver
                    continue

                # Process the best move.
                self._maybe_log(
                    f"Applying best move for query ID {query_id}: {best_move.pretty_print()} "
                    f"reducing overall SLO deviation from "
                    f"{initial_overall_dev:.4f} to {best_overall_dev:.4f} and cost from "
                    f"{initial_overall_cost:.4f} to {cost_at_best_dev:.4f}.",
                    self._verbose,
                )
                inverse_move, old_latencies = self._query_timeline.apply_move(
                    best_move,
                    verbose=self._verbose,
                )
                self._maybe_log(
                    f"Applying refreshed latencies for overlapping queries: {sorted(list(old_latencies.keys()))}",
                    self._verbose,
                )
                for query_id in old_latencies.keys():
                    query_interval = self._query_timeline.interval_for_query_id(
                        query_id
                    )
                    pos = query_id_to_sorted_pos[query_id]
                    sorted_deviations[pos] = (
                        self._query_timeline.slo_deviation_for_query_id(
                            query_id
                        ),
                        query_id,
                        next_ver_to_use,
                    )
                next_ver_to_use += 1
                applied_at_least_one_move = True

                # Re-sort the deviations.
                sorted_deviations.sort(reverse=True)
                query_id_to_sorted_pos = {
                    query_id: i
                    for i, (_, query_id, _) in enumerate(sorted_deviations)
                }

                # Take snapshot.
                self._recorder.snapshot(
                    self._query_timeline,
                    label=f"SLO iter {iteration}",
                    slo_s=self._slo_s,
                )
                iteration += 1

        #### PHASE 2: COST OPTIMIZATION ####
        eligible_cluster_names = self._eligible_cluster_names_for_next_move(
            include_inactive=True
        )
        iteration = 0

        # Initialize data structures.
        sorted_deviations = []
        last_version_checked = {}

        for query_id in self._query_timeline.query_ids:
            version = 0
            dev = self._query_timeline.slo_deviation_for_query_id(query_id)
            sorted_deviations.append((dev, query_id, version))
            last_version_checked[query_id] = -1

        sorted_deviations.sort(reverse=False)
        query_id_to_sorted_pos = {
            query_id: i for i, (_, query_id, _) in enumerate(sorted_deviations)
        }
        self._maybe_log(
            f"Initial sorted deviations: {sorted_deviations}", self._verbose
        )

        # Perform optimization.
        applied_at_least_one_move = True
        next_ver_to_use = 1
        while applied_at_least_one_move:

            self._maybe_log(
                f"Starting cost iteration {iteration} with eligible clusters: {eligible_cluster_names}",
                self._verbose,
            )

            applied_at_least_one_move = False
            for i in range(len(sorted_deviations)):
                dev, query_id, ver = sorted_deviations[i]
                self._maybe_log(
                    f"At index {i}: processing query ID {query_id} with deviation {dev:.4f} (version {ver})",
                    self._verbose,
                )

                if ver == last_version_checked[query_id]:
                    self._maybe_log(
                        f"Query ID {query_id} has already been checked for version {ver}; skipping.",
                        self._verbose,
                    )
                    continue

                initial_overall_dev = (
                    self._query_timeline.slo_violation_amount()
                )
                initial_overall_cost = self._query_timeline.total_cost()
                best_overall_cost = initial_overall_cost
                slo_dev_at_best_cost = initial_overall_dev
                best_move = None

                self._maybe_log(
                    f"Initial overall SLO deviation: {initial_overall_dev:.4f} and cost: {initial_overall_cost:.4f}",
                    self._verbose,
                )

                # Try moving the query to other clusters.
                for cluster_name in eligible_cluster_names:
                    current_cluster_name = (
                        self._query_timeline.cluster_name_for_query_id(query_id)
                    )
                    if cluster_name == current_cluster_name:
                        continue

                    query_interval = self._query_timeline.interval_for_query_id(
                        query_id
                    )

                    move = QueryMove(
                        query_id=query_id,
                        seq_num=query_interval.data["seq_num"],
                        from_cluster_name=current_cluster_name,
                        to_cluster_name=cluster_name,
                    )
                    self._maybe_log(
                        f"Evaluating move: {move.pretty_print()}",
                        self._verbose,
                    )
                    inverse_move, old_latencies = (
                        self._query_timeline.apply_move(
                            move,
                            verbose=self._verbose,
                        )
                    )
                    new_overall_dev = (
                        self._query_timeline.slo_violation_amount()
                    )
                    new_overall_cost = self._query_timeline.total_cost()
                    self._maybe_log(
                        f"After move, SLO deviation {new_overall_dev:.4f} and cost {new_overall_cost:.4f}",
                        self._verbose,
                    )

                    if (
                        self._time_is_lower(
                            new_overall_dev,
                            self._slo_violation_amount_threshold_s,
                        )
                    ) and self._cost_is_lower(
                        new_overall_cost, best_overall_cost
                    ):
                        best_overall_cost = new_overall_cost
                        slo_dev_at_best_cost = new_overall_dev
                        best_move = move

                    self._maybe_log(
                        f"Applying inverse move: {inverse_move.pretty_print()}",
                        self._verbose,
                    )
                    self._query_timeline.apply_move(
                        inverse_move,
                        old_latencies,
                        verbose=self._verbose,
                    )

                # Check the case where no move worked. If so, mark as checked.
                if best_move is None:
                    self._maybe_log(
                        f"No improving move found for query ID {query_id}; marking as checked for version {ver}.",
                        self._verbose,
                    )
                    last_version_checked[query_id] = ver
                    continue

                # Process the best move.
                self._maybe_log(
                    f"Applying best move for query ID {query_id}: {best_move.pretty_print()} "
                    f"changing overall SLO deviation from "
                    f"{initial_overall_dev:.4f} to {slo_dev_at_best_cost:.4f} and "
                    f" cost from {initial_overall_cost:.4f} to {best_overall_cost:.4f}.",
                    self._verbose,
                )
                inverse_move, old_latencies = self._query_timeline.apply_move(
                    best_move,
                    verbose=self._verbose,
                )
                self._maybe_log(
                    f"Applying refreshed latencies for overlapping queries: {sorted(list(old_latencies.keys()))}",
                    self._verbose,
                )
                for query_id in old_latencies.keys():
                    query_interval = self._query_timeline.interval_for_query_id(
                        query_id
                    )
                    pos = query_id_to_sorted_pos[query_id]
                    sorted_deviations[pos] = (
                        self._query_timeline.slo_deviation_for_query_id(
                            query_id
                        ),
                        query_id,
                        next_ver_to_use,
                    )
                next_ver_to_use += 1
                applied_at_least_one_move = True

                # Re-sort the deviations.
                sorted_deviations.sort(reverse=False)
                query_id_to_sorted_pos = {
                    query_id: i
                    for i, (_, query_id, _) in enumerate(sorted_deviations)
                }

                # Take snapshot.
                self._recorder.snapshot(
                    self._query_timeline,
                    label=f"Cost iter {iteration}",
                    slo_s=self._slo_s,
                )
                iteration += 1

        self.write_out_timeline_visualization()

        mapping = self._query_timeline.seq_num_to_cluster_name()

        mapping_out_path = os.path.join(self._out_dir, "mapping.yml")
        with open(mapping_out_path, "w") as f:
            yaml.safe_dump(mapping, f, sort_keys=False)

        # Drain the log buffer.
        self._maybe_log("Done.", self._verbose, dump_after=0)

        # Also write out a small yaml of the final SLO violation and cost.
        final_stats_out_path = os.path.join(self._out_dir, "final_stats.yml")
        with open(final_stats_out_path, "w") as f:
            d = {
                "final_slo_violation_amount": (
                    self._query_timeline.slo_violation_amount()
                ),
                "final_slo_violation_rate": (
                    self._query_timeline.slo_violation_rate()
                ),
                "final_total_cost": self._query_timeline.total_cost(),
            }
            yaml.safe_dump(d, f, sort_keys=False)

        return mapping

    def write_out_timeline_visualization(self) -> None:
        """
        Write out an HTML visualization of the timeline.
        Optionally also exports a video if export_video flag is set.
        """
        fig = render_gantt_scrubber(
            self._recorder.snapshots,
            slo_s=self._slo_s,
            violation_rate_threshold=self._slo_violation_rate_threshold,
            violation_amount_threshold=self._slo_violation_amount_threshold_s,
            optimize_cumulative_slo_violation_time=(
                self._optimize_cumulative_slo_violation_time
            ),
            workload_name=self._workload_name,
        )

        out_path = os.path.join(self._out_dir, "visualization.html")

        fig.write_html(out_path, auto_play=False, include_plotlyjs="cdn")

        # Export video if requested
        if self._export_video:
            video_out_path = os.path.join(self._out_dir, "visualization.mp4")
            export_gantt_video(
                snapshots=self._recorder.snapshots,
                slo_s=self._slo_s,
                output_path=video_out_path,
                frame_duration=self._video_frame_duration,
                constant_layout=True,
                violation_rate_threshold=self._slo_violation_rate_threshold,
                violation_amount_threshold=self._slo_violation_amount_threshold_s,
                optimize_cumulative_slo_violation_time=(
                    self._optimize_cumulative_slo_violation_time
                ),
                workload_name=self._workload_name,
            )

    def _eligible_cluster_names_for_next_move(
        self, include_inactive: bool
    ) -> list[str]:
        """
        Get the list of eligible cluster names for the next move. These are the
        clusters that already have at least one query assigned to them, plus
        optionally one inactive cluster per RPU size.

        Parameters:
            include_inactive: Whether to include inactive clusters, at most
                one per RPU size.

        Returns:
            A list of eligible cluster names.
        """
        ordered_cluster_names_per_rpu = Cluster.ordered_cluster_names_per_rpu()
        eligible_cluster_names: set[str] = set(
            self._query_timeline.active_clusters.copy()
        )
        if not include_inactive:
            return sorted(list(eligible_cluster_names))

        for rpu, cluster_names in ordered_cluster_names_per_rpu.items():
            if rpu > 32:
                continue  # Limit to clusters up to 32 RPUs for practicality.
            for cluster_name in cluster_names:
                eligible_cluster_names.add(cluster_name)
                if cluster_name not in self._query_timeline.active_clusters:
                    break  # Only add at most one inactive cluster per RPU size.
        return sorted(list(eligible_cluster_names))

    def _slo_ok(self) -> tuple[float, bool]:
        """
        Check if the SLO violation rate of the current timeline is within
        the acceptable threshold.

        Returns:
            slo_violation_rate: The current SLO violation rate.
            is_ok: Whether the SLO violation rate is within the acceptable
                threshold.
        """
        if not self._optimize_cumulative_slo_violation_time:
            slo_violation_rate = self._query_timeline.slo_violation_rate()
            return (
                slo_violation_rate,
                slo_violation_rate <= self._slo_violation_rate_threshold,
            )
        else:
            slo_violation_s = self._query_timeline.slo_violation_amount()
            return (
                slo_violation_s,
                slo_violation_s <= self._slo_violation_amount_threshold_s,
            )

    def _fmt_rate_or_amount(self, rate_or_amount: float) -> str:
        if not self._optimize_cumulative_slo_violation_time:
            return f"rate: {rate_or_amount:.4f}"
        else:
            return f"amount: {rate_or_amount:.2f}s"

    def _maybe_apply_best_move_for_slo(
        self,
        eligible_cluster_names: list[str],
        iteration: int,
        verbose: bool,
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
        initial_slo_violation, slo_ok = self._slo_ok()

        self._maybe_log(
            f"Current SLO violation {self._fmt_rate_or_amount(initial_slo_violation)}",
            verbose,
        )
        if slo_ok:
            self._maybe_log(
                f"SLO violation is within acceptable threshold; no SLO-improving moves needed.",
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
                "No candidate moves found to improve SLO violation.",
                verbose,
            )
            return False

        best_move = None
        best_slo_violation = initial_slo_violation

        for move in candidate_moves:
            self._maybe_log(f"Evaluating move: {move.pretty_print()}", verbose)
            inverse_move, old_latencies = self._query_timeline.apply_move(
                move,
                verbose=verbose,
            )
            new_slo_violation, slo_ok = self._slo_ok()
            self._maybe_log(
                f"New SLO violation {self._fmt_rate_or_amount(new_slo_violation)} after move (SLO OK: {slo_ok})",
                verbose,
            )

            if new_slo_violation < best_slo_violation:
                self._maybe_log(
                    f"Move improved SLO violation from "
                    f"{self._fmt_rate_or_amount(initial_slo_violation)} to "
                    f"{self._fmt_rate_or_amount(new_slo_violation)}.",
                    verbose,
                )
                best_slo_violation = new_slo_violation
                best_move = move

            self._maybe_log(
                f"Applying inverse move: {inverse_move.pretty_print()}", verbose
            )
            self._query_timeline.apply_move(
                inverse_move,
                old_latencies,
                verbose=verbose,
            )
            undo_slo_violation, _ = self._slo_ok()
            self._maybe_log(
                f"After undoing, SLO violation {self._fmt_rate_or_amount(undo_slo_violation)}",
                verbose,
            )

        if best_move is None:
            self._maybe_log("No SLO-improving moves found.", verbose)
            return False

        self._query_timeline.apply_move(
            best_move,
            verbose=verbose,
        )
        self._maybe_log(
            f"SLO iteration {iteration}: "
            f"reduced SLO violation from "
            f"{self._fmt_rate_or_amount(initial_slo_violation)} to "
            f"{self._fmt_rate_or_amount(best_slo_violation)} "
            f"by applying move {best_move.pretty_print()}.",
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
            verbose=verbose,
        )
        if not candidate_moves:
            self._maybe_log("No candidate moves found to reduce cost.", verbose)
            return False

        best_move = None
        best_cost = initial_cost

        for move in candidate_moves:
            self._maybe_log(f"Evaluating move: {move.pretty_print()}", verbose)
            inverse_move, old_latencies = self._query_timeline.apply_move(
                move,
                verbose=verbose,
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
            abs_tol = 1e-3

            if (slo_ok) and (
                (new_cost < (best_cost - abs_tol))
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
            self._maybe_log(
                f"Applying inverse move: {inverse_move.pretty_print()}", verbose
            )
            self._query_timeline.apply_move(
                inverse_move,
                old_latencies,
                verbose=verbose,
            )
            self._maybe_log(
                f"After undoing, total cost is {self._query_timeline.total_cost():.4f}",
                verbose,
            )

        if best_move is None:
            self._maybe_log(
                "No cost-reducing moves found that maintain SLO.", verbose
            )
            return False

        self._query_timeline.apply_move(
            best_move,
            verbose=verbose,
        )
        self._maybe_log(
            f"Cost iteration {iteration}: "
            f"reduced cost from {initial_cost:.4f} to {best_cost:.4f} "
            f"by applying move {best_move.pretty_print()}.",
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
            look_for_slo_violations=look_for_slo_violations
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
                f"{interval}: {QueryTimeline._pretty_print_queries_in_window(queries)}",
                verbose,
            )
            for query in queries:
                for target_cluster_name in eligible_cluster_names:
                    if target_cluster_name == origin_cluster_name:
                        continue
                    moves.append(
                        QueryMove(
                            query_id=query.data["query_id"],
                            seq_num=query.data["seq_num"],
                            from_cluster_name=origin_cluster_name,
                            to_cluster_name=target_cluster_name,
                        )
                    )
        self._maybe_log(
            f"Generated {len(moves)} candidate moves: {[move.pretty_print() for move in moves]}",
            verbose,
        )
        return moves
