from dataclasses import dataclass

from tqdm.auto import tqdm

from autoslo.blueprint_selection.query_timeline import QueryTimeline
from autoslo.blueprints.cluster import Cluster
from autoslo.models.iconq_model import IconqModel
from autoslo.workload_definition.workload import Workload

from autoslo.blueprint_selection.query_timeline_visualizer import (
    GanttRecorder, 
    render_gantt_scrubber,
)


@dataclass
class QueryMove:
    """
    Represents a move of a query from one cluster to another.
    """

    query_id: str
    from_cluster_name: str
    to_cluster_name: str

    def inverse(self) -> "QueryMove":
        return QueryMove(
            query_id=self.query_id,
            from_cluster_name=self.to_cluster_name,
            to_cluster_name=self.from_cluster_name,
        )


class BlueprintSelector:

    def __init__(
        self,
        workload: Workload,
        slo_s: float,
        slo_violation_rate_threshold: float,
        iconq_model_id: str,
        default_cluster_name: str,
    ) -> None:
        """
        Initialize a BlueprintSelector instance.

        Parameters:
            workload: The workload for which to select a blueprint.
            slo_s: The SLO to meet, in seconds.
            slo_violation_rate_threshold: The threshold for acceptable
                SLO violation rate.
            iconq_model_id: The ID of the IconqModel to use for predictions.
            default_cluster_name: The name of the default cluster to use.
        """
        self._workload = workload
        self._slo_s = slo_s
        self._slo_violation_rate_threshold = slo_violation_rate_threshold
        self._iconq_model_id = iconq_model_id
        self._iconq_model = IconqModel.load(model_id=iconq_model_id)
        self._default_cluster_name = default_cluster_name
        self._query_timeline = self._bootstrap_query_timeline_from_workload()

        self._recorder = GanttRecorder()
        self._recorder.snapshot(self._query_timeline, label="Start", slo_s=self._slo_s)

    def _bootstrap_query_timeline_from_workload(
        self,
    ) -> QueryTimeline:
        """
        Bootstraps a QueryTimeline from the workload using the IconqModel.

        Returns:
            A bootstrapped QueryTimeline instance.
        """

        timeline = QueryTimeline(
            iconq_query_featurizer=self._iconq_model._iconq_query_featurizer,
            iconq_interaction_featurizer=self._iconq_model._iconq_interaction_featurizer,
        )

        # Add the queries.
        for query in self._workload.queries:
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
                tpcds_temp_and_q_idx=temp_and_q_idx,
                stage_model=self._iconq_model.stage_model,
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

    def solve(self, max_iters: int = 20) -> QueryTimeline:
        """
        Solve for the optimal blueprint and query assignment.

        Parameters:
            max_iters: The maximum number of iterations for the optimization
                process.

        Returns:
            The optimized QueryTimeline instance.
        """

        eligible_cluster_names: list[str] = Cluster.all_cluster_names()

        

        # Phase I: SLO repair
        for it in range(max_iters):
            if not self._maybe_apply_best_move_for_slo(
                eligible_cluster_names, it
            ):
                break

        # Phase II: cost reduction
        for it in range(max_iters):
            if not self._maybe_apply_best_move_for_cost(
                eligible_cluster_names, it
            ):
                break

        fig = render_gantt_scrubber(
            self._recorder.snapshots,
            slo_s=self._slo_s,
            constant_layout=True,
            violation_rate_threshold=self._slo_violation_rate_threshold,
        )
        fig.write_html("blueprint_selection_timeline.html", auto_play=False)

        return self._query_timeline

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
        self, eligible_cluster_names: list[str], iteration: int
    ) -> bool:
        """
        Find and apply the best move to reduce the SLO violation rate.

        Parameters:
            eligible_cluster_names: A list of cluster names eligible for moves.
            iteration: The current iteration number.

        Returns:
            Whether a move was made.
        """
        initial_slo_violation_rate, slo_ok = self._slo_ok()
        if slo_ok:
            return False

        candidate_moves = self._candidate_moves(
            eligible_cluster_names=eligible_cluster_names,
            look_for_slo_violations=True,
        )
        if not candidate_moves:
            return False

        best_move = None
        best_slo_violation_rate = initial_slo_violation_rate

        for move in candidate_moves:
            self._apply_move(move)
            new_slo_violation_rate = self._query_timeline.slo_violation_rate(
                slo_s=self._slo_s
            )

            if new_slo_violation_rate < best_slo_violation_rate:
                best_slo_violation_rate = new_slo_violation_rate
                best_move = move
            inverse_move = move.inverse()
            self._apply_move(inverse_move)

        if best_move is None:
            return False

        self._apply_move(best_move)
        print(
            f"SLO iteration {iteration}: "
            f"reduced SLO violation rate from "
            f"{initial_slo_violation_rate:.4f} to "
            f"{best_slo_violation_rate:.4f} "
            f"by applying move {best_move}."
        )
        self._recorder.snapshot(
            self._query_timeline,
            label=f"SLO iter {iteration}",
            slo_s=self._slo_s,
        )
        return True

    def _maybe_apply_best_move_for_cost(
        self, eligible_cluster_names: list[str], iteration: int
    ) -> bool:
        """
        Find and apply the best move to reduce cost while maintaining SLO.

        Parameters:
            eligible_cluster_names: A list of cluster names eligible for moves.
            iteration: The current iteration number.

        Returns:
            Whether a move was made.
        """
        initial_cost = self._query_timeline.total_cost()

        candidate_moves = self._candidate_moves(
            eligible_cluster_names=eligible_cluster_names,
            look_for_slo_violations=False,
        )
        if not candidate_moves:
            return False

        best_move = None
        best_cost = initial_cost

        for move in candidate_moves:
            self._apply_move(move)
            _, slo_ok = self._slo_ok()
            new_cost = self._query_timeline.total_cost()

            if (slo_ok) and (new_cost < best_cost):
                best_cost = new_cost
                best_move = move
            inverse_move = move.inverse()
            self._apply_move(inverse_move)

        if best_move is None:
            return False

        self._apply_move(best_move)
        print(
            f"Cost iteration {iteration}: "
            f"reduced cost from {initial_cost:.4f} to {best_cost:.4f} "
            f"by applying move {best_move}."
        )
        self._recorder.snapshot(
            self._query_timeline,
            label=f"Cost iter {iteration}",
            slo_s=self._slo_s,
        )
        return True

    # Legacy: no longer needed since annotations are computed per snapshot in the visualizer

    def _apply_move(self, move: QueryMove):
        """
        Apply a query move.

        Parameters:
            move: The QueryMove to apply.
        """
        self._query_timeline.move_to_cluster(
            new_cluster_name=move.to_cluster_name, query_id=move.query_id
        )
        interval = self._query_timeline.interval_for_query_id(
            query_id=move.query_id
        )
        dataset = self._query_timeline.get_dataset(
            start_time_s=interval.begin,
            end_time_s=interval.end,
            use_log_runtime=self._iconq_model._trained_on_log_runtime,
        )
        predictions = self._iconq_model.predict_from_dataset(
            dataset=dataset,
        )
        for q_id, prediction in predictions.items():
            self._query_timeline.update_latency(
                query_id=q_id,
                latency_s=prediction.overall_mean_s(),
            )

    def _candidate_moves(
        self,
        eligible_cluster_names: list[str],
        look_for_slo_violations: bool,
    ) -> list[QueryMove]:
        """
        Generate a list of candidate query moves to lower cost.

        Parameters:
            eligible_cluster_names: A list of cluster names eligible for moves.
            look_for_slo_violations: Whether to look for intervals with SLO
                violations (True) or SLO slack (False).

        Returns:
            A list of candidate moves, each represented as a tuple containing
            the query id, the origin cluster name, and the target cluster name.
        """

        intervals = self._query_timeline.find_intervals_by_slo_adherence(
            slo_s=self._slo_s, look_for_slo_violations=look_for_slo_violations
        )

        moves: list[QueryMove] = []

        for origin_cluster_name, interval in intervals:
            queries = self._query_timeline.queries_in_window(
                cluster_name=origin_cluster_name,
                start_time_s=interval.begin,
                end_time_s=interval.end,
                skip_neighbors=True,
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
        return moves
