from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


QueryId = str
ClusterId = int
Time = float
RPU = int


# -----------------------------
# Inputs + outputs
# -----------------------------


@dataclass(frozen=True)
class Query:
    """Offline query metadata (extend with plan features as needed)."""

    qid: QueryId
    arrival_s: Time
    features: Optional[dict] = (
        None  # optional, for WL_model or routing heuristics
    )


@dataclass(frozen=True)
class SimResult:
    """Realized schedule for a cluster produced by contention-aware simulation."""

    start_s: Dict[QueryId, Time]
    end_s: Dict[QueryId, Time]

    def interval(self, qid: QueryId) -> Tuple[Time, Time]:
        return self.start_s[qid], self.end_s[qid]

    def latency_s(self, q: Query) -> float:
        return self.end_s[q.qid] - q.arrival_s


class Simulator:
    """
    Wrap your Cont_Model behind this interface.

    Must return realized start/end times for every query in qids.
    """

    def __init__(self, queries: Dict[QueryId, Query]):
        self.queries = queries

    def simulate_cluster(self, qids: Set[QueryId], rpu: RPU) -> SimResult:
        raise NotImplementedError


class WLModel:
    """
    Optional: workload-level tail latency predictor for pruning only.
    """

    def predict_tail_latency(
        self, qids: Set[QueryId], rpu: RPU, percentile: float
    ) -> float:
        raise NotImplementedError


# -----------------------------
# State + action
# -----------------------------


@dataclass
class Cluster:
    cid: ClusterId
    rpu: RPU
    queries: Set[QueryId] = field(default_factory=set)
    sim: Optional[SimResult] = None  # cached


@dataclass
class State:
    clusters: Dict[ClusterId, Cluster]
    next_cluster_id: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        if self.next_cluster_id != other.next_cluster_id:
            return False
        if self.clusters.keys() != other.clusters.keys():
            return False
        for cid in self.clusters.keys():
            c1 = self.clusters[cid]
            c2 = other.clusters[cid]
            if c1.rpu != c2.rpu:
                return False
            if c1.queries != c2.queries:
                return False
        return True

    def active_cids(self) -> List[ClusterId]:
        return [cid for cid, c in self.clusters.items() if c.queries]

    def remove_empty_inplace(self) -> None:
        for cid in [cid for cid, c in self.clusters.items() if not c.queries]:
            del self.clusters[cid]

    def copy(self) -> State:
        new = {
            cid: Cluster(cid=cid, rpu=c.rpu, queries=set(c.queries), sim=c.sim)
            for cid, c in self.clusters.items()
        }
        return State(clusters=new, next_cluster_id=self.next_cluster_id)


@dataclass(frozen=True)
class RerouteAction:
    """
    u = (B, a -> b, s_b)
    If dest_cid is None, create a new cluster of size dest_rpu.
    """

    batch: FrozenSet[QueryId]
    src_cid: ClusterId
    dest_cid: Optional[ClusterId]
    dest_rpu: RPU


# -----------------------------
# Helpers: intervals, cost, rho
# -----------------------------


def overlaps(i1: Tuple[Time, Time], i2: Tuple[Time, Time]) -> bool:
    return (i1[0] < i2[1]) and (i2[0] < i1[1])


def active_time_seconds(sim: SimResult, qids: Iterable[QueryId]) -> float:
    intervals = [(sim.start_s[q], sim.end_s[q]) for q in qids]
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def compute_cost(S: State) -> float:
    cost = 0.0
    for c in S.clusters.values():
        if not c.queries:
            continue
        assert c.sim is not None
        cost += c.rpu * active_time_seconds(c.sim, c.queries)
    return cost


def compute_rho(
    S: State, queries: Dict[QueryId, Query], slo_s: float, verbose: bool = False
) -> float:
    ok = 0
    for c in S.clusters.values():
        if not c.queries:
            continue
        assert c.sim is not None
        n_heavy = 0
        n_light = 0
        old_ok = ok
        for qid in c.queries:
            if c.sim.latency_s(queries[qid]) <= slo_s:
                ok += 1
            if (
                hasattr(queries[qid], "features")
                and queries[qid].features is not None
            ):
                if (
                    "is_heavy" in queries[qid].features
                    and queries[qid].features["is_heavy"]
                ):
                    n_heavy += 1
                else:
                    n_light += 1
        if verbose:
            print(
                f"Cluster {c.cid} (RPU {c.rpu}): {ok - old_ok}/{len(c.queries)} queries met SLO."
            )
            print(f"  Heavy queries: {n_heavy}, Light queries: {n_light}")
    return ok / max(1, len(queries))


def violating(
    S: State, queries: Dict[QueryId, Query], slo_s: float
) -> Set[QueryId]:
    out: Set[QueryId] = set()
    for c in S.clusters.values():
        if not c.queries:
            continue
        assert c.sim is not None
        for qid in c.queries:
            if c.sim.latency_s(queries[qid]) > slo_s:
                out.add(qid)
    return out


def event_times(S: State) -> List[Time]:
    t: List[Time] = []
    for c in S.clusters.values():
        if not c.queries or c.sim is None:
            continue
        t.extend(c.sim.start_s[q] for q in c.queries)
        t.extend(c.sim.end_s[q] for q in c.queries)
    t.sort()
    return t


def active_set_at_time(c: Cluster, t: Time) -> Set[QueryId]:
    assert c.sim is not None
    out: Set[QueryId] = set()
    for qid in c.queries:
        s, e = c.sim.interval(qid)
        if s <= t < e:
            out.add(qid)
    return out


# -----------------------------
# Candidate generation
# -----------------------------


def overlap_neighbors(c: Cluster, qid: QueryId) -> Set[QueryId]:
    assert c.sim is not None
    qi = c.sim.interval(qid)
    out: Set[QueryId] = set()
    for other in c.queries:
        if other == qid:
            continue
        if overlaps(qi, c.sim.interval(other)):
            out.add(other)
    return out


def harm_score(
    c: Cluster, qid: QueryId, queries: Dict[QueryId, Query], slo_s: float
) -> float:
    """harm(q) = sum_{q' in N(q)} max(0, ell(q')-SLO) in current schedule."""
    assert c.sim is not None
    score = 0.0
    for q2 in overlap_neighbors(c, qid):
        score += max(0.0, c.sim.latency_s(queries[q2]) - slo_s)
    return score


def choose_violation_hotspots(
    S: State, queries: Dict[QueryId, Query], slo_s: float, top_h: int
) -> List[Time]:
    V = violating(S, queries, slo_s)
    if not V:
        return []
    scored: List[Tuple[float, Time]] = []
    for t in event_times(S):
        sev = 0.0
        for c in S.clusters.values():
            if not c.queries or c.sim is None:
                continue
            for qid in c.queries & V:
                s, e = c.sim.interval(qid)
                if s <= t < e:
                    sev += c.sim.latency_s(queries[qid]) - slo_s
        if sev > 0:
            scored.append((sev, t))
    scored.sort(reverse=True)
    out: List[Time] = []
    for _, t in scored:
        if len(out) >= top_h:
            break
        if not out or abs(t - out[-1]) > 1e-9:
            out.append(t)
    return out


def choose_cost_hotspots(S: State, top_h: int) -> List[Tuple[ClusterId, Time]]:
    """
    Cost hotspots: times where an expensive cluster is active while *few others* are active.
    Intuition: moving queries away can shrink that cluster’s active union and save cost.
    Returns (cid, t) pairs.
    """
    times = event_times(S)
    if not times:
        return []
    pairs: List[Tuple[float, ClusterId, Time]] = []
    cids = S.active_cids()

    # Precompute "who is active at t" cheaply
    for t in times:
        active_clusters = 0
        for cid in cids:
            c = S.clusters[cid]
            if not c.queries or c.sim is None:
                continue
            # active if any query active
            if any(c.sim.start_s[q] <= t < c.sim.end_s[q] for q in c.queries):
                active_clusters += 1

        for cid in cids:
            c = S.clusters[cid]
            if not c.queries or c.sim is None:
                continue
            if not any(
                c.sim.start_s[q] <= t < c.sim.end_s[q] for q in c.queries
            ):
                continue

            # Score: prefer large rpu clusters, and prefer times where few other clusters are active
            score = float(c.rpu) / max(1, active_clusters)
            pairs.append((score, cid, t))

    pairs.sort(reverse=True)
    out: List[Tuple[ClusterId, Time]] = []
    seen = set()
    for _, cid, t in pairs:
        if len(out) >= top_h:
            break
        key = (cid, t)
        if key in seen:
            continue
        seen.add(key)
        out.append((cid, t))
    return out


def destination_candidates(
    S: State,
    src_cid: ClusterId,
    batch: Set[QueryId],
    allow_new: bool,
) -> List[Optional[ClusterId]]:
    """
    Prefer destinations already active during the batch's approximate time span.
    """
    src = S.clusters[src_cid]
    assert src.sim is not None
    starts = [src.sim.start_s[q] for q in batch]
    ends = [src.sim.end_s[q] for q in batch]
    if not starts:
        return []
    b_s, b_e = min(starts), max(ends)

    dests: List[Optional[int]] = []
    for cid, c in S.clusters.items():
        if cid == src_cid or not c.queries or c.sim is None:
            continue
        # destination overlaps batch if any interval overlaps [b_s,b_e]
        if any(overlaps((b_s, b_e), c.sim.interval(q)) for q in c.queries):
            dests.append(cid)

    # fallback: allow any existing destination
    if not dests:
        for cid, c in S.clusters.items():
            if cid != src_cid and c.queries:
                dests.append(cid)

    if allow_new:
        dests.append(None)

    # unique preserve order
    seen = set()
    out = []
    for d in dests:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


# -----------------------------
# Planner
# -----------------------------


class LocalSearchPlanner:
    """
    Offline 2-phase local search with batch reroute as the only primitive action.
    """

    def __init__(
        self,
        queries: Dict[QueryId, Query],
        simulator: Simulator,
        slo_s: float,
        target_x: float,
        k_max: int,
        rpu_ladder: Sequence[RPU],
        *,
        wl_model: Optional[WLModel] = None,  # optional pruning only
        wl_percentile: float = 0.95,  # percentile used for pruning
        top_h: int = 10,
        top_m: int = 10,
        top_l: int = 40,
        new_cluster_sizes: Sequence[RPU] = (2, 4),
    ):
        self.queries = queries
        self.sim = simulator
        self.slo_s = slo_s
        self.target_x = target_x
        self.k_max = k_max
        self.rpu_ladder = list(rpu_ladder)
        self.wl_model = wl_model
        self.wl_percentile = wl_percentile

        self.top_h = top_h
        self.top_m = top_m
        self.top_l = top_l
        self.new_cluster_sizes = list(new_cluster_sizes)

    def _simulate(self, S: State, cid: ClusterId) -> None:
        c = S.clusters[cid]
        c.sim = self.sim.simulate_cluster(set(c.queries), c.rpu)

    def initialize(self, s0: RPU) -> State:
        S = State(clusters={}, next_cluster_id=0)
        cid = S.next_cluster_id
        S.next_cluster_id += 1
        S.clusters[cid] = Cluster(
            cid=cid, rpu=s0, queries=set(self.queries.keys())
        )
        self._simulate(S, cid)
        return S

    def apply(self, S: State, u: RerouteAction) -> State:
        """
        Apply reroute and re-simulate only affected clusters (source + destination).
        """
        S2 = S.copy()
        src = S2.clusters[u.src_cid]
        B = set(u.batch)
        src.queries -= B

        affected: List[ClusterId] = [u.src_cid]

        if u.dest_cid is None:
            new_id = S2.next_cluster_id
            S2.next_cluster_id += 1
            S2.clusters[new_id] = Cluster(
                cid=new_id, rpu=u.dest_rpu, queries=set(B)
            )
            affected.append(new_id)
        else:
            dst = S2.clusters[u.dest_cid]
            # existing cluster keeps its size (paper model); dest_rpu included for uniformity
            dst.queries |= B
            affected.append(dst.cid)

        S2.remove_empty_inplace()
        affected = [
            cid
            for cid in affected
            if cid in S2.clusters and S2.clusters[cid].queries
        ]
        for cid in affected:
            self._simulate(S2, cid)
        return S2

    # -------- candidate construction --------

    def _prune_with_wl_model(self, batch: Set[QueryId], rpu: RPU) -> bool:
        """
        Optional conservative pruning. Return True if candidate should be kept.
        """
        if self.wl_model is None:
            return True
        # If WL_model predicts the batch alone can't meet SLO at this size, it may still help by reducing contention.
        # So keep pruning weak: only prune obviously hopeless options.
        pred = self.wl_model.predict_tail_latency(
            batch, rpu, percentile=self.wl_percentile
        )
        return pred <= 5.0 * self.slo_s  # very loose; tune or remove

    def construct_candidates_repair(self, S: State) -> List[RerouteAction]:
        """
        Phase 1: candidates from violation hotspots; batches ranked by harm.
        """
        allow_new = len(S.active_cids()) < self.k_max
        Ts = choose_violation_hotspots(
            S, self.queries, self.slo_s, top_h=self.top_h
        )
        if not Ts:
            return []

        actions: List[RerouteAction] = []
        for t in Ts:
            for a in S.active_cids():
                ca = S.clusters[a]
                if ca.sim is None:
                    continue
                active = active_set_at_time(ca, t)
                if len(active) <= 1:
                    continue

                ranked = sorted(
                    active,
                    key=lambda qid: harm_score(
                        ca, qid, self.queries, self.slo_s
                    ),
                    reverse=True,
                )
                ranked = list(ranked)[: self.top_m]

                for m in range(1, len(ranked) + 1):
                    B = set(ranked[:m])
                    dests = destination_candidates(S, a, B, allow_new=allow_new)
                    for b in dests:
                        if b is None:
                            for s_new in self.new_cluster_sizes:
                                if self._prune_with_wl_model(B, s_new):
                                    actions.append(
                                        RerouteAction(
                                            frozenset(B), a, None, s_new
                                        )
                                    )
                        else:
                            actions.append(
                                RerouteAction(
                                    frozenset(B), a, b, S.clusters[b].rpu
                                )
                            )
        return actions

    def construct_candidates_cost(self, S: State) -> List[RerouteAction]:
        """
        Phase 2: candidates aimed at cost reduction:
          - target expensive clusters during "unique activity" hotspots
          - try moving active sets (prefix batches) into other clusters (prefer overlap)
          - try emptying clusters (consolidation) by moving their whole query set
        """
        allow_new = False  # in phase 2, avoid creating new clusters
        actions: List[RerouteAction] = []

        # (A) consolidation: try emptying small clusters into larger ones if it doesn't extend active time too much
        for a in S.active_cids():
            ca = S.clusters[a]
            if ca.sim is None or not ca.queries:
                continue
            if len(S.active_cids()) <= 1:
                break
            # move entire cluster if not too large (cap for speed)
            if len(ca.queries) <= self.top_m:
                B = set(ca.queries)
                for b in destination_candidates(S, a, B, allow_new=False):
                    if b is None:
                        continue
                    actions.append(
                        RerouteAction(frozenset(B), a, b, S.clusters[b].rpu)
                    )

        # (B) cost hotspots: peel off batches that keep destination incremental active time small
        hot = choose_cost_hotspots(S, top_h=self.top_h)
        for a, t in hot:
            ca = S.clusters[a]
            if ca.sim is None:
                continue
            active = active_set_at_time(ca, t)
            if len(active) == 0:
                continue

            # For cost, prioritize queries that are "cheap to move" (low latency / non-violating)
            ranked = sorted(
                active,
                key=lambda qid: ca.sim.latency_s(self.queries[qid]),
            )
            ranked = ranked[: self.top_m]

            for m in range(1, len(ranked) + 1):
                B = set(ranked[:m])
                for b in destination_candidates(S, a, B, allow_new=allow_new):
                    if b is None:
                        continue
                    actions.append(
                        RerouteAction(frozenset(B), a, b, S.clusters[b].rpu)
                    )

        return actions

    # -------- selection --------

    def best_repair_move(
        self, S: State, verbose: bool = False
    ) -> Optional[State]:
        rho0 = compute_rho(S, self.queries, self.slo_s)
        cost0 = compute_cost(S)
        cands = self.construct_candidates_repair(S)
        if not cands:
            return None

        scored: List[Tuple[float, float, State]] = []
        for u in cands:
            Su = self.apply(S, u)
            rhou = compute_rho(Su, self.queries, self.slo_s)
            costu = compute_cost(Su)
            scored.append((rhou - rho0, costu - cost0, Su))

        scored.sort(key=lambda x: (-x[0], x[1]))
        best_gain, _, best_state = scored[0]

        if verbose:
            print(
                f"Best repair move: delta rho={best_gain:.6f}, delta cost={scored[0][1]:.6f}"
            )
            print(
                f"  From rho={rho0:.6f}, cost={cost0:.6f} to rho={compute_rho(best_state, self.queries, self.slo_s):.6f}, cost={compute_cost(best_state):.6f}"
            )
            print(
                f"  Action: move {len(cands[0].batch)} queries from cluster {cands[0].src_cid} to {'new cluster' if cands[0].dest_cid is None else f'cluster {cands[0].dest_cid}'} (RPU {cands[0].dest_rpu})"
            )

        if best_gain <= 0:
            return None
        return best_state

    def best_cost_move(self, S: State) -> Optional[State]:
        if compute_rho(S, self.queries, self.slo_s) + 1e-12 < self.target_x:
            return None
        cost0 = compute_cost(S)

        cands = self.construct_candidates_cost(S)
        if not cands:
            return None

        feasible: List[Tuple[float, State]] = []
        for u in cands:
            Su = self.apply(S, u)
            if (
                compute_rho(Su, self.queries, self.slo_s) + 1e-12
                < self.target_x
            ):
                continue
            feasible.append((compute_cost(Su), Su))

        if not feasible:
            return None
        feasible.sort(key=lambda x: x[0])
        best_cost, best_state = feasible[0]
        if best_cost >= cost0 - 1e-12:
            return None
        return best_state

    # -------- main --------

    def solve(self, s0: RPU, max_iters: int = 200) -> State:
        S = self.initialize(s0)

        # Phase I: repair
        for it in range(max_iters):
            if compute_rho(S, self.queries, self.slo_s) >= self.target_x:
                print("Reached target rho")
                break
            S2 = self.best_repair_move(S, verbose=False)
            if S2 is None:
                print(f"No more repair moves in iteration {it}")
                # Escape hatch: create a new cluster and move the worst violator.
                V = list(violating(S, self.queries, self.slo_s))
                if not V or len(S.active_cids()) >= self.k_max:
                    return S
                worst = max(
                    V,
                    key=lambda qid: self._latency_of(S, qid),
                )
                src = self._cluster_of(S, worst)
                # pick the larger of the offered new sizes
                u = RerouteAction(
                    frozenset({worst}), src, None, self.new_cluster_sizes[-1]
                )
                S = self.apply(S, u)
            else:
                S = S2

        # Phase II: cost reduction
        for it in range(max_iters):
            S2 = self.best_cost_move(S)
            if S2 is None or (S2 == S):
                print(f"No more cost moves in iteration {it}")
                break
            S = S2

        return S

    def _cluster_of(self, S: State, qid: QueryId) -> ClusterId:
        for cid, c in S.clusters.items():
            if qid in c.queries:
                return cid
        raise KeyError(qid)

    def _latency_of(self, S: State, qid: QueryId) -> float:
        cid = self._cluster_of(S, qid)
        c = S.clusters[cid]
        assert c.sim is not None
        return c.sim.latency_s(self.queries[qid])


# -----------------------------
# Minimal dummy simulator (replace with Cont_Model)
# -----------------------------


class DummySimulator(Simulator):
    """
    Toy simulator: FIFO, no contention coupling beyond queueing; duration scales with 1/RPU.
    Replace with your Cont_Model-based simulator.
    """

    def simulate_cluster(self, qids: Set[QueryId], rpu: RPU) -> SimResult:
        start: Dict[QueryId, Time] = {}
        end: Dict[QueryId, Time] = {}
        cur = 0.0
        for qid in sorted(qids, key=lambda x: self.queries[x].arrival_s):
            q = self.queries[qid]
            s = q.arrival_s
            dur = 32.0 / rpu  # placeholder runtime
            if hasattr(q, "features") and q.features is not None:
                if "is_heavy" in q.features and q.features["is_heavy"]:
                    dur *= 4.0  # heavy queries take longer
            e = s + dur
            start[qid] = s
            end[qid] = e
        return SimResult(start_s=start, end_s=end)


# -----------------------------
# Visualizer of clusters and routings
# -----------------------------


class FinalStateVisualizer:

    def plot_clusters(
        self, S: State, queries: Dict[QueryId, Query], filename: str
    ) -> None:

        # Simple Gantt chart of query assignments over time. Have a horizontal
        # "lane" per cluster, and plot each query as a line segment. Make sure
        # that line segments are offset vertically per cluster so that they don't
        # overlap. Have the cluster IDs be strings on the y axis, not numbers, and
        # include the number of their rpus.

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        y_ticks = []
        y_labels = []
        y_pos = 0

        # Before plotting, compute for each cluster how much vertical space its
        # lane needs, so that the queries will not visually overlap but we also
        # won't take up excessive vertical space. That is, make sure to reuse
        # vertical space within a cluster as much as possible.
        cluster_lanes: Dict[int, List[Tuple[Time, Time]]] = {}
        for cid, c in S.clusters.items():
            if c.sim is None:
                continue
            intervals = []
            for qid in c.queries:
                s, e = c.sim.interval(qid)
                intervals.append((s, e))
            # Greedily assign intervals to lanes
            lanes: List[List[Tuple[Time, Time]]] = []
            for interval in sorted(intervals):
                placed = False
                for lane in lanes:
                    if all(
                        not overlaps(interval, existing) for existing in lane
                    ):
                        lane.append(interval)
                        placed = True
                        break
                if not placed:
                    lanes.append([interval])
            cluster_lanes[cid] = lanes

        # Plot the queries, coloring them based on whether they are heavy, and
        # including a black outline for each query segment.

        for cid, c in S.clusters.items():
            if c.sim is None:
                continue
            lanes = cluster_lanes[cid]
            for lane_idx, lane in enumerate(lanes):
                for qid in c.queries:
                    s, e = c.sim.interval(qid)
                    if (s, e) in lane:
                        ax.plot(
                            [s, e],
                            [y_pos + lane_idx, y_pos + lane_idx],
                            linewidth=6,
                            color=(
                                "red"
                                if (
                                    hasattr(queries[qid], "features")
                                    and queries[qid].features is not None
                                    and "is_heavy" in queries[qid].features
                                    and queries[qid].features["is_heavy"]
                                )
                                else "blue"
                            ),
                            solid_capstyle="round",
                            zorder=2,
                        )
                        ax.text(
                            (s + e) / 2,
                            y_pos + lane_idx + 0.1,
                            f"{qid}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
            # Label the cluster on the y axis
            y_ticks.append(y_pos + (len(lanes) - 1) / 2)
            y_labels.append(f"Cluster {cid} (RPU {c.rpu})")
            y_pos += len(lanes) + 1  # add space between clusters

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Time (s)")
        ax.set_title("Cluster Query Assignments Over Time")
        ax.grid(True)
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == "__main__":
    qs = {
        f"q{i}": Query(
            qid=f"q{i}", arrival_s=float(i), features={"is_heavy": i < 30}
        )
        for i in range(60)
    }
    sim = DummySimulator(qs)
    slo_s = 4.0

    planner = LocalSearchPlanner(
        queries=qs,
        simulator=sim,
        slo_s=slo_s,
        target_x=1,
        k_max=4,
        rpu_ladder=[4, 8, 16, 32],
        top_h=8,
        top_m=8,
        top_l=30,
        new_cluster_sizes=[4, 8, 16, 32],
        wl_model=None,  # optionally plug your WL_model here
    )

    S = planner.solve(s0=8, max_iters=200)
    print("rho:", compute_rho(S, qs, slo_s, verbose=True))
    print("cost:", compute_cost(S))
    for cid, c in sorted(S.clusters.items()):
        print("cluster", cid, "rpu", c.rpu, "nq", len(c.queries))

    visualizer = FinalStateVisualizer()
    visualizer.plot_clusters(S, qs, "final_state.png")
