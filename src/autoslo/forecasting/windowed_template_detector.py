import numpy as np
from dataclasses import dataclass


@dataclass
class WindowCandidate:
    period_s: float
    idle_ratio: float
    on_window_rel_start_s: float
    p_value: float = 1.0


class WindowedTemplateDetector:

    DAY = 24 * 60 * 60

    # Define the candidate periods to search over (in seconds). This includes:
    # - Every minute from 1 to 9 minutes (60s to 540s)
    # - Every 5 minutes from 10 to 60 minutes (600s to 3600s)
    # - Every 30 minutes from 1 hour to 12 hours (3600s to 43200s)
    # - One or seven days (86400s and 604800s)
    CANDIDATE_PERIODS = np.concatenate(
        [
            np.arange(60, 10 * 60, 60),  # 1 to 9 minutes
            np.arange(10 * 60, 60 * 60, 5 * 60),  # 10 to 60 minutes
            np.arange(60 * 60, 12 * 60 * 60 + 1, 30 * 60),  # 1 to 12 hours
            np.array([DAY, 7 * DAY]),  # One or seven days
        ]
    )

    def __init__(
        self,
        arrival_times_s: list[float] | np.ndarray,
        num_permutations: int = 100,
        min_idle_ratio: float = 0.9,
        max_p_value=0.05,
        min_samples=10,
        permutations_seed=0,
    ):
        if type(arrival_times_s) == list:
            self._arrival_times_s = np.asarray(arrival_times_s, dtype=float)
        elif type(arrival_times_s) == np.ndarray:
            self._arrival_times_s = arrival_times_s
        else:
            raise ValueError(
                "arrival_times_s must be a list or np.ndarray, "
                f"but got {type(arrival_times_s)}"
            )

        self._arrival_times_s.sort()
        self._n = len(self._arrival_times_s)
        self._num_permutations = num_permutations
        self._min_idle_ratio = min_idle_ratio
        self._max_p_value = max_p_value
        self._min_samples = min_samples
        self._permutations_seed = permutations_seed

    def detect(self):

        fail_dict = {
            "num_samples": self._n,
            "is_windowed": False,
            "reason": f"Below min samples ({self._min_samples})",
            "period_s": None,
            "active_length_s": None,
            "idle_ratio": None,
            "p_value": None,
        }

        if self._n <= self._min_samples:
            return fail_dict

        # Determine candidate periods. Candidate periods should admit at least
        # two windows within the range.
        arrivals_range = self._arrival_times_s[-1] - self._arrival_times_s[0]
        all_periods = self.CANDIDATE_PERIODS
        eligible_periods = all_periods[all_periods <= arrivals_range / 2]

        # Among the candidate periods, find those w/ an okay idle ratio.
        candidates_with_okay_idle_ratio = (
            self.find_candidates_with_okay_idle_ratio(
                periods_s=eligible_periods
            )
        )
        if len(candidates_with_okay_idle_ratio) == 0:
            fail_dict["reason"] = (
                "No periods with okay idle ratio "
                f"(>= {self._min_idle_ratio})"
            )
            return fail_dict

        # Among candidates with an okay idle ratio, find those w/ an okay p-value.
        candidates_with_okay_p_value = self.find_candidates_with_okay_p_value(
            candidates=candidates_with_okay_idle_ratio
        )
        if len(candidates_with_okay_p_value) == 0:
            fail_dict["reason"] = (
                "No periods with okay p-value " f"(<= {self._max_p_value})"
            )
            return fail_dict

        # Among the rest, return the one with the longest period.
        best_candidate = candidates_with_okay_p_value[0]
        for candidate in candidates_with_okay_p_value:
            if candidate.period_s > best_candidate.period_s:
                best_candidate = candidate

        return {
            "num_samples": self._n,
            "is_windowed": True,
            "reason": None,
            "period_s": float(best_candidate.period_s),
            "active_length_s": float(
                best_candidate.period_s * (1 - best_candidate.idle_ratio)
            ),
            "idle_ratio": float(best_candidate.idle_ratio),
            "on_window_rel_start_s": float(
                best_candidate.on_window_rel_start_s
            ),
            "p_value": float(best_candidate.p_value),
        }

    def find_candidates_with_okay_idle_ratio(
        self, periods_s: list[float]
    ) -> list[WindowCandidate]:
        candidates_with_okay_idle_ratio: list[WindowCandidate] = []
        for period_s in periods_s:
            candidate = self.create_candidate(self._arrival_times_s, period_s)
            if candidate.idle_ratio >= self._min_idle_ratio:
                candidates_with_okay_idle_ratio.append(candidate)
        return candidates_with_okay_idle_ratio

    def create_candidate(self, arrival_times_s, period_s):
        folded_arrival_times_s = np.sort(np.mod(arrival_times_s, period_s))
        gaps = np.diff(
            np.r_[folded_arrival_times_s, folded_arrival_times_s[0] + period_s]
        )
        max_gap_idx = int(np.argmax(gaps))
        max_gap = gaps[max_gap_idx]
        idle_ratio = max_gap / period_s
        on_window_rel_start_s = (
            folded_arrival_times_s[max_gap_idx] + max_gap
        ) % period_s

        return WindowCandidate(
            period_s=period_s,
            idle_ratio=idle_ratio,
            on_window_rel_start_s=on_window_rel_start_s,
        )

    def find_candidates_with_okay_p_value(
        self, candidates: list[WindowCandidate]
    ):
        candidates_with_okay_p_value: list[WindowCandidate] = []

        for candidate in candidates:

            rng = np.random.default_rng(self._permutations_seed)

            arrivals_range = (
                self._arrival_times_s[-1] - self._arrival_times_s[0]
            )
            arrivals_lambda = arrivals_range / (self._n - 1)

            count = 0
            for _ in range(self._num_permutations):
                null_interarrivals_s = np.sort(
                    rng.poisson(lam=arrivals_lambda, size=self._n)
                )
                null_arrivals_s = np.cumsum(null_interarrivals_s)
                null_candidate = self.create_candidate(
                    null_arrivals_s, candidate.period_s
                )

                if null_candidate.idle_ratio >= candidate.idle_ratio:
                    count += 1

            p = (1 + count) / (1 + self._num_permutations)

            if p <= self._max_p_value:
                candidate.p_value = p
                candidates_with_okay_p_value.append(candidate)
        return candidates_with_okay_p_value
