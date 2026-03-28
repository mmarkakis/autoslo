"""WorkloadSampler — draw concrete workloads from a QueryReservoir."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from autoslo.tuner.forecast_policy import ForecastPolicy
from autoslo.tuner.reservoir import QueryReservoir

logger = logging.getLogger(__name__)


class WorkloadSampler:
    """Sample *N* synthetic workloads from a :class:`QueryReservoir`.

    Each sampled workload covers the time range
    ``[target_start, target_end)`` broken into hour-aligned bins.  For each
    bin, the :class:`~autoslo.tuner.forecast_policy.ForecastPolicy` decides
    how many queries to generate and how to weight the historical
    observations.  Query ``abs_start_time`` values are drawn from a Poisson
    process within each hour bin.

    Parameters
    ----------
    reservoir :
        Historical query-arrival data.
    forecast_policy :
        Policy that assigns weights and expected counts to each bin.
    schema_name :
        Schema used to label the resulting :class:`Workload` objects.
    """

    def __init__(
        self,
        reservoir: QueryReservoir,
        forecast_policy: ForecastPolicy,
        schema_name: str,
    ) -> None:
        self.reservoir = reservoir
        self.forecast_policy = forecast_policy
        self.schema_name = schema_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        target_start: datetime,
        target_end: datetime,
        n_scenarios: int,
        seed: int = 42,
    ) -> list:
        """Return *n_scenarios* sampled :class:`Workload` objects.

        Parameters
        ----------
        target_start / target_end :
            The time window to forecast.  Will be split into hour-aligned
            bins.
        n_scenarios :
            Number of independent workloads to sample.
        seed :
            Base random seed.  Scenario *i* uses ``seed + i``.

        Returns
        -------
        list[Workload]
        """
        from autoslo.workload_definition.workload import Workload

        hour_bins = self._make_hour_bins(target_start, target_end)

        workloads = []
        for i in range(n_scenarios):
            rng = np.random.default_rng(seed + i)
            rows = self._sample_one(hour_bins, rng)
            name = f"tuner_scenario_{i:03d}"
            wl = self._rows_to_workload(rows, name)
            workloads.append(wl)

        return workloads

    def sample_to_disk(
        self,
        target_start: datetime,
        target_end: datetime,
        n_scenarios: int,
        out_dir: Path,
        prefix: str = "s",
        seed: int = 42,
    ) -> list[Path]:
        """Sample workloads and persist each as a parquet file.

        Parameters
        ----------
        out_dir :
            Directory in which to write parquet files.
        prefix :
            Filename prefix (e.g. ``"t"`` for train, ``"v"`` for val).

        Returns
        -------
        list[Path]
            Paths to the written parquet files.
        """
        workloads = self.sample(target_start, target_end, n_scenarios, seed)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for idx, wl in enumerate(workloads):
            p = out_dir / f"{prefix}_{idx:03d}.parquet"
            wl.df.to_parquet(p, index=False)
            paths.append(p)

        return paths

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hour_bins(
        start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Break ``[start, end)`` into hour-aligned (bin_start, bin_end) pairs."""
        bins: list[tuple[datetime, datetime]] = []
        current = start.replace(minute=0, second=0, microsecond=0)
        if current < start:
            current = start  # first bin may be partial
        while current < end:
            bin_end = (current + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )
            if bin_end > end:
                bin_end = end
            if current < bin_end:
                bins.append((current, bin_end))
            current = bin_end
        return bins

    def _sample_one(
        self,
        hour_bins: list[tuple[datetime, datetime]],
        rng: np.random.Generator,
    ) -> list[dict]:
        """Sample query arrivals for one scenario across all hour bins."""
        all_rows: list[dict] = []
        classifications = self.reservoir.meta.get("classifications", {})

        # Identify windowed template IDs and their parameters.
        windowed_templates: dict[str, dict] = {}
        for group_id, info in classifications.items():
            if info.get("classification") == "windowed":
                windowed_templates[group_id] = info

        for bin_start, bin_end in hour_bins:
            dow = bin_start.weekday()
            hour = bin_start.hour
            bin_duration_s = (bin_end - bin_start).total_seconds()

            # Get historical observations for matching hour.
            reservoir_bin = self.reservoir.bin_df(dow, hour)
            if reservoir_bin.empty:
                # Try any day-of-week with the same hour.
                reservoir_bin = self.reservoir.df[
                    self.reservoir.df["hour"] == hour
                ].reset_index(drop=True)
            if reservoir_bin.empty:
                continue

            # Partition into normal and windowed templates.
            grouping_key = "repetition_id"
            windowed_mask = reservoir_bin[grouping_key].isin(windowed_templates)
            normal_bin = reservoir_bin[~windowed_mask].reset_index(drop=True)
            windowed_bin = reservoir_bin[windowed_mask].reset_index(drop=True)

            # --- Windowed templates: insert at periodic positions ---
            if not windowed_bin.empty:
                for group_id, group_df in windowed_bin.groupby(grouping_key):
                    info = windowed_templates.get(str(group_id), {})
                    period_s = info.get("period_s")
                    active_length_s = info.get("active_length_s")
                    on_window_rel_start_s = info.get("on_window_rel_start_s", 0.0)

                    if period_s is None or period_s <= 0:
                        # Fall back to Poisson for this group.
                        normal_bin = pd.concat(
                            [normal_bin, group_df], ignore_index=True
                        )
                        continue

                    # Determine periodic arrival times within this bin.
                    # Use absolute time: find the first window start at or
                    # after bin_start.
                    bin_start_epoch = bin_start.timestamp()
                    bin_end_epoch = bin_end.timestamp()

                    # Compute the start of the first period window at or
                    # after on_window_rel_start_s.  We anchor periods to
                    # the epoch so they're absolute.
                    first_window = on_window_rel_start_s
                    if first_window < bin_start_epoch:
                        # Advance to the first period start within the bin.
                        n_periods = int(
                            (bin_start_epoch - first_window) / period_s
                        )
                        first_window += n_periods * period_s
                        if first_window < bin_start_epoch:
                            first_window += period_s

                    # Pick a representative query_text_id from this group.
                    qtid = str(group_df["query_text_id"].iloc[0])

                    window_start = first_window
                    while window_start < bin_end_epoch:
                        window_end = window_start + (active_length_s or 0.0)
                        # Clip to the bin.
                        effective_start = max(window_start, bin_start_epoch)
                        effective_end = min(window_end, bin_end_epoch)
                        if effective_start < effective_end:
                            # Place one arrival at a random point within
                            # the active window.
                            offset = rng.uniform(
                                effective_start - bin_start_epoch,
                                effective_end - bin_start_epoch,
                            )
                            abs_time = bin_start + timedelta(seconds=float(offset))
                            all_rows.append(
                                {
                                    "abs_start_time": abs_time,
                                    "query_text_id": qtid,
                                    "repetition_id": str(group_id),
                                }
                            )
                        window_start += period_s

            # --- Normal templates: Poisson process ---
            if normal_bin.empty:
                continue

            # Assign a day index to each historical observation so the
            # forecast policy can compute weighted per-day counts.
            normal_bin = normal_bin.copy()
            n_workloads = max(1, self.reservoir.meta.get("num_workloads", 1))
            normal_bin["__obs_day_idx"] = np.arange(len(normal_bin)) % n_workloads

            # Compute per-observation-day weights.
            weights: list[float] = []
            for day_idx in range(n_workloads):
                days_back = (n_workloads - day_idx) * 7
                obs_start = bin_start - timedelta(days=days_back)
                obs_start = obs_start.replace(hour=hour, minute=0, second=0, microsecond=0)
                obs_end = obs_start + timedelta(hours=1)
                w = self.forecast_policy.weight(
                    (obs_start, obs_end), (bin_start, bin_end)
                )
                weights.append(w)

            expected = self.forecast_policy.expected_count(
                (bin_start, bin_end), normal_bin, weights
            )

            if expected <= 0:
                continue

            # Scale expected count if this is a partial hour.
            if bin_duration_s < 3600:
                expected = max(1, round(expected * bin_duration_s / 3600.0))

            # Draw arrival times via a Poisson process.
            rate = expected / bin_duration_s
            inter_arrivals = rng.exponential(1.0 / rate, size=expected * 2)
            arrival_offsets = np.cumsum(inter_arrivals)
            arrival_offsets = arrival_offsets[arrival_offsets < bin_duration_s]
            arrival_offsets = arrival_offsets[:expected]

            # Sample query_text_ids from the normal pool.
            pool = normal_bin["query_text_id"].values
            chosen_ids = rng.choice(pool, size=len(arrival_offsets), replace=True)

            for offset_s, qtid in zip(arrival_offsets, chosen_ids):
                abs_time = bin_start + timedelta(seconds=float(offset_s))
                all_rows.append(
                    {
                        "abs_start_time": abs_time,
                        "query_text_id": str(qtid),
                        "repetition_id": str(qtid),
                    }
                )

        return all_rows

    def _rows_to_workload(self, rows: list[dict], name: str):
        """Convert raw row dicts into a Workload."""
        from autoslo.workload_definition.workload import Workload

        if not rows:
            # Return an empty workload with the correct schema.
            df = pd.DataFrame(
                columns=["query_id", "abs_start_time", "query_text_id", "repetition_id"]
            )
        else:
            df = pd.DataFrame(rows)
            df = df.sort_values("abs_start_time").reset_index(drop=True)
            df["query_id"] = [f"{name}_{seq:05d}" for seq in range(len(df))]

        return Workload(name, self.schema_name, df=df)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def preview(
        self,
        target_start: datetime,
        target_end: datetime,
    ) -> pd.DataFrame:
        """Return a summary of expected query counts per hour bin.

        Useful for sanity-checking the reservoir + forecast policy before
        running the full sampling pipeline.
        """
        hour_bins = self._make_hour_bins(target_start, target_end)
        records = []
        for bin_start, bin_end in hour_bins:
            dow = bin_start.weekday()
            hour = bin_start.hour

            reservoir_bin = self.reservoir.bin_df(dow, hour)
            n_workloads = max(1, self.reservoir.meta.get("num_workloads", 1))

            # Compute expected count using the forecast policy.
            if reservoir_bin.empty:
                expected = 0
                n_unique_qtids = 0
            else:
                reservoir_bin = reservoir_bin.copy()
                reservoir_bin["__obs_day_idx"] = (
                    np.arange(len(reservoir_bin)) % n_workloads
                )
                weights = []
                for day_idx in range(n_workloads):
                    days_back = (n_workloads - day_idx) * 7
                    obs_start = bin_start - timedelta(days=days_back)
                    obs_start = obs_start.replace(
                        hour=hour, minute=0, second=0, microsecond=0
                    )
                    obs_end = obs_start + timedelta(hours=1)
                    w = self.forecast_policy.weight(
                        (obs_start, obs_end),
                        (bin_start, bin_start + timedelta(hours=1)),
                    )
                    weights.append(w)
                expected = self.forecast_policy.expected_count(
                    (bin_start, bin_start + timedelta(hours=1)),
                    reservoir_bin,
                    weights,
                )
                n_unique_qtids = reservoir_bin["query_text_id"].nunique()

            records.append(
                {
                    "bin_start": bin_start,
                    "day_of_week": dow,
                    "hour": hour,
                    "reservoir_rows": len(reservoir_bin),
                    "unique_query_text_ids": n_unique_qtids,
                    "expected_count": expected,
                }
            )

        return pd.DataFrame(records)
