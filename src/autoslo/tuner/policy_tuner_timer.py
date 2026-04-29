import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from autoslo.filesystem.structured_events import wall_clock_utc

console = Console()


@dataclass
class PhaseTimingRecord:
    """Timing record for a single PolicyTuner pipeline phase."""

    phase_key: str
    phase_name: str
    start_wall_utc: float  # epoch seconds from wall_clock_utc()
    elapsed_s: float


class PolicyTunerTimer:
    """
    Utility class for timing the phases of a PolicyTuner run and generating a
    report.
    """

    def __init__(self) -> None:
        self._phase_timings: list[PhaseTimingRecord] = []

    @staticmethod
    def _format_duration(elapsed_s: float) -> str:
        """Format a duration in seconds as a human-readable string."""
        total = int(elapsed_s)
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{elapsed_s:.1f}s"

    @contextmanager
    def timed_phase(self, phase_key: str, phase_name: str):
        """Context manager that times a pipeline phase and appends a record."""
        start_wall = wall_clock_utc()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_s = time.perf_counter() - t0
            self._phase_timings.append(
                PhaseTimingRecord(
                    phase_key=phase_key,
                    phase_name=phase_name,
                    start_wall_utc=start_wall,
                    elapsed_s=elapsed_s,
                )
            )

    def finalize(self, out_dir: Path) -> None:
        """
        Write the timing report CSV and print a rich report to the console.
        """
        csv_path = out_dir / "timing_report.csv"
        total_s = sum(r.elapsed_s for r in self._phase_timings)
        fieldnames = [
            "phase_key",
            "phase_name",
            "start_wall_utc",
            "elapsed_s",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._phase_timings:
                writer.writerow(
                    {
                        "phase_key": r.phase_key,
                        "phase_name": r.phase_name,
                        "start_wall_utc": datetime.fromtimestamp(
                            r.start_wall_utc, tz=timezone.utc
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "elapsed_s": f"{r.elapsed_s:.3f}",
                    }
                )
            if self._phase_timings:
                writer.writerow(
                    {
                        "phase_key": "TOTAL",
                        "phase_name": "TOTAL",
                        "start_wall_utc": datetime.fromtimestamp(
                            self._phase_timings[0].start_wall_utc,
                            tz=timezone.utc,
                        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "elapsed_s": f"{total_s:.3f}",
                    }
                )

        # Print to console.
        total_s = sum(r.elapsed_s for r in self._phase_timings)
        table = Table(title="Tuning Pipeline — Timing Report", show_footer=True)
        table.add_column("Phase", footer="[bold]TOTAL[/]")
        table.add_column(
            "Duration",
            footer=f"[bold]{self._format_duration(total_s)}[/]",
            justify="right",
        )
        table.add_column("Wall start (UTC)", footer="", style="dim")
        table.add_column("% of total", footer="[bold]100%[/]", justify="right")

        for r in self._phase_timings:
            duration = self._format_duration(r.elapsed_s)
            wall_str = datetime.fromtimestamp(
                r.start_wall_utc, tz=timezone.utc
            ).strftime("%H:%M:%S")
            pct = f"{r.elapsed_s / total_s * 100:.0f}%" if total_s > 0 else "–"
            table.add_row(r.phase_name, duration, wall_str, pct)

        console.print()
        console.rule("[bold cyan]Timing Report")
        console.print(table)
        console.print(f"  Timing report saved to [bold]{csv_path}[/]")
