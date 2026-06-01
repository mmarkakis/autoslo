"""Shared Rich console with optional file-tee logging for tuner runs.

All tuner components import the module-level ``console`` singleton from here
instead of creating their own ``Console()`` instances.  This allows
:class:`~autoslo.tuner.policy_tuner.PolicyTuner` to activate file mirroring
once the run directory is known, so the entire pipeline's console output is
saved to ``<run_dir>/console.log`` for later inspection.

Progress bars are managed separately (via ``rich.progress``) and are
intentionally **not** routed through this class, so they never appear in the
log file.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Any

from rich.console import Console


class TeeConsole(Console):
    """Forwards Rich ``print``/``rule`` calls to stdout *and* a log file.

    Usage::

        # At the start of a tuning run:
        console.start_file_logging(run_dir / "console.log")
        try:
            ...  # all console.print / console.rule calls land in the file too
        finally:
            console.stop_file_logging()
    """

    # Width used when rendering to the log file.  Wide enough to avoid
    # spurious table wrapping while still being readable in most editors.
    _FILE_WIDTH: int = 160

    def __init__(self) -> None:
        super().__init__()
        self._file_console: Console | None = None
        self._log_file: IO[str] | None = None

    # ------------------------------------------------------------------
    # File-logging lifecycle
    # ------------------------------------------------------------------

    def start_file_logging(self, log_path: Path) -> None:
        """Open *log_path* and start mirroring all console output to it.

        Safe to call multiple times — each call closes the previous log
        file (if any) before opening the new one.
        """
        self.stop_file_logging()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._file_console = Console(
            file=self._log_file,
            highlight=False,
            markup=True,
            width=self._FILE_WIDTH,
            no_color=True,
        )

    def stop_file_logging(self) -> None:
        """Flush and close the log file (no-op if not currently logging)."""
        if self._file_console is not None:
            self._file_console = None
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    # ------------------------------------------------------------------
    # Rich Console interface (subset used by tuner components)
    # ------------------------------------------------------------------

    def print(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        super().print(*args, **kwargs)
        if self._file_console is not None:
            self._file_console.print(*args, **kwargs)

    def rule(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # Console.rule() internally calls self.print(), which our print
        # override already mirrors to the file console — no explicit
        # _file_console.rule() call needed.
        super().rule(*args, **kwargs)


#: Module-level singleton — import this in all tuner components.
console: TeeConsole = TeeConsole()
