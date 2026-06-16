from __future__ import annotations

from typing import Optional


class SpinUpBudget:
    """Cumulative spin-up budget with a reserved-pool partition.

    Not thread-safe.

    Invariant at all times (finite mode):
    ``used + reserved + available == max_clusters``.

    Parameters
    ----------
    max_clusters :
        Total number of cumulative spin-ups permitted for the run.  Must
        be a non-negative integer.  ``None`` means unbounded.
    """

    def __init__(self, max_clusters: Optional[int]) -> None:
        if max_clusters is not None and (
            not isinstance(max_clusters, int) or max_clusters < 0
        ):
            raise ValueError(
                f"max_clusters must be a non-negative int, got {max_clusters!r}."
            )
        self._max_clusters = max_clusters
        self._used = 0
        self._reserved = 0
        self._available = max_clusters

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def reserve(self, n: int) -> None:
        """Move ``n`` units from ``_available`` to ``_reserved``.

        Raises :class:`ValueError` if ``n > available``.
        """
        if n < 0:
            raise ValueError(f"reserve(n) requires n >= 0, got {n}.")
        if self._max_clusters is None:
            return
        assert self._available is not None
        if n > self._available:
            raise ValueError(
                f"Cannot reserve {n} units: only {self._available} "
                f"available (max_clusters={self._max_clusters}, "
                f"used={self._used}, reserved={self._reserved})."
            )
        self._available -= n
        self._reserved += n

    def release_reservation(self, n: int) -> None:
        """Move up to ``n`` units from ``_reserved`` back to ``_available``.

        Idempotent: silently caps at the remaining reserved amount.
        """
        if n < 0:
            raise ValueError(
                f"release_reservation(n) requires n >= 0, got {n}."
            )
        if self._max_clusters is None:
            return
        assert self._available is not None
        actual = min(n, self._reserved)
        self._reserved -= actual
        self._available += actual

    # ------------------------------------------------------------------
    # Drawing budget
    # ------------------------------------------------------------------

    def try_consume(self) -> bool:
        """Atomically draw `1` unit from ``_available``.

        Returns ``True`` on success and ``False`` if insufficient budget
        is available (in which case nothing is consumed).
        """
        if self._max_clusters is None:
            self._used += 1
            return True
        assert self._available is not None
        if self._available < 1:
            return False
        self._available -= 1
        self._used += 1
        return True

    def try_consume_reserved(self) -> bool:
        """Atomically draw `1` unit from ``_reserved``.

        Used by scheduled spin-up execution, which must never fail
        as long as the upfront reservation was sized correctly.
        Returns ``False`` only if ``_reserved < 1`` (a programming bug —
        the config-load validation should have caught it).
        """
        if self._max_clusters is None:
            self._used += 1
            return True
        if self._reserved < 1:
            return False
        self._reserved -= 1
        self._used += 1
        return True

    # ------------------------------------------------------------------
    # Introspection (informational only — never use for control flow)
    # ------------------------------------------------------------------

    @property
    def max_clusters(self) -> Optional[int]:
        return self._max_clusters

    @property
    def used(self) -> int:
        return self._used

    @property
    def reserved(self) -> int:
        return self._reserved

    @property
    def available(self) -> Optional[int]:
        return self._available

    def snapshot(self) -> dict[str, int | None]:
        """Atomic snapshot of all counters, for logging/telemetry."""
        return {
            "max": self._max_clusters,
            "used": self._used,
            "reserved": self._reserved,
            "available": self._available,
        }

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        s = self.snapshot()
        return (
            f"SpinUpBudget(max={s['max']}, used={s['used']}, "
            f"reserved={s['reserved']}, available={s['available']})"
        )
