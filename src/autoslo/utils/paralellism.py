"""
Centralizes information
about parallelism settings used throughout the codebase.
"""

import multiprocessing as mp
from multiprocessing import context


class ParallelismConfig:
    """Central registry for parallelism settings.

    ``inner_cpus`` controls how many BLAS / OMP / Torch threads each
    worker process is allowed to use.  ``deg_of_parallelism`` is derived
    as ``cpu_count // inner_cpus``.

    Call :meth:`set_inner_cpus` early in your entry-point script to
    override the default before any workers are spawned.
    """

    _inner_cpus: int = 2

    @classmethod
    def set_inner_cpus(cls, n: int) -> None:
        """Override the per-worker BLAS / OMP thread count."""
        if n < 1:
            raise ValueError(f"inner_cpus must be >= 1, got {n}")
        cls._inner_cpus = n

    @classmethod
    def inner_cpus(cls) -> int:
        """Per-worker thread count for BLAS / OMP / Torch."""
        return cls._inner_cpus

    @classmethod
    def num_cpus(cls) -> int:
        """Total CPU count on the machine."""
        return max(1, mp.cpu_count())

    @classmethod
    def deg_of_parallelism(cls) -> int:
        """Number of concurrent worker processes."""
        return max(1, cls.num_cpus() // cls.inner_cpus())


# ---------------------------------------------------------------------------
# Backward-compatible free functions
# ---------------------------------------------------------------------------

def num_cpus() -> int:
    """Returns the number of CPUs available (delegates to
    :meth:`ParallelismConfig.num_cpus`)."""
    return ParallelismConfig.num_cpus()


def inner_level_num_cpus() -> int:
    """Returns per-worker thread count (delegates to
    :meth:`ParallelismConfig.inner_cpus`)."""
    return ParallelismConfig.inner_cpus()


def deg_of_paralellism() -> int:
    """Returns worker-process count (delegates to
    :meth:`ParallelismConfig.deg_of_parallelism`)."""
    return ParallelismConfig.deg_of_parallelism()


def _init_worker(inner_cpus: int) -> None:
    """Pool-worker initializer for ``spawn`` context.

    Sets BLAS / OpenMP thread-pool environment variables so that
    libraries imported *after* this runs honour the thread limit.
    Also synchronises :class:`ParallelismConfig` in the child process.

    Must live in a lightweight module (no numpy / torch / pandas at
    module level) so that unpickling the function reference does not
    trigger heavy imports before the env vars are set.
    """
    import os

    ncpus = str(inner_cpus)
    os.environ["OMP_NUM_THREADS"] = ncpus
    os.environ["MKL_NUM_THREADS"] = ncpus
    os.environ["OPENBLAS_NUM_THREADS"] = ncpus
    ParallelismConfig.set_inner_cpus(inner_cpus)
