"""
Centralizes information
about parallelism settings used throughout the codebase.
"""

import multiprocessing as mp
from multiprocessing import context


INNER_LEVEL_NUM_CPUS_CONSTANT = 4


def num_cpus() -> int:
    """
    Returns the number of CPUs to use for parallel processing tasks.

    Returns:
        int: Number of CPUs to use.
    """
    return max(1, mp.cpu_count())


def inner_level_num_cpus() -> int:
    """
    Returns the number of CPUs to use for inner-level parallel processing tasks.

    Returns:
        int: Number of CPUs to use for inner-level tasks.
    """
    return max(1, num_cpus() // INNER_LEVEL_NUM_CPUS_CONSTANT)


def deg_of_paralellism() -> int:
    """
    Returns the degree of parallelism to use for parallel processing tasks.

    Returns:
        int: Degree of parallelism.
    """
    return num_cpus() // inner_level_num_cpus()
