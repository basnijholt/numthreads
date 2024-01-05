"""Tests for ``numthreads`` package."""
import os

import pytest

from numthreads import (
    num_threads,
    omp_get_num_threads,
    omp_num_threads,
    omp_set_num_threads,
    set_num_threads,
)


def test_set_num_threads() -> None:
    set_num_threads(4)
    for var in [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        assert os.environ.get(var) == "4"


def test_num_threads_context_manager() -> None:
    original_values = {
        var: os.environ.get(var)
        for var in [
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]
    }

    with num_threads(2):
        for var in [
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]:
            assert os.environ.get(var) == "2"

    for var in [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        assert os.environ.get(var) == original_values[var]


@pytest.mark.skipif(
    os.name != "posix",
    reason="OMP functions are tested only on POSIX systems",
)
def test_omp_set_get_num_threads() -> None:
    omp_set_num_threads(4)

    # Now this will fail: assert omp_get_num_threads() == 4
    # because we need to call this in a parallel region.

    omp_set_num_threads(1)
    assert omp_get_num_threads() == 1


@pytest.mark.skipif(
    os.name != "posix",
    reason="OMP functions are tested only on POSIX systems",
)
def test_omp_num_threads() -> None:
    with omp_num_threads(4):
        # Now this will fail: assert omp_get_num_threads() == 4
        # because we need to call this in a parallel region.
        pass
