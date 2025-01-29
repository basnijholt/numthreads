"""Tests for ``numthreads`` package."""
import os

import pytest

from numthreads import (
    num_threads,
    omp_get_num_threads,
    omp_num_threads,
    omp_set_num_threads,
    print_current_thread_counts,
    set_num_threads,
)


@pytest.mark.parametrize("overwrite", [True, False])
def test_set_num_threads(overwrite: bool) -> None:  # noqa: FBT001
    envvars = [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    prev_values = {var: os.environ.get(var) for var in envvars}
    set_num_threads(4, overwrite=overwrite)
    for var in envvars:
        if overwrite:
            assert os.environ.get(var) == "4"
        else:
            assert os.environ.get(var) == prev_values[var]


@pytest.mark.parametrize("overwrite", [True, False])
def test_num_threads_context_manager(overwrite: bool) -> None:  # noqa: FBT001
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

    with num_threads(2, overwrite=overwrite):
        for var in [
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ]:
            if overwrite:
                assert os.environ.get(var) == "2"
            else:
                assert os.environ.get(var) == original_values[var]

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
@pytest.mark.parametrize("overwrite", [True, False])
def test_omp_set_get_num_threads(overwrite: bool) -> None:  # noqa: FBT001
    omp_set_num_threads(4, overwrite=overwrite)

    # Now this will fail: assert omp_get_num_threads() == 4
    # because we need to call this in a parallel region.
    prev_value = omp_get_num_threads()
    omp_set_num_threads(1, overwrite=overwrite)
    if overwrite:
        assert omp_get_num_threads() == 1
    else:
        assert omp_get_num_threads() == prev_value


@pytest.mark.skipif(
    os.name != "posix",
    reason="OMP functions are tested only on POSIX systems",
)
def test_omp_num_threads() -> None:
    with omp_num_threads(4):
        # Now this will fail: assert omp_get_num_threads() == 4
        # because we need to call this in a parallel region.
        pass


def test_print_current_thread_counts(capsys: pytest.CaptureFixture) -> None:
    set_num_threads(4)
    print_current_thread_counts()
    assert "OPENBLAS_NUM_THREADS: 4" in capsys.readouterr().out
