"""numthreads: Set the number of threads for OpenBLAS, MKL, OMP, NumExpr, Accelerate, and Numba."""  # noqa: E501

from __future__ import annotations

import argparse
import contextlib
import ctypes
import os
import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    import pytest

__version__ = "0.6.1"

THREAD_CONTROL_ENV_VARS = [
    "OPENBLAS_NUM_THREADS",  # OpenBLAS
    "MKL_NUM_THREADS",  # MKL
    "OMP_NUM_THREADS",  # OMP
    "NUMEXPR_NUM_THREADS",  # NumExpr
    "VECLIB_MAXIMUM_THREADS",  # Accelerate
    "NUMBA_NUM_THREADS",  # Numba
]


def set_num_threads(n: int = 1, *, overwrite: bool = True) -> None:
    """Set the number of threads via environment variables.

    Sets the following environment variables:

    - ``"OPENBLAS_NUM_THREADS"`` for OpenBLAS (Open Basic Linear Algebra Subprograms)
    - ``"MKL_NUM_THREADS"`` for MKL (Intel Math Kernel Library)
    - ``"OMP_NUM_THREADS"`` for OMP (OpenMP)
    - ``"NUMEXPR_NUM_THREADS"`` for NumExpr (NumPy expression evaluator)
    - ``"VECLIB_MAXIMUM_THREADS"`` for Accelerate (macOS)

    Note that setting these environment variables typically needs to be done before
    importing the relevant libraries. For example, if you want to set the number of
    threads for NumPy, you should set the environment variables before importing
    NumPy. When using OMP (OpenMP), you can use the ``omp_set_num_threads`` function
    to change the number of threads *after* importing the library.

    Parameters
    ----------
    n
        Number of threads to set.
    overwrite
        If true, overwrite the existing value.

    """
    for var in THREAD_CONTROL_ENV_VARS:
        if overwrite or var not in os.environ:
            os.environ[var] = str(n)


# Function alias
set = set_num_threads  # noqa: A001


@contextlib.contextmanager
def num_threads(n: int = 1, *, overwrite: bool = True) -> Generator[None, None, None]:
    """Context manager to set and then restore thread number settings."""
    original_settings = {var: os.environ.get(var) for var in THREAD_CONTROL_ENV_VARS}

    set_num_threads(n, overwrite=overwrite)

    try:
        yield
    finally:
        for var, value in original_settings.items():
            if value is None:  # pragma: no cover
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def _load_omp_library() -> ctypes.CDLL:
    """Loads the OpenMP library based on the operating system."""
    system = platform.system()

    if system == "Darwin":  # pragma: no cover
        lib_name = "libomp.dylib"
    elif system == "Linux":  # pragma: no cover
        lib_name = "libgomp.so.1"
    elif system == "Windows":  # pragma: no cover
        lib_name = "libiomp5md.dll"
    else:  # pragma: no cover
        msg = f"Unsupported operating system: {system}"
        raise NotImplementedError(msg)

    try:
        return ctypes.CDLL(lib_name)
    except OSError as e:  # pragma: no cover
        msg = f"Error loading {lib_name}. Make sure OpenMP is installed."
        raise OSError(msg) from e


def omp_set_num_threads(num_threads: int, *, overwrite: bool = True) -> None:
    """Sets the number of threads to be used by OpenMP parallel regions.

    When using OMP (OpenMP), you can use this function to change the number of threads
    *after* importing the library. This overrides whatever value is set in the
    ``"OMP_NUM_THREADS"`` environment variable.

    Parameters
    ----------
    num_threads
        Number of threads to set.
    overwrite
        If true, overwrite the existing value.
    """
    if overwrite:
        omp_lib = _load_omp_library()
        omp_lib.omp_set_num_threads.argtypes = [ctypes.c_int]
        omp_lib.omp_set_num_threads(num_threads)


def omp_get_num_threads() -> int:
    """Returns the number of threads in the current OpenMP parallel region."""
    omp_lib = _load_omp_library()
    omp_lib.omp_get_num_threads.restype = ctypes.c_int
    return omp_lib.omp_get_num_threads()


@contextlib.contextmanager
def omp_num_threads(
    num_threads: int,
    *,
    overwrite: bool = True,
) -> Generator[None, None, None]:
    """Context manager to set and then restore OpenMP thread number settings.

    Parameters
    ----------
    num_threads
        Number of threads to set in the OpenMP parallel regions.
    overwrite
        Whether to overwrite existing values.

    Examples
    --------
    >>> import numpy as np
    >>> from numthreads import omp_num_threads
    >>> with omp_num_threads(2):
    ...     np.ones(10)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    """
    original_num_threads = omp_get_num_threads()
    omp_set_num_threads(num_threads, overwrite=overwrite)

    try:
        yield
    finally:
        omp_set_num_threads(original_num_threads)


def print_current_thread_counts() -> None:
    """Prints the current values of the thread control environment variables."""
    for var in THREAD_CONTROL_ENV_VARS:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")


def main() -> None:  # pragma: no cover
    """Command-line interface."""
    description = (
        "Set the number of threads for OpenBLAS, MKL, OMP, NumExpr, and Accelerate."
        " Usage: Run `numthreads <number>` to print the export commands."
        " On Unix-like systems (Linux, macOS, WSL), use `eval $(numthreads <number>)`"
        " in your shell to apply these settings."
        " On Windows, in PowerShell, use `Invoke-Expression $(numthreads <number>)`."
    )
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument(
        "n",
        help="Number of threads to set or use 'get' to display current settings.",
    )
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    if args.n == "get":
        print_current_thread_counts()
        return

    n = int(args.n)
    system = platform.system()
    if system == "Windows":
        export_commands = " & ".join(
            f"$env:{var}='{n}'" for var in THREAD_CONTROL_ENV_VARS
        )
        print(f'powershell.exe -Command "{export_commands}"')
    else:
        export_commands = " ; ".join(
            f"export {var}='{n}'" for var in THREAD_CONTROL_ENV_VARS
        )
        print(export_commands)


# Pytest Plugin Integration
def pytest_addoption(parser: pytest.Parser) -> None:  # pragma: no cover
    """Add the --numthreads option to the pytest command line."""
    parser.addoption(
        "--numthreads",
        action="store",
        default=None,  # Default to None, so we don't set unnecessarily
        type=int,
        help="Number of threads to set for OpenBLAS, MKL, OMP, NumExpr, and Accelerate",
    )


def pytest_configure(config: pytest.Config) -> None:  # pragma: no cover
    """Set the number of threads based on the --numthreads option."""
    numthreads = config.getoption("--numthreads")
    if numthreads is not None:
        set_num_threads(numthreads)


if __name__ == "__main__":
    main()
