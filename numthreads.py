"""numthreads: Set the number of threads for OpenBLAS, MKL, OMP, NumExpr, and Accelerate."""  # noqa: E501
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

__version__ = "0.1.0"

THREAD_CONTROL_ENV_VARS = [
    "OPENBLAS_NUM_THREADS",  # OpenBLAS
    "MKL_NUM_THREADS",  # MKL
    "OMP_NUM_THREADS",  # OMP
    "NUMEXPR_NUM_THREADS",  # NumExpr
    "VECLIB_MAXIMUM_THREADS",  # Accelerate
]


def set_num_threads(n: int = 1) -> None:
    """Set the number of threads for OpenBLAS, MKL, OMP, NumExpr, and Accelerate."""
    for var in THREAD_CONTROL_ENV_VARS:
        os.environ[var] = str(n)


@contextlib.contextmanager
def num_threads(n: int = 1) -> Generator[None, None, None]:
    """Context manager to set and then restore thread number settings."""
    original_settings = {var: os.environ.get(var) for var in THREAD_CONTROL_ENV_VARS}

    set_num_threads(n)

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


def omp_set_num_threads(num_threads: int) -> None:
    """Sets the number of threads to be used by OpenMP parallel regions.

    Parameters
    ----------
    num_threads
        Number of threads to set.
    """
    omp_lib = _load_omp_library()
    omp_lib.omp_set_num_threads(num_threads)


def omp_get_num_threads() -> int:
    """Returns the number of threads in the current OpenMP parallel region."""
    omp_lib = _load_omp_library()
    omp_lib.omp_get_num_threads.restype = ctypes.c_int
    return omp_lib.omp_get_num_threads()


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
        type=int,
        help="Number of threads to set.",
    )
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    n = args.n
    system = platform.system()
    if system == "Windows":
        export_commands = " & ".join(
            f"$env:{var}={n}" for var in THREAD_CONTROL_ENV_VARS
        )
        print(f'powershell.exe -Command "{export_commands}"')
    else:
        export_commands = " ; ".join(
            f"export {var}={n}" for var in THREAD_CONTROL_ENV_VARS
        )
        print(export_commands)


if __name__ == "__main__":
    main()
