<img src="https://media.githubusercontent.com/media/basnijholt/nijho.lt/3986f45bae9ea4e834486eab4f7f6963d980c7b6/content/project/numthreads/featured.png" align="right" style="width: 200px;" />

<h1 align="center">numthreads</h1>
<h3 align="center">Set the number of threads used by OpenBLAS, MKL, OMP, NumExpr, and Accelerate</h3>

[![PyPI](https://img.shields.io/pypi/v/numthreads.svg)](https://pypi.python.org/pypi/numthreads)
[![Build Status](https://github.com/basnijholt/numthreads/actions/workflows/pytest.yml/badge.svg)](https://github.com/basnijholt/numthreads/actions/workflows/pytest.yml)
[![CodeCov](https://codecov.io/gh/basnijholt/numthreads/branch/main/graph/badge.svg)](https://codecov.io/gh/basnijholt/numthreads)
[![GitHub Repo stars](https://img.shields.io/github/stars/basnijholt/numthreads)](https://github.com/basnijholt/numthreads)
[![Documentation](https://readthedocs.org/projects/numthreads/badge/?version=latest)](https://numthreads.readthedocs.io/)

`numthreads` is a really tiny and simple Python package designed to set the number of threads for various computing libraries including OpenBLAS, Intel's Math Kernel Library (MKL), OpenMP, NumExpr, and Accelerate.
The number of threads can be set via the command line or in Python code.
The performance of many numerical algorithms varies significantly based on the number of threads employed.
While increasing the number of threads can often accelerate these algorithms, it's not always the case.
In some instances, using more threads may actually impede computational efficiency.
Therefore, it's important to be able to easily set the number of threads used by these libraries.

- Simple and straightforward command-line interface.
- Sets thread count for OpenBLAS, MKL, OpenMP, NumExpr, and Accelerate.
- Context manager support for temporary thread setting in Python code.
- Cross-platform compatibility (Linux, macOS, Windows).
- Tiny (≤7KB) and no dependencies.

<!-- toc-start -->

## :books: Table of Contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [:package: Installation](#package-installation)
- [:rocket: Quick Start](#rocket-quick-start)
  - [Using as a Python Module](#using-as-a-python-module)
  - [Command Line Interface](#command-line-interface)
    - [Unix-like Systems (Linux, macOS, WSL)](#unix-like-systems-linux-macos-wsl)
    - [Windows (PowerShell)](#windows-powershell)
  - [Get the number of threads](#get-the-number-of-threads)
- [:question: Getting Help](#question-getting-help)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<!-- toc-end -->

## :package: Installation

To install `numthreads`, run the following command:

```bash
pip install numthreads
```

## :rocket: Quick Start

Get started with `numthreads` in a few seconds.

### Using as a Python Module

You can also use `numthreads` as a Python module:

```python
from numthreads import set_num_threads

set_num_threads(4)
```

This will set the number of threads using the following environment variables:
- OpenBLAS (via `OPENBLAS_NUM_THREADS`)
- MKL (via `MKL_NUM_THREADS`)
- OpenMP (via `OMP_NUM_THREADS`)
- NumExpr (via `NUMEXPR_NUM_THREADS`)
- Accelerate (via `VECLIB_MAXIMUM_THREADS`)

or use it as a context manager:

```python
from numthreads import num_threads

with num_threads(4):
    # Your code here will run with the specified number of threads
    pass
```

> [!WARNING]
> Since environment variables are global and typically need to be set before importing any libraries, it's recommended to set the number of threads at the beginning of your Python script.

To set OMP (OpenMP) threads at any time, you can use `omp_set_num_threads` or the `omp_num_threads` context manager:

```python
from numthreads import omp_set_num_threads

omp_set_num_threads(4)
```

or

```python
from numthreads import omp_num_threads

with omp_num_threads(4):
    # Your code here will run with the specified number of threads
    pass
```

### Command Line Interface

After installing `numthreads`, you can easily set the number of threads used by supported libraries via the command line. For example, to print the command to set the number of threads to 4, run:

```bash
numthreads 4
```
<!-- CODE:BASH:START -->
<!-- echo '```bash' -->
<!-- numthreads 4 -->
<!-- echo '```' -->
<!-- CODE:END -->
Which will print the following:
<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```bash
export OPENBLAS_NUM_THREADS='4' ; export MKL_NUM_THREADS='4' ; export OMP_NUM_THREADS='4' ; export NUMEXPR_NUM_THREADS='4' ; export VECLIB_MAXIMUM_THREADS='4'
```

<!-- OUTPUT:END -->

#### Unix-like Systems (Linux, macOS, WSL)

To apply the settings in your shell:

```bash
eval $(numthreads <number_of_threads>)
```

#### Windows (PowerShell)

In PowerShell, use:

```powershell
Invoke-Expression $(numthreads <number_of_threads>)
```

### Get the number of threads

To get the number of threads currently set, run:

```bash
eval $(numthreads 12)  # first set something
numthreads get
```
<!-- CODE:BASH:START -->
<!-- echo '```bash' -->
<!-- eval $(numthreads 12) -->
<!-- numthreads get -->
<!-- echo '```' -->
<!-- CODE:END -->
Which will print the following:
<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```bash
OPENBLAS_NUM_THREADS: 12
MKL_NUM_THREADS: 12
OMP_NUM_THREADS: 12
NUMEXPR_NUM_THREADS: 12
VECLIB_MAXIMUM_THREADS: 12
```

<!-- OUTPUT:END -->

Check the `numthreads -h` message for more information:

<!-- CODE:BASH:START -->
<!-- echo '```bash' -->
<!-- numthreads -h -->
<!-- echo '```' -->
<!-- CODE:END -->
<!-- OUTPUT:START -->
<!-- ⚠️ This content is auto-generated by `markdown-code-runner`. -->
```bash
usage: numthreads [-h] n

Set the number of threads for OpenBLAS, MKL, OMP, NumExpr, and Accelerate.
Usage: Run `numthreads <number>` to print the export commands. On Unix-like
systems (Linux, macOS, WSL), use `eval $(numthreads <number>)` in your shell
to apply these settings. On Windows, in PowerShell, use `Invoke-Expression
$(numthreads <number>)`.

positional arguments:
  n           Number of threads to set or use 'get' to display current
              settings.

options:
  -h, --help  show this help message and exit
```

<!-- OUTPUT:END -->

## :question: Getting Help

For more information, or to report issues, please visit [numthreads GitHub repository](https://github.com/basnijholt/numthreads).
