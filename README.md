<img src="https://media.githubusercontent.com/media/basnijholt/nijho.lt/3986f45bae9ea4e834486eab4f7f6963d980c7b6/content/project/numthreads/featured.png" align="right" style="width: 200px;" />

<h1 align="center">numthreads</h1>
<h3 align="center">Set the number of threads used by OpenBLAS, MKL, OMP, NumExpr, and Accelerate</h3>

[![PyPI](https://img.shields.io/pypi/v/numthreads.svg)](https://pypi.python.org/pypi/numthreads)
[![Build Status](https://github.com/basnijholt/numthreads/actions/workflows/pytest.yml/badge.svg)](https://github.com/basnijholt/numthreads/actions/workflows/pytest.yml)
[![CodeCov](https://codecov.io/gh/basnijholt/numthreads/branch/main/graph/badge.svg)](https://codecov.io/gh/basnijholt/numthreads)
[![GitHub Repo stars](https://img.shields.io/github/stars/basnijholt/numthreads)](https://github.com/basnijholt/numthreads)
[![Documentation](https://readthedocs.org/projects/numthreads/badge/?version=latest)](https://numthreads.readthedocs.io/)

`numthreads` is a concise, easy-to-use tool designed to set the number of threads for various computing libraries including OpenBLAS, Intel's Math Kernel Library (MKL), OpenMP, NumExpr, and Accelerate.
This Python-based utility aids in optimizing the performance of numerical and scientific computing applications by allowing users to efficiently control the threading behavior of these key libraries.

- Simple and straightforward command-line interface.
- Sets thread count for OpenBLAS, MKL, OpenMP, NumExpr, and Accelerate.
- Context manager support for temporary thread setting in Python code.
- Cross-platform compatibility (Linux, macOS, Windows).
- No dependencies.

<!-- toc-start -->

## :books: Table of Contents

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [:package: Installation](#package-installation)
- [:rocket: Quick Start](#rocket-quick-start)
  - [Unix-like Systems (Linux, macOS, WSL)](#unix-like-systems-linux-macos-wsl)
  - [Windows (PowerShell)](#windows-powershell)
  - [Using as a Python Module](#using-as-a-python-module)
- [:question: Getting Help](#question-getting-help)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<!-- toc-end -->

## :package: Installation

To install `numthreads`, run the following command:

```bash
pip install "numthreads"
```

## :rocket: Quick Start

After installing `numthreads`, you can easily set the number of threads used by supported libraries via the command line. For example, to set the number of threads to 4, run:

```bash
numthreads 4
```

### Unix-like Systems (Linux, macOS, WSL)

To apply the settings in your shell:

```bash
eval $(numthreads <number_of_threads>)
```

### Windows (PowerShell)

In PowerShell, use:

```powershell
Invoke-Expression $(numthreads <number_of_threads>)
```

### Using as a Python Module

You can also use `numthreads` as a Python module:

```python
from numthreads import set_num_threads

set_num_threads(4)
```

or

```python
from numthreads import num_threads

with num_threads(4):
    # Your code here will run with the specified number of threads
    pass
```

## :question: Getting Help

For more information, or to report issues, please visit [numthreads GitHub repository](https://github.com/basnijholt/numthreads).
