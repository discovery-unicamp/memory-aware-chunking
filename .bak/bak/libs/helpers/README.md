# Helpers Library

## Overview

The Helpers Library is designed to centralize and organize the utility code used in various experiments.
It includes tools for generating synthetic datasets, managing Dask operators, and other essential functions required for executing and benchmarking experiments.

## Features

- **Synthetic Dataset Generation**  
  Easily create synthetic seismic datasets for testing and simulation purposes.

- **Dask Operator Management**  
  Store and retrieve Dask operators to streamline experimentation with Dask-based workflows.

## Installation

Within the Jupyter notebook of the experiment you can do the following:

```python
import sys
import os

helpers_path = os.path.abspath('../libs/helpers')
helpers_path not in sys.path and sys.path.append(helpers_path)
```

## Usage

### Generating Synthetic Datasets

To generate a synthetic seismic dataset:

```python
from helpers.datasets import generate_seismic_data

data = generate_seismic_data(samples=1000, inlines=20)
```

### Using Dask Operators

Load and apply Dask operators for experiments:

```python
from helpers.dask_operators import my_dask_operator

result = my_dask_operator(data)
```