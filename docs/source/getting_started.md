# Getting Started

## Installation

We recommend you use install `movement` inside a [conda](https://docs.conda.io/en/latest/)
or [mamba](https://mamba.readthedocs.io/en/latest/index.html) environment.
In the following we assume you have `conda` installed,
but the same commands will also work with `mamba`/`micromamba`.


First, create and activate an environment.
You can call your environment whatever you like, we've used `movement-env`.

```sh
conda create -n movement-env -c conda-forge python=3.10 pytables
conda activate movement-env
```

Next install the `movement` package:

::::{tab-set}

:::{tab-item} Users
To get the latest release from PyPI:

```sh
pip install movement
```
If you have an older version of `movement` installed in the same environment,
you can update to the latest version with:

```sh
pip install --upgrade movement
```
:::

:::{tab-item} Developers
To get the latest development version, clone the
[GitHub repository](https://neuroinformatics-unit.github.io/movement/)
and then run from inside the repository:

```sh
pip install -e .[dev]  # works on most shells
pip install -e '.[dev]'  # works on zsh (the default shell on macOS)
```

This will install the package in editable mode, including all `dev` dependencies.
Please see the [contributing guide](./contributing.rst) for more information.
:::

::::


## Usage

### Loading data
You can load predicted pose tracks for the pose estimation software packages
[DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) or [SLEAP](https://sleap.ai/).

First import the `load_poses` function from the `movement.io` module:

```python
from movement.io import load_poses
```

Then, use the `from_dlc_file` or `from_sleap_file` functions to load the data.

::::{tab-set}

:::{tab-item} SLEAP

Load from [SLEAP analysis files](https://sleap.ai/tutorials/analysis.html) (`.h5`):
```python
ds = load_poses.from_sleap_file("/path/to/file.analysis.h5", fps=30)
```
:::

:::{tab-item} DeepLabCut

Load pose estimation outputs from `.h5` files:
```python
ds = load_poses.from_dlc_file("/path/to/file.h5", fps=30)
```

You may also load `.csv` files (assuming they are formatted as DeepLabCut expects them):
```python
ds = load_poses.from_dlc_file("/path/to/file.csv", fps=30)
```

If you have already imported the data into a pandas DataFrame, you can
convert it to a `movement` dataset with:
```python
import pandas as pd

df = pd.read_hdf("/path/to/file.h5")
ds = load_poses.from_dlc_df(df, fps=30)
```
:::

::::
