(target-installation)=
# Installation

## Create a conda environment

:::{admonition} Use a conda environment
:class: note
We recommend you install movement inside a [conda](conda:)
or [mamba](mamba:) environment, to avoid dependency conflicts with other packages.
In the following we assume you have `conda` installed,
but the same commands will also work with `mamba`/`micromamba`.
:::

First, create and activate an environment with some prerequisites.
You can call your environment whatever you like, we've used `movement-env`.

```sh
conda create -n movement-env -c conda-forge python=3.11 pytables
conda activate movement-env
```

## Install the package

Then install the `movement` package as described below.

::::{tab-set}

:::{tab-item} Without the GUI
To install the core package from PyPI:

```sh
pip install movement
```
If you have an older version of `movement` installed in the same environment,
you can update to the latest version with:

```sh
pip install --upgrade movement
```
:::

:::{tab-item} With the GUI
To install the package including the GUI (napari plugin) from PyPI:

```sh
pip install "movement[napari]"
```
If you have an older version of `movement` installed in the same environment,
you can update to the latest version with:

```sh
pip install --upgrade "movement[napari]"
```
:::

:::{tab-item} For developers
To get the latest development version, clone the
[GitHub repository](movement-github:)
and then run from inside the repository:

```sh
pip install -e .[dev]  # works on most shells
pip install -e ".[dev]"  # works on zsh (the default shell on macOS)
```

This will install the package in editable mode, including all `dev` dependencies.
Please see the [contributing guide](target-contributing) for more information.
:::

::::

## Check the installation

To verify that the installation was successful, you can run the following
command (with the `movement-env` activated):

```sh
movement info
```

You should see a printout including the version numbers of `movement`
and some of its dependencies.

To test the GUI installation, you can run:

```sh
napari -w movement
```

This should open a new `napari` window with the `movement` plugin loaded
on the right side.
