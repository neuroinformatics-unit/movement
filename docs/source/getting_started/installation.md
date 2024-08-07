(target-installation)=
# Installation

## Install the package

:::{admonition} Use a conda environment
:class: note
We recommend you install movement inside a [conda](conda:)
or [mamba](mamba:) environment, to avoid dependency conflicts with other packages. In the following we use `movement-env` as the environment name, but you can choose any name you like.
We will assume you have `conda` installed,
but the same commands will also work with `mamba`/`micromamba`.
:::

::::{tab-set}

:::{tab-item} Conda
Create and activate an environment with movement installed.
```sh
conda create -n movement-env -c conda-forge movement
conda activate movement-env
```
:::

:::{tab-item} Pip
Create and activate an environment with some prerequisites:
```sh
conda create -n movement-env -c conda-forge python=3.11 pytables
conda activate movement-env
```

Install the latest movement release from PyPI:
```sh
pip install movement
```

(Optional) Update an existing installation of movement to the latest version within the same environment:
```sh
pip install --upgrade movement
```
:::

:::{tab-item} Development
Create and activate an environment with some prerequisites:
```sh
conda create -n movement-env -c conda-forge python=3.11 pytables
conda activate movement-env
```

Clone the [GitHub repository](movement-github:) to get the latest development version and then run from within the repository:
```sh
pip install -e .[dev]  # works on most shells
pip install -e '.[dev]'  # works on zsh (the default shell on macOS)
```

This will install the package in editable mode, including all `dev` dependencies.
Please see the [contributing guide](target-contributing) for more information.
:::

::::

## Check the installation

To verify that the installation was successful, you can run the following
command (with `movement-env` activated):

```sh
movement info
```

You should see a printout including the version numbers of movement
and some of its dependencies.
