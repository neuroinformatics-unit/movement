(target-installation)=
# Installation

## Install the package
:::{admonition} Use a conda environment
:class: note
To avoid dependency conflicts with other packages, it is best practice to install Python packages within a virtual environment.
We recommend using [conda](conda:) or [mamba](mamba:) to create and manage this environment, as they simplify the installation process.
The following instructions assume that you have conda installed, but the same commands will also work with `mamba`/`micromamba`.
:::

### Users
To install movement in a new environment, follow one of the options below.
We will use `movement-env` as the environment name, but you can choose any name you prefer.

::::{tab-set}
:::{tab-item} Conda
Create and activate an environment with movement installed:
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
:::
::::

### Developers
If you are a developer looking to contribute to movement, please refer to our [contributing guide](target-contributing) for detailed setup instructions and guidelines.

## Check the installation
To verify that the installation was successful, run (with `movement-env` activated):
```sh
movement info
```
You should see a printout including the version numbers of movement
and some of its dependencies.

## Update the package
To update an existing installation of movement to the latest version,
run (with `movement-env` activated):

::::{tab-set}
:::{tab-item} Conda
```sh
conda update movement
```
:::
:::{tab-item} Pip
```sh
pip install --upgrade movement
```
:::
::::
