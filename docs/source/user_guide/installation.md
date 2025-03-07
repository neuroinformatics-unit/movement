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
To create an environment with the core package only:
```sh
conda create -n movement-env -c conda-forge movement
```

If you wish to use the GUI, which additionally requires [napari](napari:),
you should instead run:
```sh
conda create -n movement-env -c conda-forge movement napari pyqt
```
You may exchange `pyqt` for `pyside2` if you prefer.
See [napari's installation guide](napari:tutorials/fundamentals/installation.html)
for more information on the backends.

To activate the environment:
```sh
conda activate movement-env
```
:::

:::{tab-item} Pip
Create and activate an environment with some prerequisites:
```sh
conda create -n movement-env -c conda-forge python=3.12 pytables
conda activate movement-env
```
Install the core package from the latest release on PyPI:
```sh
pip install movement
```
If you wish to use the GUI, which additionally requires [napari](napari:),
you should instead run:
```sh
pip install movement[napari]   # works on most shells
pip install 'movement[napari]' # works on zsh (default shell on macOS)
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

To test the GUI installation, you can run:

```sh
movement launch
```

This is equivalent to running `napari -w movement` and should open the `napari`
window with the `movement` widget docked on the right-hand side.

## Update the package
To update movement to the latest version, we recommend installing it in a new environment,
as this prevents potential compatibility issues caused by changes in dependency versions.

To uninstall an existing environment named `movement-env`:
```sh
conda env remove -n movement-env
```
:::{tip}
If you are unsure about the environment name, you can get a list of the environments on your system with:
```sh
conda env list
```
:::
Once the environment has been removed, you can create a new one following the [installation instructions](#install-the-package) above.
