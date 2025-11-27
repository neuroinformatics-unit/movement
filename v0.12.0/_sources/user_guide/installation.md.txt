(target-installation)=
# Installation

## Create a virtual environment

While not strictly required, we strongly recommended installing `movement` in a
clean virtual environment, using tools such as
[conda](conda:) or [uv](uv:getting-started/installation/).

::::{tab-set}
:::{tab-item} conda
Create and activate a new [conda environment](conda:user-guide/tasks/manage-environments.html):
```sh
conda create -y -n movement-env -c conda-forge python=3.13
conda activate movement-env
```

We used `movement-env` as the environment name, but you can choose any name you prefer.
:::

:::{tab-item} uv
Create and activate a new [virtual environment](uv:pip/environments/) inside your project directory:

```sh
uv venv --python=3.13

source .venv/bin/activate  # On macOS and Linux
.venv\Scripts\activate     # On Windows PowerShell
```
:::
::::

## Install the package
With your environment activated, install `movement` using one of the methods below.

::::{tab-set}
:::{tab-item} From conda-forge using conda
Install the core package:
```sh
conda install -c conda-forge movement
```

If you wish to use the GUI, which requires [napari](napari:), run instead:
```sh
conda install -c conda-forge movement napari pyqt
```
You may exchange `pyqt` for `pyside6` if you prefer a different Qt backend.
See [napari's installation guide](napari:tutorials/fundamentals/installation.html)
for more details on available backends.

:::

:::{tab-item} From PyPI using pip
Install the core package:
```sh
pip install movement
```
If you wish to use the GUI, which requires [napari](napari:), run instead:
```sh
pip install "movement[napari]"
```
:::

:::{tab-item} From PyPI using uv
Install the core package:
```sh
uv pip install movement
```
If you wish to use the GUI, which requires [napari](napari:), run instead:
```sh
uv pip install "movement[napari]"
```
:::

::::

:::{dropdown} Note for Apple Silicon users with macOS 13 or earlier
:color: info
:icon: info

If you are using macOS 13 or earlier on Apple Silicon (M-series),
we recommend installing `movement` via `conda-forge`.
Alternatively, upgrade to macOS 14 to use any of the installation methods above.
:::

:::{admonition} For developers
:class: tip

If you would like to contribute to `movement`, see our [contributing guide](target-contributing)
for detailed developer setup instructions and coding guidelines.
:::

## Verify the installation
With your virtual environment activated, run:
```sh
movement info
```
You should see a printout including the version numbers of `movement`
and some of its dependencies.

To test the GUI installation:

```sh
movement launch
```

This is equivalent to running `napari -w movement` and should open the `napari`
window with the `movement` widget docked on the right-hand side.

## Update the package

:::{dropdown} Always update using the same package manager used for installation
:icon: info
:color: info

If your environment was created with `conda`, first check which channel `movement` was installed from before updating.
Run `conda list movement` in your active `conda` environment and look at the **Channel** column:
- If the channel is `conda-forge`, update using `conda`.
- If the channel is `pypi`, update using `pip`.

:::


::::{tab-set}
:::{tab-item} conda
```sh
conda update -c conda-forge movement -y
```
:::

:::{tab-item} pip
```sh
pip install -U movement
```
:::

:::{tab-item} uv
```sh
uv pip install -U movement
```
:::
::::


If the above fails, try installing `movement` in a fresh new environment to avoid dependency conflicts.

First remove the existing environment:

::::{tab-set}
:::{tab-item} conda
```sh
conda env remove -n movement-env
```

This command assumes your environment is named `movement-env`.
If you are unsure about the name, you can get a list of the environments
on your system with `conda env list`.
:::

:::{tab-item} uv
Delete the `.venv` folder in your project directory.

```powershell
rm -rf .venv       # On macOS and Linux
rmdir /s /q .venv  # On Windows PowerShell
```

Optionally, you can clean the `uv` cache for unused packages:
```sh
uv cache prune
```
:::
::::

Once the environment has been removed, you can create a new one following the [instructions](#create-a-virtual-environment) above.
