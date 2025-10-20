(target-installation)=
# Installation

## Use a virtual environment

While not strictly required, it is highly recommended to install `movement` into a
clean virtual environment, using tools such as [conda](conda:) or [uv](uv:).

This should be set up before installing `movement`.

::::{tab-set}
:::{tab-item} conda
Create and activate a new environment:
```sh
conda create -y -n movement-env -c conda-forge python=3.13
conda activate movement-env
```

We used `movement-env` as the environment name, but you can choose any name you prefer.
:::

:::{tab-item} uv
Make sure [uv is installed on your system](uv:getting-started/installation).
Then create a [virtual environment](uv:pip/environments/) in your project directory:

```sh
uv init
uv venv --python=3.13
```

Activate the virtual environment:
```bash
source .venv/bin/activate  # On macOS and Linux
```
```powershell
.venv\Scripts\activate     # On Windows PowerShell
```
:::
::::

## Install the package
With your environment activated, install `movement` using one of the methods below.

::::{tab-set}
:::{tab-item} From conda-forge using conda
To install the core package:
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

If you are running macOS 13 or earlier on Apple Silicon (M-series) hardware,
we recommend installing `movement` via `conda-forge`,
which provides pre-built Qt libraries compatible with macOS 13.
Alternatively, upgrading to macOS 14 or later enables installation via `pip` or `uv`
without compatibility issues.

:::

:::{admonition} For developers
:class: tip

If you plan to contribute to `movement`, see the [contributing guide](target-contributing)
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

Always update using the same package manager you used for installation.

If you are not sure which one you used, you can run `conda list movement` from your active `conda` environment
and check whether the output shows `conda-forge` or `pypi` as the channel.

To update `movement`, run the appropriate command below:

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
Simply delete the `.venv` folder in your project directory.

```bash
rm -rf .venv       # On macOS and Linux
```
```powershell
rmdir /s /q .venv  # On Windows PowerShell
```

Optionally, you may also clean the `uv` cache for unused packages:
```sh
uv cache prune
```
:::
::::

Once the environment has been removed, you can create a new one following the [instructions](#use-a-virtual-environment) above.
