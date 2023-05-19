[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
![CI](https://img.shields.io/github/actions/workflow/status/neuroinformatics-unit/movement/test_and_deploy.yml?label=CI)
[![codecov](https://codecov.io/gh/neuroinformatics-unit/movement/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/neuroinformatics-unit/movement)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# movement

Kinematic analysis of animal 🐝 🦀 🐀 🐒 body movements for neuroscience and ethology research 🔬.

## Status
The package is currently in early development 🏗️ and is not yet ready for use. Stay tuned ⌛

## Aims
* Load keypoint tracks from pose estimation software (e.g. [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) or [SLEAP](https://sleap.ai/))
* Evaluate the quality of the tracks and perform data cleaning
* Calculate kinematic variables (e.g. speed, acceleration, joint angles, etc.)
* Produce reports and visualise the results

## Related projects
The following projects cover related needs and served as inspiration for this project:
* [DLC2Kinematics](https://github.com/AdaptiveMotorControlLab/DLC2Kinematics)
* [PyRat](https://github.com/pyratlib/pyrat)
* [Kino](https://github.com/BrancoLab/Kino)
* [WAZP](https://github.com/SainsburyWellcomeCentre/WAZP)

## How to contribute
### Setup
* We recommend you install `movement` inside a [conda](https://docs.conda.io/en/latest/) environment.
Assuming you have `conda` installed, the following will create and activate an environment containing Python 3 as well as the required `pytables` library. You can call your environment whatever you like, we've used `movement-env`.

  ```sh
  conda create -n movement-env -c conda-forge python=3.11 pytables
  conda activate movement-env
  ```

* Next clone the repository and install the package in editable mode (including all `dev` dependencies):

  ```bash
  git clone https://github.com/neuroinformatics-unit/movement
  cd movement
  pip install -e '.[dev]'
  ```
* Initialize the pre-commit hooks:

  ```bash
  pre-commit install
  ```

### Workflow
* Create a new branch, make your changes, and stage them.
* When you try to commit, the pre-commit hooks will be triggered. These include linting with [`ruff`](https://github.com/charliermarsh/ruff) and auto-formatting with [`black`](https://github.com/psf/black). Stage any changes made by the hooks, and commit. You may also run the pre-commit hooks manually, at any time, with `pre-commit run --all-files`.
* Push your changes to GitHub and open a draft pull request.
* If all checks (e.g. linting, type checking, testing) run successfully, you may mark the pull request as ready for review.
* For debugging purposes, you may also want to run the tests and the type checks locally, before pushing. This can be done with the following commands:
    ```bash
    cd movement
    pytest
    mypy -p movement
    ```
* When your pull request is approved, squash-merge it into the `main` branch and delete the feature branch.

### Versioning and deployment
The package is deployed to PyPI automatically when a new release is created on GitHub. We use [semantic versioning](https://semver.org/), with `MAJOR`.`MINOR`.`PATCH` version numbers.

We use [`setuptools_scm`](https://github.com/pypa/setuptools_scm), which automatically [infers the version using git](https://github.com/pypa/setuptools_scm#default-versioning-scheme). To manually set a new semantic version, create an appropriate tag and push it to GitHub. Make sure to commit any changes you wish to be included in this version. E.g. to bump the version to `1.0.0`:

```bash
git add .
git commit -m "Add new changes"
git tag -a v1.0.0 -m "Bump to version 1.0.0"
git push --follow-tags
```

## License

⚖️ [BSD 3-Clause](./LICENSE)

## Template
This package layout and configuration (including pre-commit hooks and GitHub actions) have been copied from the [python-cookiecutter](https://github.com/SainsburyWellcomeCentre/python-cookiecutter) template.
