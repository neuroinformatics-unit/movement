[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
![CI](https://img.shields.io/github/actions/workflow/status/neuroinformatics-unit/movement/test_and_deploy.yml?label=CI)
[![codecov](https://codecov.io/gh/neuroinformatics-unit/movement/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/neuroinformatics-unit/movement)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# movement

Kinematic analysis of animal 🐝 🦀 🐀 🐒 body movements for neuroscience and ethology research 🔬.

- Read the [documentation](https://neuroinformatics-unit.github.io/movement/) for more information.
- If you wish to contribute, please read the [contributing guide](./CONTRIBUTING.md).

## Status
> **Warning**
> - 🏗️ The package is currently in early development. Stay tuned ⌛
> - It is not sufficiently tested to be used for scientific analysis
> - The interface is subject to changes. [Open an issue](https://github.com/neuroinformatics-unit/movement/issues) if you have suggestions.

## Aims
* Load pose tracks from pose estimation software packages (e.g. [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) or [SLEAP](https://sleap.ai/))
* Evaluate the quality of the tracks and perform data cleaning operations
* Calculate kinematic variables (e.g. speed, acceleration, joint angles, etc.)
* Produce reports and visualise the results

## Related projects
The following projects cover related needs and served as inspiration for this project:
* [DLC2Kinematics](https://github.com/AdaptiveMotorControlLab/DLC2Kinematics)
* [PyRat](https://github.com/pyratlib/pyrat)
* [Kino](https://github.com/BrancoLab/Kino)
* [WAZP](https://github.com/SainsburyWellcomeCentre/WAZP)

## License
⚖️ [BSD 3-Clause](./LICENSE)

## Template
This package layout and configuration (including pre-commit hooks and GitHub actions) have been copied from the [python-cookiecutter](https://github.com/neuroinformatics-unit/python-cookiecutter) template.
