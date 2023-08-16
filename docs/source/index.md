# movement

Kinematic analysis of animal 🐝 🦀 🐀 🐒 body movements for neuroscience and ethology research.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fas}`rocket;sd-text-primary` Getting Started
:link: getting_started
:link-type: doc

Install and try it out.
:::

:::{grid-item-card} {fas}`chalkboard-user;sd-text-primary` Examples
:link: auto_examples/index
:link-type: doc

Example use cases.
:::

:::{grid-item-card} {fas}`code;sd-text-primary` API Reference
:link: api_index
:link-type: doc

Index of all functions, classes, and methods.
:::
::::

## Status
:::{warning}
- 🏗️ The package is currently in early development. Stay tuned ⌛
- It is not sufficiently tested to be used for scientific analysis
- The interface is subject to changes. [Open an issue](https://github.com/neuroinformatics-unit/movement/issues) if you have suggestions.
:::


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


```{toctree}
:maxdepth: 2
:hidden:

getting_started
auto_examples/index
api_index
```
