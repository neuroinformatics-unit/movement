(target-movement)=
# movement

A Python toolbox for analysing body movements across space and time, to aid the study of animal behaviour in neuroscience.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fas}`rocket;sd-text-primary` Getting Started
:link: getting_started/index
:link-type: doc

Installation, first steps and key concepts.
:::

:::{grid-item-card} {fas}`chalkboard-user;sd-text-primary` Examples
:link: examples/index
:link-type: doc

A gallery of examples using movement.
:::

:::{grid-item-card} {fas}`comments;sd-text-primary` Join the movement
:link: community/index
:link-type: doc

How to get in touch and contribute.
:::
::::

![](_static/movement_overview.png)

## Overview

Pose estimation tools, such as [DeepLabCut](dlc:) and [SLEAP](sleap:) are now commonplace when processing video data of animal behaviour. There is not yet a standardised, easy-to-use way to process the *pose tracks* produced from these software packages.

movement aims to provide a consistent modular interface to analyse pose tracks, allowing steps such as data cleaning, visualisation and motion quantification.
We aim to support a range of pose estimation packages, along with 2D or 3D tracking of single or multiple individuals.

Find out more on our [mission and scope](target-mission) statement and our [roadmap](target-roadmaps).

```{include} /snippets/status-warning.md
```

```{toctree}
:maxdepth: 2
:hidden:

getting_started/index
examples/index
community/index
api_index
```
