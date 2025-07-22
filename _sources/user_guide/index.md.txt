# User guide

Start by [installing the package](installation.md) and
[loading your own tracking data](input_output.md), or playing with some
[sample data](target-sample-data) provided with the package.

Before you dive deeper, we highly recommend reading about the structure
and usage of [movement datasets](movement_dataset.md), which are a central
concept in the package.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {fas}`wrench;sd-text-primary` Installation
:link: installation
:link-type: doc

Install the package with `conda` or `pip`.
:::

:::{grid-item-card} {fas}`download;sd-text-primary` Input/Output
:link: input_output
:link-type: doc

Load and save tracking data.
:::

:::{grid-item-card} {fas}`table;sd-text-primary` The movement datasets
:link: movement_dataset
:link-type: doc

Learn about our data structures.
:::

:::{grid-item-card} {fas}`line-chart;sd-text-primary` Graphical User Interface
:link: gui
:link-type: doc

Use our `napari` plugin to view and explore your data interactively.
:::

::::


```{toctree}
:maxdepth: 2
:hidden:

installation
input_output
movement_dataset
gui
```
