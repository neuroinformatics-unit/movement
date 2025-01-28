(target-napari-plugin)=
# The napari plugin

To enable the interactive visualisation of motion tracks in
`movement`, we have created an experimental plugin for [napari](napari:).
Currently, the plugin supports loading 2D
[pose datasets](target-poses-and-bboxes-dataset), and visualising them
as points overlaid on video frames.

:::{warning}
This plugin is still in the early stages of development and offers
limited functionality. We are working on ironing out the kinks and
gradually adding more features. [Get in touch](target-get-in-touch)
if you find any bugs or have suggestions for improvements!
:::

## Installation

The `napari` plugin is shipped with the `movement` package starting from
version `0.1.0`. If you install `movement` via `conda`, `napari` is already
included as a dependency. If you use the `pip` installer, make sure to
install `movement` with the `[napari]` extra:

```sh
pip install movement[napari]
```

## Usage

Type the following command in your terminal:

```sh
napari -w movement
```

This will open the `napari` window with the `movement` plugin docked on the
right-hand side.
