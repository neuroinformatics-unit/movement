---
blogpost: true
date: FIXME
author: Will Graham
location: London, England
category: release
language: English
---

# Spatial Navigation Feature Update

_This is a short summary of the spatial navigation features introduced during Jan-Feb 2025._

## Original Roadmap

Between January and February 2025, the `movement` developer team decided to focus on expanding the suite of tools useful for the analysis of spatial navigation.
This work started with the creation of so-called "backend" functions that can perform general operations on vectors in an efficient manner, from which friendlier functions could be provided.
Once these features were in place, development would start on supporting "Regions of Interest" (RoIs) within `movement`.
RoIs are essentially labelled regions in space that have some significance in the experimental setup - the location of a nest, or the extent of the enclosure, for example.
By providing a way to store and describe these objects within `movement`, analysis workflows can be simplified and can make use of the existing "backend" functions to provide convenient access to interesting quantities; such as the distance of each individual from a RoI, or the relative orientation of an individual from the nearest wall of the enclosure.

There were also a few independent bursts of development in other areas; including providing sample pupilometry data and an example workflow showing how `movement` can be used to analyse it, making it easier to produce common plots that appear in analysis, and the ability to scale data.

A copy of the original roadmap that was [shared on Zulip](https://neuroinformatics.zulipchat.com/#narrow/channel/406001-Movement/topic/Roadmap.3A.20Spatial.20Navigation/near/495022291) is provided below.

![Original feature roadmap for Jan-Feb.](../_static/blog_posts/arc-roadmap.png)

## What's Been Introduced?

Pupilometry and scaling... either here or maybe as one section?

### Plotting Made Easier

The `movement.plots` submodule has been created, which provides some helpful wrapper functions for producing some of the more common plot types that come out of the analysis of `movement` datasets.
These plots can be added to existing figure axes you have created, and you can pass them the same formatting arguments as you would to the appropriate `matplotlib.pyplot`.
Currently, the submodule has two wrappers to use;

- `plot_trajectory`, which creates a plot of the trajectory of a given keypoint (or the centroid of a collection of keypoints) with a single line of code. Introduced in [#394](https://github.com/neuroinformatics-unit/movement/pull/394).
- `plot_occupancy`, which creates an occupancy plot of an individual, given its position data. Collections of individuals are aggregated over, and if multiple keypoints are provided, the occupancy of the centroid is calculated. Introduced in [#403](https://github.com/neuroinformatics-unit/movement/pull/403).

Examples to showcase the use of these plotting functions are currently [being produced](https://github.com/neuroinformatics-unit/movement/issues/415).
[Our other examples](https://movement.neuroinformatics.dev/examples/index.html) have also been updated to use these functions where possible.

### make broadcastable

### regions of interest

<!-- :::{tip}
See our [installation guide](target-installation) for instructions on how to
install the latest version or upgrade from an existing installation.
:::

__Input/Output__

- We have added the {func}`movement.io.load_poses.from_multiview_files` function to support loading pose tracking data from multiple camera views.
- We have made several small improvements to reading bounding box tracks. See our new {ref}`example <sphx_glr_examples_load_and_upsample_bboxes.py>` to learn more about working with bounding boxes.
- We have added a new {ref}`example <sphx_glr_examples_convert_file_formats.py>` on using `movement` to convert pose tracking data between different file formats.

__Kinematics__

The {mod}`kinematics <movement.kinematics>` module has been moved from `movement.analysis.kinematics` to `movement.kinematics` and packs a number of new functions:
- {func}`compute_forward_vector <movement.kinematics.compute_forward_vector>`
- {func}`compute_head_direction_vector <movement.kinematics.compute_head_direction_vector>`
- {func}`compute_pairwise_distances <movement.kinematics.compute_pairwise_distances>`
- {func}`compute_speed <movement.kinematics.compute_speed>`
- {func}`compute_path_length <movement.kinematics.compute_path_length>`

__Breaking changes__

- We have dropped support for using filtering and
kinematic functions via the `move` accessor syntax,
because we've found the concept hard to convey to new users. All functions are henceforth solely accessible by importing them from the relevant modules. Having one way of doing things simplifies the mental model for users and reduces the maintenance effort on our side. See an example below:

  ```python
  # Instead of:
  position_filt = ds.move.median_filter(window=5)
  velocity = ds.move.compute_velocity()

  # Use:
  from movement.filtering import median_filter
  from movement.kinematics import compute_velocity

  position_filt = median_filter(ds.position, window=5)
  velocity = compute_velocity(ds.position)
  ```
- We have slightly modified the [structure of movement datasets](target-poses-and-bboxes-dataset), by changing the order of dimensions. This should have no effect when indexing data by dimension names, i.e. using the {meth}`xarray.Dataset.sel` or {meth}`xarray.Dataset.isel` methods. However, you may need to update your code if you are using Numpy-style indexing, for example:

  ```python
  # Indexing with dimension names (recommended, works always)
  position = ds.position.isel(
      individuals=0, keypoints=-1     # first individual, last keypoint
  )

  # Numpy-style indexing with the old dimension order (will no longer work)
  position = ds.position[:, 0, -1, :]  # time, individuals, keypoints, space

  # Numpy-style indexing with the updated dimension order (use this instead)
  position = ds.position[:, :, -1, 0]  # time, space, keypoints, individuals
  ```

## Looking to v0.1 and beyond

Over the last 1.5 years, we have gradually built up the core functionalities we envisioned for `movement` version `v0.1`,
as described in our [roadmap](target-roadmaps).
These have included [input/output support](target-io) for a few popular animal tracking frameworks, as well as methods for data cleaning and computing kinematic variables.

What we're still missing is a [napari](napari:) plugin for `movement`, which we envision both as an interactive visualisation framework for motion tracking data as well as a graphical user interface for `movement`.
We have been working on a minimal version of this plugin for a while and are expecting to ship it as part of the `v0.1` release in early 2025.

After `v0.1`, we'll be switching to [semantic versioning](https://semver.org/), as it applies to MINOR (new features) and PATCH (bug fixes) versions. Until we are ready for a `v1` MAJOR version, we cannot commit to backward compatibility, but any breaking changes will be clearly communicated in the release notes.

## Announcing movement Community Calls

We are committed to fostering openness, transparency, and a strong sense of
community within the `movement` project.
Starting next year, we will host regular Community Calls via Zoom.

The calls will take place every second Friday from **11:00 to 11:45 GMT**,
beginning on **10 January 2025**.
These calls are open to anyone interested in contributing to `movement` or
sharing feedback on the project's progress and direction.

A few days before each call, we will post an announcement on Zulip with the Zoom link and agenda.
We encourage everyone who's interested in
joining to follow this [Zulip topic](movement-community-calls:)
to stay updated. -->
