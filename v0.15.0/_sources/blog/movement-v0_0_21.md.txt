---
blogpost: true
date: Dec 5, 2024
author: Niko Sirmpilatze
location: London, England
category: release
language: English
---

# Release v0.0.21 and next steps

_This is our inaugaural blogpost, containing a summary of the `v0.0.21` release and a preview of what's coming next in 2025._

## What's new in movement v0.0.21?

:::{tip}
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
to stay updated.
