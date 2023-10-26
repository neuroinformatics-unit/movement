(target-roadmap)=
# Roadmap

The roadmap outlines **current development priorities** and aims to **guide core developers** and to **encourage community contributions**. It is a living document and will be updated as the project evolves.

The roadmap is **not meant to limit** movement features, as we are open to suggestions and contributions. Join our [Zulip chat](https://neuroinformatics.zulipchat.com/#narrow/stream/406001-Movement/topic/Welcome!) to share your ideas. We will take community demand and feedback into account when planning future releases.

## Long-term vision
The following features are being considered for the first stable version `v1.0`.

- __Import/Export pose tracks from/to diverse formats__. We aim to interoperate with leading tools for animal pose estimation and behaviour classification, and to enable conversions between their formats.
- __Standardise the representation of pose tracks__. We represent pose tracks as [xarray data structures](https://docs.xarray.dev/en/latest/user-guide/data-structures.html) to allow for labelled dimensions and performant processing.
- __Interactively visualise pose tracks__. We are considering [napari](https://napari.org/) as a visualisation and GUI framework.
- __Clean pose tracks__, including, but not limited to, handling of missing values, filtering, smoothing, and resampling.
- __Derive kinematic variables__ like velocity, acceleration, joint angles, etc., focusing on those prevalent in neuroscience.
- __Integrate spatial data about the animal's environment__ for combined analysis with pose tracks. This covers regions of interest (ROIs) such as the arena in which the animal is moving and the location of objects within it.
- __Define and transform coordinate systems__. Coordinates can be relative to the camera, environment, or the animal itself (egocentric).

## Short-term milestone - `v0.1`
We plan to release version `v0.1` of movement in early 2024, providing a minimal set of features to demonstrate the project's potential and to gather feedback from users. At minimum, it should include the following features:

- Importing pose tracks from [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut) and [SLEAP](https://sleap.ai/) into a common `xarray.Dataset` structure. This has been largely accomplished, but some remaining work is required to handle special cases.
- Visualisation of pose tracks using [napari](https://napari.org/). We aim to represent pose tracks via the [napari tracks layer](https://napari.org/stable/howtos/layers/tracks.html) and overlay them on a video frame. This should be accompanied by a minimal GUI widget to allow selection of a subset of the tracks to plot. This line of work is still in a pilot phase. We may decide to use a different visualisation framework if we encounter roadblocks.
- At least one function for cleaning the pose tracks. Once the first one is in place, it can serve as a template for others.
- Computing velocity and acceleration from pose tracks. Again, this should serve as a template for other kinematic variables.
- Package release on PyPI and conda-forge, along with documentation. The package is already available on [PyPI](https://pypi.org/project/movement/) and the [documentation website](https://movement.neuroinformatics.dev/) is up and running. We plan to also release it on conda-forge to enable one-line installation.
