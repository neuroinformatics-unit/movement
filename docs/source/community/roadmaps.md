(target-roadmaps)=
# Roadmaps

The roadmap outlines **current development priorities** and aims to **guide core developers** and to **encourage community contributions**. It is a living document and will be updated as the project evolves.

The roadmap is **not meant to limit** movement features, as we are open to suggestions and contributions. Join our [Zulip chat](movement-zulip:) to share your ideas. We will take community demand and feedback into account when planning future releases.

## Long-term vision
The following features are being considered for the first stable version `v1.0`.

- __Import/Export pose tracks from/to diverse formats__. We aim to interoperate with leading tools for animal pose estimation and behaviour classification, and to enable conversions between their formats.
- __Standardise the representation of pose tracks__. We represent pose tracks as [xarray data structures](xarray:user-guide/data-structures.html) to allow for labelled dimensions and performant processing.
- __Interactively visualise pose tracks__. We are considering [napari](napari:) as a visualisation and GUI framework.
- __Clean pose tracks__, including, but not limited to, handling of missing values, filtering, smoothing, and resampling.
- __Derive kinematic variables__ like velocity, acceleration, joint angles, etc., focusing on those prevalent in neuroscience.
- __Integrate spatial data about the animal's environment__ for combined analysis with pose tracks. This covers regions of interest (ROIs) such as the arena in which the animal is moving and the location of objects within it.
- __Define and transform coordinate systems__. Coordinates can be relative to the camera, environment, or the animal itself (egocentric).

## Short-term milestone - `v0.1`
We plan to release version `v0.1` of movement in early 2024, providing a minimal set of features to demonstrate the project's potential and to gather feedback from users. At minimum, it should include:

- [x] Ability to import pose tracks from [DeepLabCut](dlc:), [SLEAP](sleap:) and [LightningPose](lp:) into a common `xarray.Dataset` structure.
- [x] At least one function for cleaning the pose tracks.
- [x] Ability to compute velocity and acceleration from pose tracks.
- [x] Public website with [documentation](target-movement).
- [x] Package released on [PyPI](https://pypi.org/project/movement/).
- [ ] Package released on [conda-forge](https://conda-forge.org/).
- [ ] Ability to visualise pose tracks using [napari](napari:). We aim to represent pose tracks via napari's [Points](napari:howtos/layers/points) and [Tracks](napari:howtos/layers/tracks) layers and overlay them on video frames.
