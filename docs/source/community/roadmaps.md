(target-roadmaps)=
# Roadmaps

This page outlines **current development priorities** and aims to **guide core developers** and to **encourage community contributions**. It is a living document and will be updated as the project evolves.

The roadmaps are **not meant to limit** `movement` features, as we are open to suggestions and contributions. Join our [Zulip chat](movement-zulip:) to share your ideas. We will take community feedback into account when planning future releases.

## Long-term vision
The following features are being considered for the first stable version `v1.0`.

- __Import/Export motion tracks from/to diverse formats__. We aim to interoperate with leading tools for animal tracking and behaviour classification, and to enable conversions between their formats.
- __Standardise the representation of motion tracks__. We represent tracks as [xarray data structures](xarray:user-guide/data-structures.html) to allow for labelled dimensions and performant processing.
- __Interactively visualise motion tracks__. We are experimenting with [napari](napari:) as a visualisation and GUI framework.
- __Clean motion tracks__, including, but not limited to, handling of missing values, filtering, smoothing, and resampling.
- __Derive kinematic variables__ like velocity, acceleration, joint angles, etc., focusing on those prevalent in neuroscience and ethology.
- __Integrate spatial data about the animal's environment__ for combined analysis with motion tracks. This covers regions of interest (ROIs) such as the arena in which the animal is moving and the location of objects within it.
- __Define and transform coordinate systems__. Coordinates can be relative to the camera, environment, or the animal itself (egocentric).
- __Provide common metrics for specialised applications__. These applications could include gait analysis, pupillometry, spatial
navigation, social interactions, etc.
- __Integrate with neurophysiological data analysis tools__. We eventually aim to facilitate combined analysis of motion and neural data.

## Short-term milestone - `v0.1`
We plan to release version `v0.1` of `movement` in early 2025, providing a minimal set of features to demonstrate the project's potential and to gather feedback from users. At minimum, it should include:

- [x] Ability to import pose tracks from [DeepLabCut](dlc:), [SLEAP](sleap:) and [LightningPose](lp:) into a common `xarray.Dataset` structure.
- [x] At least one function for cleaning the pose tracks.
- [x] Ability to compute velocity and acceleration from pose tracks.
- [x] Public website with [documentation](target-movement).
- [x] Package released on [PyPI](https://pypi.org/project/movement/).
- [x] Package released on [conda-forge](https://anaconda.org/conda-forge/movement).
- [ ] Ability to visualise pose tracks using [napari](napari:). We aim to represent pose tracks as napari [layers](napari:howtos/layers/index.html), overlaid on video frames.
