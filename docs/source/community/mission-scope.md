(target-mission)=
# Mission & Scope

## Mission

`movement` aims to **facilitate the study of animal behaviour**
by providing a suite of **Python tools to analyse body movements**
across space and time.

## Scope

At its core, `movement` handles the positions of one or more individuals
tracked over time. An individual's position at a given time can be represented
in various ways: a single keypoint (usually the centroid), a set of keypoints
(also known as the pose), a bounding box, or a segmentation mask.
The spatial coordinates of these representations may be defined in 2D (x, y)
or 3D (x, y, z). The pose and mask representations also carry some information
about the individual's posture.

Animal tracking frameworks such as [DeepLabCut](dlc:) or [SLEAP](sleap:) can
generate these representations from video data by detecting body parts and
tracking them across frames. In the context of `movement`, we refer to the
resulting tracks according to their respective representations—for
example, pose tracks, bounding boxes' tracks, or motion tracks in general.

Our vision is to present a **consistent interface for motion tracks** paired
with **modular and accessible analysis tools**. We aim to accommodate data
from a range of animal tracking frameworks, in **2D or 3D**, tracking
**single or multiple individuals**. As such, `movement` can be considered as
downstream of tools like DeepLabCut and SLEAP. The focus is on providing
functionalities for data cleaning, visualization, and motion quantification
(see the [Roadmap](target-roadmaps) for details).

In the study of animal behavior, motion tracks are often used to extract and
label discrete actions, sometimes referred to as behavioral syllables or
states. While `movement` is not designed for such tasks, it may generate
features useful for action segmentation and recognition. We may develop
packages specialized for this purpose, which will be compatible with
`movement` and the existing ecosystem of related tools.

## Design principles

`movement` is committed to:
- __Ease of installation and use__. We aim for a cross-platform installation and are mindful of dependencies that may compromise this goal.
- __User accessibility__, catering to varying coding expertise by offering both a GUI and a Python API.
- __Comprehensive documentation__, enriched with tutorials and examples.
- __Robustness and maintainability__ through high test coverage.
- __Scientific accuracy and reproducibility__ by validating inputs and outputs.
- __Performance and responsiveness__, especially for large datasets, using parallel processing where appropriate.
- __Modularity and flexibility__. We envision `movement` as a platform for new tools and analyses, offering users the building blocks to craft their own workflows.

Some of these principles are shared with, and were inspired by, napari's [Mission and Values](napari:community/mission_and_values) statement.
