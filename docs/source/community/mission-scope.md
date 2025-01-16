(target-mission)=
# Mission & Scope

## Mission

`movement` aims to **facilitate the study of animal behaviour**
by providing a suite of **Python tools to analyse body movements**
across space and time.

## Scope

At its core, `movement` handles the position and/or orientation
of one or more individuals over time.

There are a few common ways of representing animal motion from video
recordings: an animal's position could be reduced to that of a single keypoint
tracked on its body (usually the centroid), or instead a set of keypoints
(often referred to as the pose) to better capture its orientation as well as
the positions of limbs and appendages. The animal's position could be also
tracked as a bounding box drawn around each individual, or as a segmentation
mask that indicates the pixels belonging to each individual. Depending on the
research question or the application, one or other format may be more
convenient. The spatial coordinates of these representations may be defined
in 2D (x, y) or 3D (x, y, z).

Animal tracking frameworks such as [DeepLabCut](dlc:) or [SLEAP](sleap:) can
generate keypoint representations from video data by detecting body parts and
tracking them across frames. In the context of `movement`, we refer to these
trajectories as _tracks_: we use _pose tracks_ to refer to the trajectories
of a set of keypoints, _bounding boxes' tracks_ to refer to the trajectories
of bounding boxes' centroids, or _motion tracks_ in the more general case.

Our vision is to present a **consistent interface for representing motion
tracks** along with **modular and accessible analysis tools**. We aim to
support data from a range of animal tracking frameworks, in **2D or 3D**,
tracking **single or multiple individuals**. As such, `movement` can be
considered as operating downstream of tools like DeepLabCut and SLEAP.
The focus is on providing functionalities for data cleaning, visualisation,
and motion quantification (see the [Roadmap](target-roadmaps) for details).

In the study of animal behaviour, motion tracks are often used to extract and
label discrete actions, sometimes referred to as behavioural syllables or
states. While `movement` is not designed for such tasks, it can be used to
generate features that are relevant for action recognition.

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
