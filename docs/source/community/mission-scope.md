(target-mission)=
# Mission & Scope

## Mission

[movement](target-movement) aims to **facilitate the study of animal behaviour in neuroscience** by providing a suite of **Python tools to analyse body movements** across space and time.

## Scope

At its core, movement handles trajectories of *keypoints*, which are specific body parts of an *individual*. An individual's posture or *pose* is represented by a set of keypoint coordinates, given in 2D (x,y) or 3D (x,y,z). The sequential collection of poses over time forms *pose tracks*. In neuroscience, these tracks are typically extracted from video data using software like [DeepLabCut](dlc:) or [SLEAP](sleap:).

With movement, our vision is to present a **consistent interface for pose tracks** and to **analyze them using modular and accessible tools**. We aim to accommodate data from a range of pose estimation packages, in **2D or 3D**, tracking **single or multiple individuals**. The focus will be on providing functionalities for data cleaning, visualisation and motion quantification (see the [Roadmap](target-roadmaps) for details).

While movement is not designed for behaviour classification or action segmentation, it may extract features useful for these tasks. We are planning to develop separate packages for this purpose, which will be compatible with movement and the existing ecosystem of related tools.

## Design principles

movement is committed to:
- __Ease of installation and use__. We aim for a cross-platform installation and are mindful of dependencies that may compromise this goal.
- __User accessibility__, catering to varying coding expertise by offering both a GUI and a Python API.
- __Comprehensive documentation__, enriched with tutorials and examples.
- __Robustness and maintainability__ through high test coverage.
- __Scientific accuracy and reproducibility__ by validating inputs and outputs.
- __Performance and responsiveness__, especially for large datasets, using parallel processing where appropriate.
- __Modularity and flexibility__. We envision movement as a platform for new tools and analyses, offering users the building blocks to craft their own workflows.

Some of these principles are shared with, and were inspired by, napari's [Mission and Values](napari:community/mission_and_values) statement.
