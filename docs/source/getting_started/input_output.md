(target-io)=
# Input/Output

(target-formats)=
## Supported formats
(target-supported-formats)=
`movement` supports the analysis of trajectories of keypoints (_pose tracks_) and of bounding boxes' centroids (_bounding boxes' tracks_).

To analyse pose tracks, `movement` supports loading data from various pose estimation frameworks.
Currently, these include:
- [DeepLabCut](dlc:) (DLC)
- [SLEAP](sleap:) (SLEAP)
- [LightingPose](lp:) (LP)

To analyse bounding boxes' tracks, `movement` currently supports the [VGG Image Annotator](via:) (VIA) format for [tracks annotation](via:docs/face_track_annotation.html#:~:text=A%20snippet%20of%20this%20csv%20file%20is%20shown%20below)

:::{note}
At the moment `movement` only deals with tracked data: either keypoints or bounding boxes whose identities are known from one frame to the next, for a consecutive set of frames. For the pose estimation case, this means it only deals with the predictions output by the software packages mentioned above. It currently does not support loading manually labelled data, since this is most often defined over a non-continuous set of frames.
:::

(target-loading-pose-tracks)=
## Loading pose tracks

The pose track loading functionalities are provided by the
{mod}`movement.io.load_poses` module, which can be imported as follows:

```python
from movement.io import load_poses
```

Depending on the source software, one of the following functions can be used.

::::{tab-set}

:::{tab-item} SLEAP

To load [SLEAP analysis files](sleap:tutorials/analysis) in .h5 format (recommended):
```python
ds = load_poses.from_sleap_file("/path/to/file.analysis.h5", fps=30)

# or equivalently
ds = load_poses.from_file(
    "/path/to/file.analysis.h5", source_software="SLEAP", fps=30
)
```
To load [SLEAP analysis files](sleap:tutorials/analysis) in .slp format (experimental, see notes in {func}`movement.io.load_poses.from_sleap_file`):

```python
ds = load_poses.from_sleap_file("/path/to/file.predictions.slp", fps=30)
```
:::

:::{tab-item} DeepLabCut

To load DeepLabCut files in .h5 format:
```python
ds = load_poses.from_dlc_file("/path/to/file.h5", fps=30)

# or equivalently
ds = load_poses.from_file(
    "/path/to/file.h5", source_software="DeepLabCut", fps=30
)
```

To load DeepLabCut files in .csv format:
```python
ds = load_poses.from_dlc_file("/path/to/file.csv", fps=30)
```
:::

:::{tab-item} LightningPose

To load LightningPose files in .csv format:
```python
ds = load_poses.from_lp_file("/path/to/file.analysis.csv", fps=30)

# or equivalently
ds = load_poses.from_file(
    "/path/to/file.analysis.csv", source_software="LightningPose", fps=30
)
```
:::

::::

The loaded data include the predicted trajectories for each individual and
keypoint, as well as the associated point-wise confidence values reported by
the pose estimation software.

For more information on data structure, see the [movement dataset](target-dataset) page.

You can also try `movement` out on some [sample data](target-sample-data)
included with the package.

(target-saving-pose-tracks)=
## Saving pose tracks
[movement datasets](target-dataset) can be saved in a variety of
formats, including DeepLabCut-style files (.h5 or .csv) and
[SLEAP-style analysis files](sleap:tutorials/analysis) (.h5).

To export pose tracks from `movement`, first import the {mod}`movement.io.save_poses` module:

```python
from movement.io import save_poses
```

Then, depending on the desired format, use one of the following functions:

:::::{tab-set}

::::{tab-item} SLEAP

To save as a SLEAP analysis file in .h5 format:
```python
save_poses.to_sleap_analysis_file(ds, "/path/to/file.h5")
```

:::{note}
When saving to SLEAP-style files, only `track_names`, `node_names`, `tracks`, `track_occupancy`,
and `point_scores` are saved. `labels_path` will only be saved if the source
file of the dataset is a SLEAP .slp file. Otherwise, it will be an empty string.
Other attributes and data variables
(i.e., `instance_scores`, `tracking_scores`, `edge_names`, `edge_inds`, `video_path`,
`video_ind`, and `provenance`) are not currently supported. To learn more about what
each attribute and data variable represents, see the
[SLEAP documentation](sleap:api/sleap.info.write_tracking_h5.html#module-sleap.info.write_tracking_h5).
:::
::::

::::{tab-item} DeepLabCut

To save as a DeepLabCut file, in .h5 or .csv format:
```python
save_poses.to_dlc_file(ds, "/path/to/file.h5")  # preferred format
save_poses.to_dlc_file(ds, "/path/to/file.csv")
```

The {func}`movement.io.save_poses.to_dlc_file` function also accepts
a `split_individuals` boolean argument. If set to `True`, the function will
save the data as separate single-animal DeepLabCut-style files.

::::

::::{tab-item} LightningPose

To save as a LightningPose file in .csv format:
```python
save_poses.to_lp_file(ds, "/path/to/file.csv")
```
:::{note}
Because LightningPose follows the single-animal
DeepLabCut .csv format, the above command is equivalent to:
```python
save_poses.to_dlc_file(ds, "/path/to/file.csv", split_individuals=True)
```
:::

::::
:::::
