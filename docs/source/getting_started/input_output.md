(target-io)=
# Input/Output

(target-formats)=
## Supported formats
(target-supported-formats)=
`movement` supports the analysis of trajectories of keypoints (_pose tracks_) and of bounding boxes' centroids (_bounding boxes' tracks_).

To analyse pose tracks, `movement` supports loading data from various frameworks:
- [DeepLabCut](dlc:) (DLC)
- [SLEAP](sleap:) (SLEAP)
- [LightingPose](lp:) (LP)

To analyse bounding boxes' tracks, `movement` currently supports the [VGG Image Annotator](via:) (VIA) format for [tracks annotation](via:docs/face_track_annotation.html).

:::{note}
At the moment `movement` only deals with tracked data: either keypoints or bounding boxes whose identities are known from one frame to the next, for a consecutive set of frames. For the pose estimation case, this means it only deals with the predictions output by the software packages above. It currently does not support loading manually labelled data (since this is most often defined over a non-continuous set of frames).
:::

Below we explain how you can load pose tracks and bounding boxes' tracks into `movement`, and how you can export a `movement` poses dataset to different file formats. You can also try `movement` out on some [sample data](target-sample-data)
included with the package.


(target-loading-pose-tracks)=
## Loading pose tracks

The pose tracks loading functionalities are provided by the
{mod}`movement.io.load_poses` module, which can be imported as follows:

```python
from movement.io import load_poses
```

To read a pose tracks file into a [movement poses dataset](target-poses-and-bboxes-dataset), we provide specific functions for each of the supported formats. We additionally provide a more general `from_numpy()` method, with which we can build a [movement poses dataset](target-poses-and-bboxes-dataset) from a set of NumPy arrays.

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

:::{tab-item} From NumPy

In the example below, we create random position data for two individuals, ``Alice`` and ``Bob``,
with three keypoints each: ``snout``, ``centre``, and ``tail_base``. These keypoints are tracked in 2D space for 100 frames, at 30 fps. The confidence scores are set to 1 for all points.

```python
import numpy as np

ds = load_poses.from_numpy(
    position_array=np.random.rand((100, 2, 3, 2)),
    confidence_array=np.ones((100, 2, 3)),
    individual_names=["Alice", "Bob"],
    keypoint_names=["snout", "centre", "tail_base"],
    fps=30,
)
```
:::

::::

The resulting poses data structure `ds` will include the predicted trajectories for each individual and
keypoint, as well as the associated point-wise confidence values reported by
the pose estimation software.

For more information on the poses data structure, see the [movement poses dataset](target-poses-and-bboxes-dataset) page.


(target-loading-bbox-tracks)=
## Loading bounding boxes' tracks
To load bounding boxes' tracks into a [movement bounding boxes dataset](target-poses-and-bboxes-dataset), we need the functions from the
{mod}`movement.io.load_bboxes` module. This module can be imported as:

```python
from movement.io import load_bboxes
```

We currently support loading bounding boxes' tracks in the VGG Image Annotator (VIA) format only. However, like in the poses datasets, we additionally provide a `from_numpy()` method, with which we can build a [movement bounding boxes dataset](target-poses-and-bboxes-dataset) from a set of NumPy arrays.

::::{tab-set}
:::{tab-item} VGG Image Annotator

To load a VIA tracks .csv file:
```python
ds = load_bboxes.from_via_tracks_file("path/to/file.csv", fps=30)

# or equivalently
ds = load_bboxes.from_file(
    "path/to/file.csv",
    source_software="VIA-tracks",
    fps=30,
)
```
:::

:::{tab-item} From NumPy

In the example below, we create random position data for two bounding boxes, ``id_0`` and ``id_1``,
both with the same width (40 pixels) and height (30 pixels). These are tracked in 2D space for 100 frames, which will be numbered in the resulting dataset from 0 to 99. The confidence score for all bounding boxes is set to 0.5.

```python
import numpy as np

ds = load_bboxes.from_numpy(
    position_array=np.random.rand(100, 2, 2),
    shape_array=np.ones((100, 2, 2)) * [40, 30],
    confidence_array=np.ones((100, 2)) * 0.5,
    individual_names=["id_0", "id_1"]
)
```
:::

::::

The resulting data structure `ds` will include the centroid trajectories for each tracked bounding box, the boxes' widths and heights, and their associated confidence values if provided.

For more information on the bounding boxes data structure, see the [movement bounding boxes dataset](target-poses-and-bboxes-dataset) page.


(target-saving-pose-tracks)=
## Saving pose tracks
[movement poses datasets](target-poses-and-bboxes-dataset) can be saved in a variety of
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


(target-saving-bboxes-tracks)=
## Saving bounding boxes' tracks

We currently do not provide explicit methods to export a movement bounding boxes dataset in a specific format. However, you can easily save the bounding boxes' trajectories to a .csv file using the standard Python library `csv`.

Here is an example of how you can save a bounding boxes dataset to a .csv file:

```python
# define name for output csv file
file = 'tracking_output.csv"

# open the csv file in write mode
with open(filepath, mode="w", newline="") as file:
    writer = csv.writer(file)

    # write the header
    writer.writerow(["frame_idx", "bbox_ID", "x", "y", "width", "height", "confidence"])

    # write the data
    for individual in ds.individuals.data:
        for frame in ds.time.data:
            x, y = ds.position.sel(time=frame, individuals=individual).data
            width, height = ds.shape.sel(time=frame, individuals=individual).data
            confidence = ds.confidence.sel(time=frame, individuals=individual).data
            writer.writerow([frame, individual, x, y, width, height, confidence])

```
Alternatively, we can convert the `movement` bounding boxes' dataset to a pandas DataFrame with the {func}`.xarray.DataArray.to_dataframe()` method, wrangle the dataframe as required, and then apply the {func}`.pandas.DataFrame.to_csv()` method to save the data as a .csv file.
