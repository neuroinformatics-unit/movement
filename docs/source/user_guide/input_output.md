(target-io)=
# Input/Output

(target-formats)=
## Supported formats
(target-supported-formats)=
`movement` supports the analysis of trajectories of keypoints (_pose tracks_) and of bounding box centroids (_bounding box tracks_).

To analyse pose tracks, `movement` supports loading data from various frameworks:
- [DeepLabCut](dlc:) (DLC)
- [SLEAP](sleap:) (SLEAP)
- [LightingPose](lp:) (LP)
- [Anipose](anipose:) (Anipose)

Additionally, `movement` supports loading data stored in [Neurodata Without Borders (NWB)](https://nwb-overview.readthedocs.io/en/latest/) format (using the [``ndx-pose``](https://github.com/rly/ndx-pose) extension).

To analyse bounding box tracks, `movement` currently supports the [VGG Image Annotator](via:) (VIA) format for [tracks annotation](via:docs/face_track_annotation.html).

:::{note}
At the moment `movement` only deals with tracked data: either keypoints or bounding boxes whose identities are known from one frame to the next, for a consecutive set of frames. For the pose estimation case, this means it only deals with the predictions output by the software packages above. It currently does not support loading manually labelled data (since this is most often defined over a non-continuous set of frames).
:::

Below we explain how you can load pose tracks and bounding box tracks into `movement`, and how you can export a `movement` poses dataset to different file formats. You can also try `movement` out on some [sample data](target-sample-data)
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

:::{tab-item} Anipose
To load Anipose files in .csv format:
```python
ds = load_poses.from_anipose_file(
    "/path/to/file.analysis.csv", fps=30, individual_name="individual_0"
)  # Optionally specify the individual name; defaults to "individual_0"

# or equivalently
ds = load_poses.from_file(
    "/path/to/file.analysis.csv",
    source_software="Anipose",
    fps=30,
    individual_name="individual_0",
)
```
:::

:::{tab-item} NWB
To load NWB files in .nwb format:
```python
ds = load_poses.from_nwb_file(
    "path/to/file.nwb",
    processing_module_key="behavior",
    pose_estimation_key="PoseEstimation",
)  # Optionally specify the name of the ProcessingModule and PoseEstimation objects.
# Defaults are "behavior" and "PoseEstimation", respectively.

# or equivalently
ds = load_poses.from_file(
    "path/to/file.nwb",
    source_software="NWB",
    processing_module_key="behavior",
    pose_estimation_key="PoseEstimation",
)
```
The above functions also accept an {class}`NWBFile<pynwb.file.NWBFile>` object as input:
```python
with pynwb.NWBHDF5IO("path/to/file.nwb", mode="r") as io:
    nwb_file = io.read()
    ds = load_poses.from_nwb_file(
        nwb_file, pose_estimation_key="PoseEstimation"
    )
```
:::

:::{tab-item} From NumPy
In the example below, we create random position data for two individuals, ``Alice`` and ``Bob``,
with three keypoints each: ``snout``, ``centre``, and ``tail_base``. These keypoints are tracked in 2D space for 100 frames, at 30 fps. The confidence scores are set to 1 for all points.
```python
import numpy as np

rng = np.random.default_rng(seed=42)
ds = load_poses.from_numpy(
    position_array=rng.random((100, 2, 3, 2)),
    confidence_array=np.ones((100, 3, 2)),
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
## Loading bounding box tracks
To load bounding box tracks into a [movement bounding boxes dataset](target-poses-and-bboxes-dataset), we need the functions from the
{mod}`movement.io.load_bboxes` module. This module can be imported as:

```python
from movement.io import load_bboxes
```

We currently support loading bounding box tracks in the VGG Image Annotator (VIA) format only. However, like in the poses datasets, we additionally provide a `from_numpy()` method, with which we can build a [movement bounding boxes dataset](target-poses-and-bboxes-dataset) from a set of NumPy arrays.

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
Note that the x,y coordinates in the input VIA tracks .csv file represent the the top-left corner of each bounding box. Instead the corresponding ``movement`` dataset `ds` will hold in its `position` array the centroid of each bounding box.
:::

:::{tab-item} From NumPy
In the example below, we create random position data for two bounding boxes, ``id_0`` and ``id_1``,
both with the same width (40 pixels) and height (30 pixels). These are tracked in 2D space for 100 frames, which will be numbered in the resulting dataset from 0 to 99. The confidence score for all bounding boxes is set to 0.5.
```python
import numpy as np

rng = np.random.default_rng(seed=42)
ds = load_bboxes.from_numpy(
    position_array=rng.random((100, 2, 2)),
    shape_array=np.ones((100, 2, 2)) * [40, 30],
    confidence_array=np.ones((100, 2)) * 0.5,
    individual_names=["id_0", "id_1"]
)
```
:::

::::

The resulting data structure `ds` will include the centroid trajectories for each tracked bounding box, the boxes' widths and heights, and their associated confidence values if provided.

For more information on the bounding boxes data structure, see the [movement dataset](target-poses-and-bboxes-dataset) page.


(target-saving-pose-tracks)=
## Saving pose tracks
[movement poses datasets](target-poses-and-bboxes-dataset) can be saved in a variety of
formats:
- DeepLabCut-style files (.h5 or .csv)
- [SLEAP-style analysis files](sleap:tutorials/analysis) (.h5)
- NWB files (.nwb)

To export pose tracks from `movement`, first import the {mod}`movement.io.save_poses` module:

```python
from movement.io import save_poses
```

Then, depending on the desired format, use one of the following functions:

:::::{tab-set}

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

::::{tab-item} NWB
To convert a `movement` poses dataset to {class}`NWBFile<pynwb.file.NWBFile>` objects:
```python
nwb_files = save_poses.to_nwb_file(ds)
```
The {func}`movement.io.save_poses.to_nwb_file` function also accepts
a {class}`movement.io.nwb.NWBFileSaveConfig` object as its ``config`` argument
for customising metadata such as session or subject information in the resulting `NWBFile`s
(see {func}`the API reference<movement.io.save_poses.to_nwb_file>` for examples).

These `NWBFile`s can then be saved to disk as .nwb files using {class}`pynwb.NWBHDF5IO`:
```python
from pynwb import NWBHDF5IO

for file in nwb_files:
    with NWBHDF5IO(f"{file.identifier}.nwb", "w") as io:
        io.write(file)
```
:::{note}
To allow adding additional data to NWB files before saving, {func}`to_nwb_file<movement.io.save_poses.to_nwb_file>` does not write to disk directly.
Instead, it returns a list of {class}`NWBFile<pynwb.file.NWBFile>` objects---one per individual in the dataset---since NWB files are designed to represent data from a single individual.
:::
::::

:::::


(target-saving-bboxes-tracks)=
## Saving bounding box tracks

We currently do not provide explicit methods to export a movement bounding boxes dataset in a specific format. However, you can easily save the bounding box tracks to a .csv file using the standard Python library `csv`.

Here is an example of how you can save a bounding boxes dataset to a .csv file:

```python
# define name for output csv file
filepath = "tracking_output.csv"

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
Alternatively, we can convert the `movement` bounding boxes dataset to a pandas DataFrame with the {meth}`xarray.DataArray.to_dataframe` method, wrangle the dataframe as required, and then apply the {meth}`pandas.DataFrame.to_csv` method to save the data as a .csv file.


(target-sample-data)=
## Sample data

`movement` includes some sample data files that you can use to
try the package out. These files contain pose and bounding box tracks from
various [supported formats](target-supported-formats).

You can list the available sample data files using:

```python
from movement import sample_data

file_names = sample_data.list_datasets()
print(*file_names, sep='\n')  # print each sample file in a separate line
```

Each sample file is prefixed with the name of the software package
that was used to generate it.

To load one of the sample files as a
[movement dataset](target-poses-and-bboxes-dataset), use the
{func}`movement.sample_data.fetch_dataset()` function:

```python
filename = "SLEAP_three-mice_Aeon_proofread.analysis.h5"
ds = sample_data.fetch_dataset(filename)
```
Some sample datasets also have an associated video file
(the video for which the data was predicted). You can request
to download the sample video by setting `with_video=True`:

```python
ds = sample_data.fetch_dataset(filename, with_video=True)
```

If available, the video file is downloaded and its path is stored
in the `video_path` attribute of the dataset (i.e., `ds.video_path`).
This attribute will not be set if no video is
available for this dataset, or if you did not request it.

Some datasets also include a sample frame file, which is a single
still frame extracted from the video. This can be useful for visualisation
(e.g., as a background image for plotting trajectories). If available,
this file is always downloaded when fetching the dataset,
and its path is stored in the `frame_path` attribute
(i.e., `ds.frame_path`). If no frame file is available for the dataset,
the `frame_path` attribute will not be set.

:::{dropdown} Under the hood
:color: info
:icon: info
When you import the `sample_data` module with `from movement import sample_data`,
`movement` downloads a small metadata file to your local machine with information about the latest sample datasets available. Then, the first time you call the `fetch_dataset()` function, `movement` downloads the requested file to your machine and caches it in the `~/.movement/data` directory. On subsequent calls, the data are directly loaded from this local cache.
:::
