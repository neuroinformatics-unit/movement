(target-io)=
# Input/Output

## Overview

Our goal with `movement` is to enable pipelines that are input-agnostic,
meaning they are not tied to a specific motion tracking tool or data format.
Therefore, our input/output functions are designed to facilitate data flows
between various [supported formats](target-supported-formats) and `movement`'s own
[data structure](target-poses-and-bboxes-dataset) based on [xarray](xarray:).

Two functions cover most use cases:

- {func}`~movement.io.load.load_dataset` [loads data](target-load-data) from any supported format into a `movement` dataset.
- {func}`~movement.io.save.save_dataset` [saves a `movement` dataset](target-save-data) to any supported format.

Among the supported formats, `movement`'s native [netCDF](target-netcdf) format preserves all variables and metadata. **This is the recommended way to save your analysis state**, allowing your `movement`-powered workflows to resume exactly where they left off.

You are also welcome to try `movement` by loading some [sample data](target-sample-data) included with the package.

(target-supported-formats)=
## Supported formats

`movement` supports the analysis of trajectories of keypoints (_pose tracks_) and of bounding box centroids (_bounding box tracks_),
which are represented as [movement datasets](target-poses-and-bboxes-dataset)
and can be [loaded from](target-load-data) and [saved to](target-save-data) the following formats.

| Software                                                                    | Abbreviation | Format                                                                                                                    | Dataset Type         | Supported Operations |
| --------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------- | -------------------- | -------------------- |
| `movement`                                                                  |              | [netCDF](target-netcdf) .nc file saved with `movement`                                                                    | Pose or Bounding box | Load & Save          |
| [DeepLabCut](dlc:)                                                          | DLC          | DLC-style .h5 or .csv file, or corresponding pandas DataFrame                                                             | Pose                 | Load & Save          |
| [SLEAP](sleap:)                                                             | SLEAP        | [analysis](sleap-docs:tutorial/exporting-the-results/#analysis-hdf5) .h5 or .slp file                                     | Pose                 | Load & Save\*        |
| [LightningPose](lp:)                                                        | LP           | DLC-style .csv file, or corresponding pandas DataFrame                                                                    | Pose                 | Load & Save          |
| [Anipose](anipose:)                                                         |              | triangulation .csv file, or corresponding pandas DataFrame                                                                | Pose                 | Load                 |
| [COCO](coco:)                                                               | COCO         | COCO keypoint detection [results](coco:format-results) .json file, optionally with a COCO [annotations](coco:format-data) .json file | Pose        | Load                 |
| [VGG Image Annotator](via:)                                                 | VIA          | .csv file for [tracks annotation](via:docs/face_track_annotation.html)                                                    | Bounding box         | Load & Save          |
| [Neurodata Without Borders](https://nwb-overview.readthedocs.io/en/latest/) | NWB          | .nwb file or NWBFile object with the [ndx-pose extension](https://github.com/rly/ndx-pose)                                | Pose                 | Load & Save          |
| Any                                                                         |              | NumPy arrays                                                                                                              | Pose or Bounding box | Load & Save\*\*      |

\*SLEAP files can only be saved in the analysis .h5 format.

\*\*Exporting any `movement` DataArray to a NumPy array is as simple as calling {meth}`xarray.DataArray.to_numpy`, so no specialised "Export/Save As" function is needed, see [xarray's documentation](xarray:user-guide/duckarrays.html) for more details.

:::{tip}
Use {func}`~movement.io.load.get_supported_source_software` and {func}`~movement.io.save.get_supported_target_software` to list the software and file suffixes that `movement` supports **loading from** and **saving to**, respectively.
:::

:::{note}
Currently, `movement` focuses on tracked data, i.e. keypoints or bounding boxes whose identities are known from one frame to the next. Loading manually labelled (training) data is not yet supported.
:::

(target-load-data)=
## Load data

The {func}`~movement.io.load.load_dataset` function provides a unified, format-agnostic way to load poses or bounding boxes from any of the [supported formats](target-supported-formats).
Internally, it dispatches to the appropriate format-specific loading function
(e.g. those in the {mod}`movement.io.load_poses` and {mod}`movement.io.load_bboxes` modules) based on the `source_software` parameter.

By default, `movement` infers `source_software` from the file format via
{func}`~movement.io.load.infer_source_software`:

```python
from movement.io import load_dataset

ds = load_dataset(
    "/path/to/file.h5",
    source_software="auto",  # default
    fps=30,
)
```

This reads the file and returns a [movement dataset](target-poses-and-bboxes-dataset)
containing the tracked trajectories, the associated confidence values and
some metadata attributes. Providing an `fps` (frames per second) value ensures
that time coordinates are in seconds rather than in frame numbers.

When a file format could have plausibly come from multiple software packages
(e.g. DeepLabCut-style `.csv` files, which are also used by LightningPose),
`movement` records the ambiguity in the loaded dataset's attributes
(e.g. `ds.attrs["source_software"] = "DeepLabCut/LightningPose"`).
If you know the source software or want to ensure a specific loader
is used, you can set `source_software` explicitly:

```python
ds = load_dataset(
    "/path/to/file.csv",
    source_software="DeepLabCut",
    fps=30,
)
```

Some formats support additional options, which can be passed as keyword arguments to {func}`~movement.io.load.load_dataset` and are forwarded to the format-specific loader
(see [](target-format-specific)).

:::{note}
Before reading, the loading functions validate the input file to ensure it has the appropriate suffix and structure for the chosen format. For `movement` netCDF files, this means checking that the file contains a {mod}`valid movement dataset<movement.validators.datasets>` with the required dimensions and data variables. See [Working with netCDF](target-netcdf) if you need to bypass this validation.
:::

For more information on the resulting data structure, see the [movement datasets](target-poses-and-bboxes-dataset) page.

(target-load-multiview)=
### Load multiple views

If the same scene was recorded from multiple camera views,
you can load one file per view and combine them into a single dataset
with {func}`~movement.io.load.load_multiview_dataset`.
The resulting dataset has an additional `view` dimension, whose coordinates are the
view names given as keys of the input dictionary:

```python
from movement.io import load_multiview_dataset

ds = load_multiview_dataset(
    {
        "left": "/path/to/left.h5",
        "right": "/path/to/right.h5",
    },
    source_software="DeepLabCut",
    fps=30,
)
```

All views must share identical `time` coordinates.

(target-save-data)=
## Save data

The {func}`~movement.io.save.save_dataset` function provides a unified, format-agnostic way to save poses or bounding boxes to any of the [supported formats](target-supported-formats).
Internally, it dispatches to the appropriate format-specific saving function
(e.g. those in the {mod}`movement.io.save_poses` and {mod}`movement.io.save_bboxes` modules) based on the `target_software` parameter.

By default, the dataset is saved in `movement`'s native [netCDF](target-netcdf) (`.nc`) format (equivalent to explicitly passing `target_software="movement"`):

```python
from movement.io import save_dataset

save_dataset(ds, "/path/to/file.nc")
```

To save to a third-party format instead, set `target_software` explicitly:

```python
# poses
save_dataset(ds, "/path/to/file.h5", target_software="DeepLabCut")

# bounding boxes
save_dataset(ds, "/path/to/file.csv", target_software="VIA-tracks")
```

Some formats support additional options, which can be passed as keyword arguments to {func}`~movement.io.save.save_dataset` and are forwarded to the format-specific writer
(see [](target-format-specific)).

:::{note}
Before writing, the saving functions validate the dataset to ensure it is a {mod}`valid movement dataset<movement.validators.datasets>` with the required dimensions and data variables, and that the file path is writable with the appropriate suffix for the chosen format. See [Working with netCDF](target-netcdf) if you need to bypass this validation.
:::

(target-format-specific)=
## Format-specific options

The dropdowns below show the options available for loading from and saving to each format,
along with the equivalent format-specific loaders and writers.
Some formats also have functions for in-memory data, such as pandas DataFrames or NumPy arrays.

::::{dropdown} movement
:name: target-format-movement

**Load** (.nc)

To load a `movement` dataset saved as a [netCDF](target-netcdf) file:
```python
ds = load_dataset("/path/to/file.nc")
```
The `fps` argument is ignored for netCDF files, as the time coordinates are stored in the file.
Any additional keyword arguments are forwarded to {func}`xarray.load_dataset`.

**Save** (.nc)

To save a `movement` dataset as a [netCDF](target-netcdf) file:
```python
save_dataset(ds, "/path/to/file.nc")
```
Any additional keyword arguments are forwarded to {meth}`xarray.Dataset.to_netcdf`.
::::

::::{dropdown} DeepLabCut
:name: target-format-dlc

**Load** (.h5 or .csv)

To load a DeepLabCut .h5 or .csv file:
```python
ds = load_dataset("/path/to/file.h5", source_software="DeepLabCut", fps=30)
```

This is equivalent to calling {func}`~movement.io.load_poses.from_dlc_file`:
```python
from movement.io import load_poses

ds = load_poses.from_dlc_file("/path/to/file.h5", fps=30)
```

You can also load any pandas DataFrame `df` that's
formatted in the DeepLabCut style:
```python
ds = load_poses.from_dlc_style_df(df, fps=30)
```

:::{note}
In `movement`, pose data can only be loaded if all individuals have the same set of keypoints (i.e., the same labeled body parts). While DeepLabCut supports assigning keypoints that are not shared across individuals (see the [DeepLabCut documentation for multi-animal projects](https://deeplabcut.github.io/DeepLabCut/docs/quick-start/tutorial_maDLC.html)), this feature is not currently supported in `movement`.
:::

**Save** (.h5 or .csv)

To save as a DeepLabCut .h5 (preferred) or .csv file:
```python
save_dataset(ds, "/path/to/file.h5", target_software="DeepLabCut")
```

By default, multi-animal datasets are saved as a single multi-animal file. To save them as separate single-animal DeepLabCut-style files instead, set `split_individuals=True`:
```python
save_dataset(
    ds,
    "/path/to/file.h5",
    target_software="DeepLabCut",
    split_individuals=True,
)
```

This is equivalent to calling {func}`~movement.io.save_poses.to_dlc_file`:
```python
from movement.io import save_poses

save_poses.to_dlc_file(ds, "/path/to/file.h5", split_individuals=True)
```

To convert the dataset to a DeepLabCut-style pandas DataFrame without saving it, use {func}`~movement.io.save_poses.to_dlc_style_df`.
::::

::::{dropdown} SLEAP
:name: target-format-sleap

**Load** (.h5 or .slp)

To load a [SLEAP analysis file](sleap-docs:tutorial/exporting-the-results/#analysis-hdf5) in .h5 format (recommended):
```python
ds = load_dataset("/path/to/file.analysis.h5", source_software="SLEAP", fps=30)
```

To load a SLEAP file in .slp format (experimental, see notes in {func}`~movement.io.load_poses.from_sleap_file`):
```python
ds = load_dataset(
    "/path/to/file.predictions.slp", source_software="SLEAP", fps=30
)
```

This is equivalent to calling {func}`~movement.io.load_poses.from_sleap_file`:
```python
from movement.io import load_poses

ds = load_poses.from_sleap_file("/path/to/file.analysis.h5", fps=30)
```

**Save** (analysis .h5)

To save as a SLEAP analysis .h5 file:
```python
save_dataset(ds, "/path/to/file.h5", target_software="SLEAP")
```

This is equivalent to calling {func}`~movement.io.save_poses.to_sleap_analysis_file`:
```python
from movement.io import save_poses

save_poses.to_sleap_analysis_file(ds, "/path/to/file.h5")
```
When saving to SLEAP-style files, only `track_names`, `node_names`, `tracks`,
`track_occupancy`, and either `point_scores` or `instance_scores` are saved.
Point-wise `confidence` is saved as `point_scores`, while individual-wise
`confidence` is saved as `instance_scores`; the other is filled with NaNs.
`labels_path` will only be saved if the source
file of the dataset is a SLEAP .slp file. Otherwise, it will be an empty string.
Other attributes and data variables
(e.g. `tracking_scores`, `edge_names`, `video_path`) are not currently supported.

To learn more about what each attribute and data variable represents, see the
[SLEAP documentation](sleap-docs:api/info/write_tracking_h5/#sleap.info.write_tracking_h5).
::::

::::{dropdown} LightningPose
:name: target-format-lp

**Load** (.csv)

To load a LightningPose .csv file:
```python
ds = load_dataset("/path/to/file.csv", source_software="LightningPose", fps=30)
```

This is equivalent to calling {func}`~movement.io.load_poses.from_lp_file`:
```python
from movement.io import load_poses

ds = load_poses.from_lp_file("/path/to/file.csv", fps=30)
```

Because LightningPose follows the DeepLabCut dataframe format, you can also load an appropriately formatted pandas DataFrame `df`:
```python
ds = load_poses.from_dlc_style_df(df, fps=30, source_software="LightningPose")
```

**Save** (.csv)

To save as a LightningPose .csv file:
```python
save_dataset(ds, "/path/to/file.csv", target_software="LightningPose")
```

This is equivalent to calling {func}`~movement.io.save_poses.to_lp_file`:
```python
from movement.io import save_poses

save_poses.to_lp_file(ds, "/path/to/file.csv")
```

Because LightningPose follows the single-animal
DeepLabCut .csv format, the above is equivalent to:
```python
save_poses.to_dlc_file(ds, "/path/to/file.csv", split_individuals=True)
```
::::

::::{dropdown} Anipose
:name: target-format-anipose

**Load** (triangulation .csv)

To load an Anipose triangulation file, optionally specifying the name of the tracked individual (defaults to `"id_0"`):
```python
ds = load_dataset(
    "/path/to/file.csv",
    source_software="Anipose",
    fps=30,
    individual_name="id_0",
)
```

This is equivalent to calling {func}`~movement.io.load_poses.from_anipose_file`:
```python
from movement.io import load_poses

ds = load_poses.from_anipose_file(
    "/path/to/file.csv", fps=30, individual_name="id_0"
)
```

You can also load any pandas DataFrame `df` that's formatted in the Anipose triangulation style:
```python
ds = load_poses.from_anipose_style_df(df, fps=30, individual_name="id_0")
```

**Save**

Not supported.
::::

::::{dropdown} COCO
:name: target-format-coco

**Load** (.json)

To load a [COCO keypoint detection results file](coco:format-results):
```python
ds = load_dataset("/path/to/results.json", source_software="COCO", fps=30)
```

COCO results files store keypoints as unnamed coordinate lists, so keypoints are assigned default names `keypoint_0`, `keypoint_1`, etc. in the resulting dataset.
To use the keypoint names defined for each category instead, you can also provide the corresponding [COCO annotations file](coco:format-data) in .json format:
```python
ds = load_dataset(
    "/path/to/results.json",
    source_software="COCO",
    fps=30,
    annotations_file="/path/to/annotations.json",
)
```

This is equivalent to calling {func}`~movement.io.load_poses.from_coco_file`:
```python
from movement.io import load_poses

ds = load_poses.from_coco_file(
    "/path/to/results.json",
    fps=30,
    annotations_file="/path/to/annotations.json",
)
```

:::{note}
COCO keypoint results do not contain track identities, so when there are multiple animals per frame, individuals may not correspond to the same animal across frames.
See {func}`~movement.io.load_poses.from_coco_file` for ways to handle this, and for further options such as using image IDs as frame numbers.
:::

**Save**

Not supported.
::::

::::{dropdown} VIA-tracks
:name: target-format-via

**Load** (.csv)

To load bounding box tracks from a [VGG Image Annotator (VIA)](via:) tracks .csv file:
```python
ds = load_dataset("/path/to/file.csv", source_software="VIA-tracks", fps=30)
```

To retain the original frame numbers from the file (rather than offsetting them so
that the first tracked frame is frame 0):
```python
ds = load_dataset(
    "/path/to/file.csv",
    source_software="VIA-tracks",
    use_frame_numbers_from_file=True,
)
```

This is equivalent to calling {func}`~movement.io.load_bboxes.from_via_tracks_file`:
```python
from movement.io import load_bboxes

ds = load_bboxes.from_via_tracks_file("/path/to/file.csv", fps=30)
```

:::{admonition} Bounding boxes format
:class: note
Note that the x,y coordinates in the input VIA tracks .csv file represent the top-left corner of each bounding box. Instead the corresponding `movement` dataset `ds` will hold in its `position` array the centroid of each bounding box.
:::

**Save** (.csv)

To save bounding box tracks as a [VIA tracks .csv file](via:docs/face_track_annotation.html), so that you can visualise and correct them with the [VGG Image Annotator (VIA-2) software](via:via.html):
```python
save_dataset(ds, "/path/to/file.csv", target_software="VIA-tracks")
```

To set a custom image file extension for the associated video frames:
```python
save_dataset(
    ds,
    "/path/to/file.csv",
    target_software="VIA-tracks",
    image_file_suffix=".jpg",
)
```

This is equivalent to calling {func}`~movement.io.save_bboxes.to_via_tracks_file`:
```python
from movement.io import save_bboxes

save_bboxes.to_via_tracks_file(ds, "/path/to/file.csv")
```

By default, the track IDs are derived from the trailing numbers in the individuals' names, but you can also set `track_ids_from_trailing_numbers=False` to assign the track IDs sequentially (0, 1, 2, ...) based on the alphabetically sorted list of individuals.
::::

::::{dropdown} NWB
:name: target-format-nwb

**Load** (.nwb)

To load an NWB file, optionally specifying which `ProcessingModule` and `PoseEstimation` object to load:
```python
ds = load_dataset(
    "/path/to/file.nwb",
    source_software="NWB",
    processing_module_key="behavior",
    pose_estimation_key="PoseEstimation",
)
```

This is equivalent to calling {func}`~movement.io.load_poses.from_nwb_file`:
```python
from movement.io import load_poses

ds = load_poses.from_nwb_file(
    "/path/to/file.nwb",
    processing_module_key="behavior",
    pose_estimation_key="PoseEstimation",
)
```

The `fps` argument is ignored for NWB files, as the frame rate is read or estimated from the file's metadata.

You can also pass an {class}`~pynwb.file.NWBFile` object instead of a file path:
```python
import pynwb

with pynwb.NWBHDF5IO("/path/to/file.nwb", mode="r") as io:
    nwb_file = io.read()
    ds = load_dataset(nwb_file, source_software="NWB")
```

**Save** (.nwb)

To save a poses dataset as NWB file(s):
```python
save_dataset(ds, "/path/to/file.nwb", target_software="NWB")
```
Since NWB files are designed to represent data from a single individual, multi-individual datasets are saved as one file per individual, with the individual's name appended to the file name (e.g. `file_Alice.nwb`, `file_Bob.nwb`).

Metadata such as session or subject information can be customised by passing a {class}`~movement.io.nwb.NWBFileSaveConfig` object as the `config` argument
(see the {func}`API reference<movement.io.save_poses.to_nwb_file_object>` for examples).

To add additional data to the NWB files before saving, use {func}`~movement.io.save_poses.to_nwb_file_object` instead.
This does not write to disk, but returns an {class}`~pynwb.file.NWBFile` object (or a list of them, one per individual) that you can then save with {class}`pynwb.NWBHDF5IO`:
```python
from pynwb import NWBHDF5IO
from movement.io import save_poses

nwb_files = save_poses.to_nwb_file_object(ds)

for file in nwb_files:
    with NWBHDF5IO(f"{file.identifier}.nwb", "w") as io:
        io.write(file)
```
::::

::::{dropdown} NumPy
:name: target-format-numpy

**Load**

To build a dataset from NumPy arrays, use the `from_numpy` function of the {mod}`~movement.io.load_poses` or {mod}`~movement.io.load_bboxes` module.

In the example below, we create random position data for two individuals, `Alice` and `Bob`, with three keypoints each: `snout`, `centre`, and `tail_base`. These keypoints are tracked in 2D space for 100 frames, at 30 fps. The confidence scores are set to 1 for all points.
```python
import numpy as np
from movement.io import load_poses

rng = np.random.default_rng(seed=42)
ds = load_poses.from_numpy(
    position_array=rng.random((100, 2, 3, 2)),
    confidence_array=np.ones((100, 3, 2)),
    individual_names=["Alice", "Bob"],
    keypoint_names=["snout", "centre", "tail_base"],
    fps=30,
)
```

Similarly, we can create random position data for two bounding boxes, `id_0` and `id_1`,
both with the same width (40 pixels) and height (30 pixels). These are tracked in 2D space for 100 frames, which will be numbered in the resulting dataset from 0 to 99. The confidence score for all bounding boxes is set to 0.5.
```python
import numpy as np
from movement.io import load_bboxes

rng = np.random.default_rng(seed=42)
ds = load_bboxes.from_numpy(
    position_array=rng.random((100, 2, 2)),
    shape_array=np.ones((100, 2, 2)) * [40, 30],
    confidence_array=np.ones((100, 2)) * 0.5,
    individual_names=["id_0", "id_1"],
)
```

**Save**

Any `movement` DataArray can be exported to a NumPy array with {meth}`xarray.DataArray.to_numpy`:
```python
position_array = ds.position.to_numpy()
```
::::

::::{dropdown} Custom .csv
:name: target-format-csv

**Save** (.csv)

You can also save the tracks to a .csv file with a custom header using the standard Python library `csv`.
Below is an example of how you can export a `movement` bounding boxes dataset this way:
```python
import csv

# define name for output csv file
filepath = "tracking_output.csv"

# open the csv file in write mode
with open(filepath, mode="w", newline="") as file:
    writer = csv.writer(file)

    # write the header
    writer.writerow(
        ["frame_idx", "bbox_ID", "x", "y", "width", "height", "confidence"]
    )

    # write the data
    for individual in ds.individual.data:
        for frame in ds.time.data:
            x, y = ds.position.sel(time=frame, individual=individual).data
            width, height = ds.shape.sel(
                time=frame, individual=individual
            ).data
            confidence = ds.confidence.sel(
                time=frame, individual=individual
            ).data
            writer.writerow(
                [frame, individual, x, y, width, height, confidence]
            )
```
:::{admonition} Using `pandas`
:class: note
If you prefer to work with `pandas`, you can alternatively convert the `movement` dataset to a `pandas` DataFrame with the {meth}`xarray.Dataset.to_dataframe` method, wrangle the dataframe as required, and then apply the {meth}`pandas.DataFrame.to_csv` method to save the data as a .csv file.
:::
::::

(target-netcdf)=
## Working with netCDF

Because `movement` datasets are {class}`xarray.Dataset` objects,
they can be saved to any of the [file formats supported by xarray](xarray:user-guide/io.html).
`movement` uses netCDF as its native file format: a binary format for self-described datasets
that originated in the geosciences.
netCDF files on disk directly correspond to {class}`xarray.Dataset` objects,
making them the recommended way to preserve the complete state of your analysis,
including all variables, coordinates, and attributes.

Below is an example of how you may integrate netCDF into your
`movement`-powered workflows, using {func}`~movement.io.load.load_dataset`
and {func}`~movement.io.save.save_dataset`
(see [](target-format-movement)):

```python
from movement.io import load_dataset, save_dataset
from movement.filtering import rolling_filter
from movement.kinematics import compute_speed

ds = load_dataset("path/to/my_data.h5", source_software="DeepLabCut", fps=30)

# Apply a rolling median filter to smooth the position data
ds["position_smooth"] = rolling_filter(
    ds["position"], window=5, statistic="median"
)

# Compute speed based on the smoothed position data
ds["speed"] = compute_speed(ds["position_smooth"])

# Save the dataset to a netCDF file
# This includes the original position and confidence data,
# the smoothed position, and the computed speed
save_dataset(ds, "path/to/my_data_processed.nc")
```

:::{tip}
In some cases you may want to use our [GUI](target-gui) to inspect
motion tracks you've processed with `movement`.
If so, make sure to save them to a netCDF file that satisfies the
[GUI compatibility requirements](target-gui-compatible-netcdf).
:::

### Skip validation

If you prefer to skip `movement`-specific validation (e.g. for an arbitrary, non-`movement`-specific dataset),
you can use xarray's native [I/O methods](xarray:user-guide/io.html),
{meth}`xarray.Dataset.to_netcdf` and {func}`xarray.load_dataset`, instead.
Note that such files may not load in the GUI.

```python
import xarray as xr

# Save
ds.to_netcdf("/path/to/my_data.nc")

# Load
ds = xr.load_dataset("/path/to/my_data.nc")
```

Similarly, an {class}`xarray.DataArray` object (e.g. the `position` variable
of a `movement` dataset) can be saved to disk using
{meth}`xarray.DataArray.to_netcdf`, and loaded from disk using
{func}`xarray.open_dataarray`.
As netCDF files correspond to Dataset objects,
these functions internally convert the DataArray to a Dataset before saving,
and then convert back when loading.

:::{note}
xarray also supports compression and chunking options with netCDF, which can be useful for managing large datasets.
For more details, see the [xarray documentation on netCDF](xarray:user-guide/io.html).
:::

(target-sample-data)=
## Sample data

`movement` includes some sample data files that you can use to
try the package out. These files contain pose and bounding box tracks from
various [supported formats](target-supported-formats).

You can list the available sample data files using:

```python
from movement import sample_data

file_names = sample_data.list_datasets()
print(*file_names, sep="\n")  # print each sample file in a separate line
```

Each sample file is prefixed with the name (or abbreviation)
of the software package that was used to generate it.

To load one of the sample files as a
[movement dataset](target-poses-and-bboxes-dataset), use the
{func}`~movement.sample_data.fetch_dataset` function:

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
(e.g. as a background image for plotting trajectories). If available,
this file is always downloaded when fetching the dataset,
and its path is stored in the `frame_path` attribute
(i.e., `ds.frame_path`). If no frame file is available for the dataset,
the `frame_path` attribute will not be set.

:::{dropdown} Under the hood
:color: info
:icon: info
When you import the {mod}`~movement.sample_data` module with `from movement import sample_data`,
`movement` downloads a small metadata file to your local machine with information about the latest sample datasets available. Then, the first time you call the {func}`~movement.sample_data.fetch_dataset` function, `movement` downloads the requested file to your machine and caches it in the `~/.movement/data` directory. On subsequent calls, the data are directly loaded from this local cache.
:::
