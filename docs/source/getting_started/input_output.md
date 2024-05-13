(target-io)=
# Input/Output

(target-formats)=
## Supported formats
(target-supported-formats)=
`movement` can load pose tracks from various pose estimation frameworks.
Currently, these include:
- [DeepLabCut](dlc:)
- [SLEAP](sleap:),
- [LightingPose](lp:)

:::{warning}
`movement` only deals with the predicted pose tracks output by these
software packages. It does not support the training or labelling of the data.
:::

(target-loading)=
## Loading pose tracks

The loading functionalities are provided by the
`movement.io.load_poses` module, which can be imported as follows:

```python
from movement.io import load_poses
```

Depending on the source sofrware, one of the following functions can be used.

::::{tab-set}

:::{tab-item} SLEAP

Load from [SLEAP analysis .h5 files](sleap:tutorials/analysis):
```python
ds = load_poses.from_sleap_file("/path/to/file.analysis.h5", fps=30)

# or equivalently
ds = load_poses.from_file(
    "/path/to/file.analysis.h5", source_software="SLEAP", fps=30
)
```
:::

:::{tab-item} DeepLabCut

Load from DeepLabCut (DLC) .h5 files:
```python
ds = load_poses.from_dlc_file("/path/to/file.h5", fps=30)

# or equivalently
ds = load_poses.from_file(
    "/path/to/file.h5", source_software="DeepLabCut", fps=30
)
```

You may also load .csv files
(assuming they are formatted as DeepLabCut expects them):
```python
ds = load_poses.from_dlc_file("/path/to/file.csv", fps=30)
```

If you have already imported the data into a {class}`pandas.DataFrame` named
`df` you can convert it to a movement dataset with:
```python
ds = load_poses.from_dlc_df(df, fps=30)
```
:::

:::{tab-item} LightningPose

Load from LightningPose (LP) .csv files:
```python
ds = load_poses.from_lp_file("/path/to/file.analysis.csv", fps=30)

# or equivalently
ds = load_poses.from_file(
    "/path/to/file.analysis.csv", source_software="LightningPose", fps=30
)
```
:::

::::

The loaded data include the predicted positions for each individual and
keypoint as well as the associated point-wise confidence values, as reported by
the pose estimation software. See the [movement dataset](target-dataset) page
for more information on data structure.

You can also try `movement` out on some [sample data](target-sample-data)
included with the package.

(target-saving)=
## Saving pose tracks
[movement datasets](target-dataset) can be save to disk in a variety of
formats, including DeepLabCut-style files (.h5 or .csv) and
[SLEAP-style analysis files](sleap:tutorials/analysis) (.h5).

First import the `movement.io.save_poses` module:

```python
from movement.io import save_poses
```

Then, depending on the desired format, use one of the following functions:

:::::{tab-set}

::::{tab-item} SLEAP

Save to SLEAP-style analysis files (.h5):
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

Save to DeepLabCut-style files (.h5 or .csv):
```python
save_poses.to_dlc_file(ds, "/path/to/file.h5")  # preferred format
save_poses.to_dlc_file(ds, "/path/to/file.csv")
```

The {func}`movement.io.save_poses.to_dlc_file` function also accepts
a `split_individuals` boolean argument. If set to `True`, the function will
split the data into single-animal DeepLabCut-style files.

::::

::::{tab-item} LightningPose

Save to LightningPose (LP) files (.csv).
```python
save_poses.to_lp_file(ds, "/path/to/file.csv")
```
:::{note}
Because LP saves pose estimation outputs in the same format as single-animal
DeepLabCut projects, the above command is equivalent to:
```python
save_poses.to_dlc_file(ds, "/path/to/file.csv", split_individuals=True)
```
:::

::::
:::::
