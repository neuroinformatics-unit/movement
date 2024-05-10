(target-dataset)=
# movement dataset

When you load predicted pose tracks into `movement`, they are represented
as an {class}`xarray.Dataset` object, which is a container for multiple data
arrays. Each array is in turn represented as a {class}`xarray.DataArray`
object, which you can think of as a multi-dimensional {class}`numpy.ndarray`
with `pandas`-style indexing and labelling.

So, a *movement dataset* is simply a :py:class:`xarray.Dataset` with a specific
structure to represent pose tracks, associated confidence scores and relevant
metadata. Each dataset consists of *data variables*, *dimensions*,
*coordinates* and *attributes*. In what follows, we describe each of these
components in the context of a *movement dataset*.

To learn more about `xarray` data structures in general, see the relevant
[documentation](xarray:user-guide/data-structures.html)

![](../_static/dataset_structure.png)

## Dimensions and coordinates
In `xarray` [terminology](xarray:user-guide/plotting.html),
each axis is called a *dimension* (`dim`), while
the labelled "ticks" along each axis are called *coordinates* (`coords`).

A *movement dataset* has the following *dimensions*:
- `time`, with size equal to the number of frames in the video
- `individuals`, with size equal to the number of tracked individuals/instances
- `keypoints`, with size equal to the number of tracked keypoints per individual
- `space`: the number of spatial dimensions, either two (2D) or three (3D)

Appropriate *coordinates* are assigned to each *dimension*.
- `individuals` are labelled with a list of unique names (str), e.f. `mouse1`, `mouse2`, etc.
- `keypoints` are likewise labelled with a list of unique body part names (str), e.g. `snout`, `right_ear`, etc.
- `space` is labelled with either `x`, `y` (2D) or `x`, `y`, `z` (3D).
- `time` is labelled in seconds if `fps` is provided, otherwise the *coordinates* are expressed in frames (ascending 0-indexed integers).

## Data variables

A *movement dataset* contains two *data variables* stored as {class}`xarray.DataArray` objects:
- `position`: the 2D or 3D locations of the keypoints over time, with shape (`time`, `individuals`, `keypoints`, `space`).
- `confidence`: the point-wise confidence scores associated with each predicted keypoint (as reported by the pose estimation model), with shape (`time`, `individuals`, `keypoints`).

Grouping *data variables* together in a single dataset makes it easier to
keep track of the relationships between them, and makes sense when they
share some common *dimensions* (as is the case here).

## Attributes

Related metadata that do not constitute arrays but instead take the form
of key-value pairs can be stored as *attributes* - i.e. inside the specially
designated `attrs` dictionary. Right after loading a *movement dataset*,
the following *attributes* are stored:
- `fps`: the number of frames per second in the video
- `time_unit`: the unit of the `time` *coordinates*, frames or seconds
- `source_software`: the software from which the poses were loaded
- `source_file`: the path to the file from which the poses were loaded

Some of the sample datasets provided with the `movement` package
may also possess additional *attributes*, such as:
- `video_path`: the path to the video file corresponding to the pose tracks
- `frame_path`: the path to a single still frame from the video

## Using xarray's built-in functionality

Since a *movement dataset* is an `xarray.Dataset` consisting of
`xarray.DataArray` objects, you can use all of `xarray`'s intuitive interface
and rich built-in functionalities for data manipulation and analysis.
For example, you can access the **data variables** and **attributes** of
a dataset `ds` using 'dot' notation (e.g. `ds.position`, `ds.fps`),
[index and select data](xarray:user-guide/indexing.html) by *coordinate* label
(`ds.sel()`) or position (`ds.isel()`), use *dimension* names for intelligent
[data aggregation and broadcasting](xarray:user-guide/computation.html),
and use the built-in [plotting methods](xarray:user-guide/plotting.html).

## Accessing `movement`-specific functionality

`movement` extends `xarray`'s functionality with a number of convenience
methods that are specific to the data structure of a *movement dataset*
described above. This additional functionality is implemented in the
:py:class:`movement.MovementDataset` class, which is an accessor to the
underlying `xarray.Dataset` object. To avoid conflicts with `xarray`'s
built-in methods, `movement`-specific methods are accessed using the
`move` keyword, for example `ds.move.compute_velocity()`.

## Adding data variable and attributes to a dataset

The `movement dataset` structure we described above is the default structure
created after loading pose tracks into `movement`. However, you can also
add more *data variables* and *attributes* to a dataset, if and when it makes
sense for your convenience. For example, this could be desirable for storing
*data variables* that are derived from the original pose tracks, and share
some or all of the same *dimensions*. For example, to add a `velocity` data
variable to a dataset `ds`, run `ds["velocity"] = ds.move.compute_velocity()`.
The produced `velocity` *data variable* has the same *dimensions* as the
original `position` *data variable*, and can be used in the same way.
