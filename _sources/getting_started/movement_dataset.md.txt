(target-dataset)=
# The movement dataset

When you load predicted pose tracks into `movement`, they are represented
as an {class}`xarray.Dataset` object, which is a container for multiple data
arrays. Each array is in turn represented as an {class}`xarray.DataArray`
object, which you can think of as a multi-dimensional {class}`numpy.ndarray`
with pandas-style indexing and labelling.

So, a `movement` dataset is simply an {class}`xarray.Dataset` with a specific
structure to represent pose tracks, associated confidence scores and relevant
metadata. Each dataset consists of **data variables**, **dimensions**,
**coordinates** and **attributes**.

In the next section, we will describe the
structure of a `movement` dataset in some detail.
To learn more about `xarray` data structures in general, see the relevant
[documentation](xarray:user-guide/data-structures.html).


## Dataset structure

![](../_static/dataset_structure.png)

You can always inspect the structure of a `movement` dataset `ds` by simply
printing it:
```python
ds = load_poses.from_dlc_file("/path/to/file.h5", fps=30)
print(ds)
```
If you are working in a Jupyter notebook, you can also view an interactive
representation of the dataset by typing its name - e.g. `ds` - in a cell.

### Dimensions and coordinates
In `xarray` [terminology](xarray:user-guide/terminology.html),
each axis is called a **dimension** (`dim`), while
the labelled "ticks" along each axis are called **coordinates** (`coords`).

A `movement` dataset has the following **dimensions**:
- `time`, with size equal to the number of frames in the video
- `individuals`, with size equal to the number of tracked individuals/instances
- `keypoints`, with size equal to the number of tracked keypoints per individual
- `space`: the number of spatial dimensions, either two (2D) or three (3D)

Appropriate **coordinates** are assigned to each **dimension**.
- `individuals` are labelled with a list of unique names, e.g. `mouse1`, `mouse2`, etc.
- `keypoints` are likewise labelled with a list of unique body part names, e.g. `snout`, `right_ear`, etc.
- `space` is labelled with either `x`, `y` (2D) or `x`, `y`, `z` (3D).
- `time` is labelled in seconds if `fps` is provided, otherwise the **coordinates** are expressed in frames (ascending 0-indexed integers).

### Data variables

A `movement` dataset contains two **data variables** stored as {class}`xarray.DataArray` objects:
- `position`: the 2D or 3D locations of the keypoints over time, with shape (`time`, `individuals`, `keypoints`, `space`).
- `confidence`: the point-wise confidence scores associated with each predicted keypoint (as reported by the pose estimation model), with shape (`time`, `individuals`, `keypoints`).

Grouping **data variables** together in a single dataset makes it easier to
keep track of the relationships between them, and makes sense when they
share some common **dimensions** (as is the case here).

### Attributes

Related metadata that do not constitute arrays but instead take the form
of key-value pairs can be stored as **attributes** - i.e. inside the specially
designated `attrs` dictionary. Right after loading a `movement` dataset,
the following **attributes** are stored:
- `fps`: the number of frames per second in the video
- `time_unit`: the unit of the `time` **coordinates**, frames or seconds
- `source_software`: the software from which the poses were output
- `source_file`: the path to the file from which the poses were loaded

Some of the [sample datasets](target-sample-data) provided with
the `movement` package may also possess additional **attributes**, such as:
- `video_path`: the path to the video file corresponding to the pose tracks
- `frame_path`: the path to a single still frame from the video

## Working with movement datasets

### Using xarray's built-in functionality

Since a `movement` dataset is an {class}`xarray.Dataset` consisting of
{class}`xarray.DataArray` objects, you can use all of `xarray`'s intuitive interface
and rich built-in functionalities for data manipulation and analysis.
For example, you can access the **data variables** and **attributes** of
a dataset `ds` using 'dot' notation (e.g. `ds.position`, `ds.fps`),
[index and select data](xarray:user-guide/indexing.html) by **coordinate** label
(`ds.sel()`) or position (`ds.isel()`), use **dimension** names for intelligent
[data aggregation and broadcasting](xarray:user-guide/computation.html),
and use the built-in [plotting methods](xarray:user-guide/plotting.html).

As an example, here's how you can use the `sel` method to select subsets of
data:

```python
# select the first 100 seconds of data
ds_sel = ds.sel(time=slice(0, 100))

# select specific individuals or keypoints
ds_sel = ds.sel(individuals=["individual1", "individual2"])
ds_sel = ds.sel(keypoints="snout")

# combine selections
ds_sel = ds.sel(
    time=slice(0, 100),
    individuals=["individual1", "individual2"],
    keypoints="snout"
)
```
Such selections can also be applied to the **data variables**,
returning an {class}`xarray.DataArray` rather than an {class}`xarray.Dataset`:

```python
position = ds.position.sel(individuals="individual1", keypoints="snout")
```

### Accessing movement-specific functionality

`movement` extends `xarray`'s functionality with a number of convenience
methods that are specific to the structure of a `movement` dataset, as
described above. The additional functionality is implemented in the
{class}`movement.move_accessor.MovementDataset` class, which is an accessor to the
underlying {class}`xarray.Dataset` object. To avoid conflicts with `xarray`'s
built-in methods, `movement`-specific methods are accessed using the
`move` keyword, for example:

```python
# compute position derivatives for all individuals and keypoints across time
velocity = ds.move.compute_velocity()
acceleration = ds.move.compute_acceleration()
```

### Modifying movement datasets

The `velocity` and `acceleration` produced in the above example are also
{class}`xarray.DataArray` objects, with the same **dimensions** as the
original `position` **data variable**. In some cases, you may wish to
add these or other new **data variables** to the `movement` dataset for
convenience, which can be done by simply assigning them to the dataset
with an appropriate name:

```python
ds["velocity"] = velocity
ds["acceleration"] = acceleration
# henceforth accessible as ds.velocity and ds.acceleration
```

Custom **attributes** can also be added to the dataset:

```python
ds.attrs["my_custom_attribute"] = "my_custom_value"
# henceforth accessible as ds.my_custom_attribute
```
