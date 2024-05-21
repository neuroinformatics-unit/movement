(target-sample-data)=
# Sample data

`movement` includes some sample data files that you can use to
try out the package. These files contain predicted pose tracks from
various [supported formats](target-supported-formats).

You can list the available sample data files using:

```python
from movement import sample_data

file_names = sample_data.list_datasets()
print(file_names)
```

This will print a list of file names containing sample pose data.
Each file is prefixed with the name of the pose estimation software package
that was used to generate it - either "DLC", "SLEAP", or "LP".

To load one of the sample datasets, you can use the
{func}`movement.sample_data.fetch_dataset()` function:

```python
ds = sample_data.fetch_dataset("DLC_two-mice.predictions.csv")
```
This function loads the sample pose data as a
[movement dataset](target-dataset). Some sample datasets may also have an
associated video file (the video based on which the poses were predicted)
or a single frame extracted from that video. These files are not directly
loaded into the `movement` dataset, but their paths can be accessed as dataset attributes:

```python
ds.frame_path
ds.video_path
```
If the value of one of these attributes is `None`, it means that the
associated file is not available for the sample dataset.

Under the hood, the first time you call the `fetch_dataset()` function,
it will download the corresponding files to your local machine and cache them
in the `~/.movement/data` directory. On subsequent calls, the data are directly
loaded from the local cache.
