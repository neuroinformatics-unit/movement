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

This prints a list of file names containing sample pose data.
Each file is prefixed with the name of the pose estimation software package
that was used to generate it - either "DLC", "SLEAP", or "LP".

To load one of the sample datasets as a  
[movement dataset](target-dataset), use the  
{func}`movement.sample_data.fetch_dataset()` function:

```python
filename = "SLEAP_three-mice_Aeon_proofread.analysis.h5"
ds = sample_data.fetch_dataset(filename)
```
Some sample datasets also have an associated video file
(the video from which the poses were predicted),
which you can request by setting `with_video=True`:

```python
ds = sample_data.fetch_dataset(filename, with_video=True)
```

If available, the video file is downloaded and its path is stored
in the `video_path` attribute of the dataset (i.e., `ds.video_path`).
The value of this attribute is `None` if no video file is
available for this dataset or if you did not request it
(`with_video=False`, which is the default).

Some datasets also have an associated frame file, which is a single
still frame extracted from the video. This can be useful for visualisation
(e.g., as a background image for plotting trajectories). If available,
this file is always downloaded when fetching the dataset,
and its path is stored in the `frame_path` attribute
(i.e., `ds.frame_path`). If no frame file is available for the dataset,
this attribute's value is `None`.

:::{note}
Under the hood, the first time you call the `fetch_dataset()` function,
it downloads the corresponding files to your local machine and caches them
in the `~/.movement/data` directory. On subsequent calls, the data are directly
loaded from this local cache.
:::
