(target-sample-data)=
# Sample data

movement includes some sample data files that you can use to
try the package out. These files contain pose and bounding boxes' tracks from
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
The value of this attribute is `None` if no video file is
available for this dataset, or if you did not request it.

Some datasets also include a sample frame file, which is a single
still frame extracted from the video. This can be useful for visualisation
(e.g., as a background image for plotting trajectories). If available,
this file is always downloaded when fetching the dataset,
and its path is stored in the `frame_path` attribute
(i.e., `ds.frame_path`). If no frame file is available for the dataset,
 `ds.frame_path=None`.

:::{dropdown} Under the hood
:color: info
:icon: info
When you import the `sample_data` module with `from movement import sample_data`,
movement downloads a small metadata file to your local machine with information about the latest sample datasets available. Then, the first time you call the `fetch_dataset()` function, movement downloads the requested file to your machine and caches it in the `~/.movement/data` directory. On subsequent calls, the data are directly loaded from this local cache.
:::
