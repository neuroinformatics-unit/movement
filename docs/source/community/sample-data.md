(target-contributing-sample-data)=
# Sample data
We maintain some sample datasets to be used for testing, examples and tutorials on an
[external data repository](gin:neuroinformatics/movement-test-data).
Our hosting platform of choice is called [GIN](gin:) and is maintained
by the [German Neuroinformatics Node](https://www.g-node.org/).
GIN has a GitHub-like interface and git-like
[CLI](gin:G-Node/Info/wiki/GIN+CLI+Setup#quickstart) functionalities.

Currently, the data repository contains sample pose estimation data files
stored in the `poses` folder, and tracked bounding boxes data files under the `bboxes` folder. For some of these files, we also host
the associated video file (in the `videos` folder) and/or a single
video frame (in the `frames`) folder. These can be used to develop and
test visualisations, e.g. to overlay the data on video frames.
The `metadata.yaml` file holds metadata for each sample dataset,
including information on data provenance as well as the mapping between data files and related
video/frame files.

For most sample datasets, the tracking data lives in a single file under `poses` or `bboxes`.
However, some tools—like [TRex](TRex:)—may split their tracking outputs across multiple files.
In those cases, the dataset is distributed as a ZIP archive containing every relevant file, and is automatically extracted when fetched.

## Fetching data
To fetch the data from GIN, we use the [pooch](https://www.fatiando.org/pooch/latest/index.html)
Python package, which can download data from pre-specified URLs and store them
locally for all subsequent uses. It also provides some nice utilities,
like verification of sha256 hashes and decompression of archives.

The relevant functionality is implemented in the {mod}`movement.sample_data` module.
The most important parts of this module are:

1. The `SAMPLE_DATA` download manager object.
2. The {func}`list_datasets()<movement.sample_data.list_datasets>` function, which returns a list of the available poses and bounding boxes datasets (file names of the data files).
3. The {func}`fetch_dataset_paths()<movement.sample_data.fetch_dataset_paths>` function, which returns a dictionary containing local paths to the files associated with a particular sample dataset: `poses` or `bboxes`, `frame`, `video`. If the relevant files are not already cached locally, they will be downloaded.
4. The {func}`fetch_dataset()<movement.sample_data.fetch_dataset>` function, which downloads the files associated with a given sample dataset (same as `fetch_dataset_paths()`) and additionally loads the pose or bounding box data into `movement`, returning an `xarray.Dataset` object. If available, the local paths to the associated video and frame files are stored as dataset attributes, with names `video_path` and `frame_path`, respectively.

By default, the downloaded files are stored in the `~/.movement/data` folder.
This can be changed by setting the `DATA_DIR` variable in the `sample_data.py` file.

## Adding new data
Only core `movement` developers may add new files to the external data repository.
Make sure to run the following procedure on a UNIX-like system, as we have observed some weird behaviour on Windows (some sha256sums may end up being different).
To add a new file, you will need to:

1. Create a [GIN](gin:) account.
2. Request collaborator access to the [movement data repository](gin:neuroinformatics/movement-test-data) if you don't already have it.
3. Install and configure the [GIN CLI](gin:G-Node/Info/wiki/GIN+CLI+Setup#quickstart) by running `gin login` in a terminal with your GIN credentials.
4. Clone the `movement` data repository to your local machine using `gin get neuroinformatics/movement-test-data`, then run `gin download --content` to download all the files.
5. Add your new files to the appropriate folders (`poses`, `bboxes`, `videos`, and/or `frames`) following the existing file naming conventions.
6. Add metadata for your new files to `metadata.yaml` using the [example entry below](target-metadata-yaml) as a template. You can leave all `sha256sum` values as `null` for now.
7. Update file hashes in `metadata.yaml` by running `python update_hashes.py` from the root of the [movement data repository](gin:neuroinformatics/movement-test-data). This script computes SHA256 hashes for all data files and updates the corresponding `sha256sum` values in the metadata file. Make sure you're in a [Python environment with `movement` installed](target-creating-a-development-environment).
8. Commit your changes using `gin commit -m <message> <filename>` for specific files or `gin commit -m <message> .` for all changes.
9. Upload your committed changes to the GIN repository with `gin upload`. Use `gin download` to pull the latest changes or `gin sync` to synchronise changes bidirectionally.
10. [Verify](target-verify-sample-data) the new files can be fetched and loaded correctly using the {mod}`movement.sample_data` module.

(target-metadata-yaml)=
## `metadata.yaml` example entry
```yaml
SLEAP_three-mice_Aeon_proofread.analysis.h5:
  sha256sum: null
  source_software: SLEAP
  type: poses
  fps: 50
  species: mouse
  number_of_individuals: 3
  shared_by:
    name: Chang Huan Lo
    affiliation: Sainsbury Wellcome Centre, UCL
  frame:
    file_name: three-mice_Aeon_frame-5sec.png
    sha256sum: null
  video:
    file_name: three-mice_Aeon_video.avi
    sha256sum: null
  note: All labels were proofread (user-defined) and can be considered ground truth.
    It was exported from the .slp file with the same prefix.
```

(target-verify-sample-data)=
## Verifying sample data

To verify that a sample dataset can be fetched and loaded correctly:

```python
from movement import sample_data

# Fetch and load the dataset
ds = sample_data.fetch_dataset("SLEAP_three-mice_Aeon_proofread.analysis.h5")

# Verify it loaded correctly
print(ds)
```

This displays the dataset's structure (dimensions, coordinates, data variables,
and attributes), confirming the data was loaded successfully.

If the sample dataset also includes a video, pass `with_video=True` to
verify that the video is correctly linked to the dataset:

```python
ds = sample_data.fetch_dataset(
    "SLEAP_three-mice_Aeon_proofread.analysis.h5",
    with_video=True,
)
print(ds.video_path)
```
