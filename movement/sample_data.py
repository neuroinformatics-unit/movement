"""Fetch and load sample datasets.

This module provides functions for fetching and loading sample data used in
tests, examples, and tutorials. The data are stored in a remote repository
on GIN and are downloaded to the user's local machine the first time they
are used.
"""

import logging
from pathlib import Path

import pooch
import xarray
import yaml
from requests.exceptions import RequestException

from movement.io import load_bboxes, load_poses
from movement.utils.logging import log_error, log_warning

logger = logging.getLogger(__name__)

# URL to the remote data repository on GIN
# noinspection PyInterpreter
DATA_URL = (
    "https://gin.g-node.org/neuroinformatics/movement-test-data/raw/master"
)

# Save data in ~/.movement/data
DATA_DIR = Path("~", ".movement", "data").expanduser()
# Create the folder if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File name for the .yaml file in DATA_URL containing dataset metadata
METADATA_FILE = "metadata.yaml"


def _download_metadata_file(file_name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download the metadata yaml file.

    This function downloads the yaml file containing sample metadata from
    the ``movement`` data repository and saves it in the specified directory
    with a temporary filename - temp_{file_name} - to avoid overwriting any
    existing files.

    Parameters
    ----------
    file_name : str
        Name of the metadata file to fetch.
    data_dir : pathlib.Path, optional
        Directory to store the metadata file in. Defaults to the constant
        ``DATA_DIR``. Can be overridden for testing purposes.

    Returns
    -------
    path : pathlib.Path
        Path to the downloaded file.

    """
    local_file_path = pooch.retrieve(
        url=f"{DATA_URL}/{file_name}",
        known_hash=None,
        path=data_dir,
        fname=f"temp_{file_name}",
        progressbar=False,
    )
    logger.debug(
        f"Successfully downloaded sample metadata file {file_name} "
        f"from {DATA_URL} to {data_dir}"
    )
    return Path(local_file_path)


def _fetch_metadata(
    file_name: str, data_dir: Path = DATA_DIR
) -> dict[str, dict]:
    """Download the metadata yaml file and load it as a dictionary.

    Parameters
    ----------
    file_name : str
        Name of the metadata file to fetch.
    data_dir : pathlib.Path, optional
        Directory to store the metadata file in. Defaults to
        the constant ``DATA_DIR``. Can be overridden for testing purposes.

    Returns
    -------
    dict
        A dictionary containing metadata for each sample dataset, with the
        dataset file name as the key.

    """
    local_file_path = Path(data_dir / file_name)
    failed_msg = "Failed to download the newest sample metadata file."

    # try downloading the newest metadata file
    try:
        downloaded_file_path = _download_metadata_file(file_name, data_dir)
        # if download succeeds, replace any existing local metadata file
        downloaded_file_path.replace(local_file_path)
    # if download fails, try loading an existing local metadata file,
    # otherwise raise an error
    except RequestException as exc_info:
        if local_file_path.is_file():
            log_warning(
                f"{failed_msg} Will use the existing local version instead."
            )
        else:
            raise log_error(RequestException, failed_msg) from exc_info

    with open(local_file_path) as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    return metadata


def _generate_file_registry(metadata: dict[str, dict]) -> dict[str, str]:
    """Generate a file registry based on the contents of the metadata.

    This includes files containing poses, frames, videos, or bounding boxes
    data.

    Parameters
    ----------
    metadata : dict
        List of dictionaries containing metadata for each sample dataset.

    Returns
    -------
    dict
        Dictionary mapping file paths to their SHA-256 checksums.

    """
    file_registry = {}
    for ds, val in metadata.items():
        file_registry[f"{val['type']}/{ds}"] = val["sha256sum"]
        for key in ["video", "frame"]:
            file_name = val[key]["file_name"]
            if file_name:
                file_registry[f"{key}s/{file_name}"] = val[key]["sha256sum"]
    return file_registry


# Create a download manager for the sample data
metadata = _fetch_metadata(METADATA_FILE, DATA_DIR)
file_registry = _generate_file_registry(metadata)
SAMPLE_DATA = pooch.create(
    path=DATA_DIR,
    base_url=f"{DATA_URL}/",
    retry_if_failed=0,
    registry=file_registry,
)


def list_datasets() -> list[str]:
    """List available sample datasets.

    Returns
    -------
    filenames : list of str
        List of filenames for available sample datasets.

    """
    return list(metadata.keys())


def fetch_dataset_paths(filename: str, with_video: bool = False) -> dict:
    """Get paths to sample dataset and any associated frames or videos.

    The data are downloaded from the ``movement`` data repository to the user's
    local machine upon first use and are stored in a local cache directory.
    The function returns the paths to the downloaded files.

    Parameters
    ----------
    filename : str
        Name of the sample data file to fetch.
    with_video : bool, optional
        Whether to download the associated video file (if available). If set
        to False, the "video" entry in the returned dictionary will be None.
        Defaults to False.

    Returns
    -------
    paths : dict
        Dictionary mapping file types to their respective paths. The possible
        file types are: "poses" or "bboxes" (depending on tracking type),
        "frame", "video". A None value for "frame" or "video" indicates that
        the file is either not available or not requested
        (if ``with_video=False``).

    Examples
    --------
    Fetch a sample dataset and get the paths to the file containing the
    predicted poses, as well as the associated frame and video files:

    >>> from movement.sample_data import fetch_dataset_paths
    >>> paths = fetch_dataset_paths(
    ...     "DLC_single-mouse_EPM.predictions.h5", with_video=True
    ... )
    >>> poses_path = paths["poses"]
    >>> frame_path = paths["frame"]
    >>> video_path = paths["video"]

    If the sample dataset contains bounding boxes instead of
    poses, use ``paths["bboxes"]`` instead of ``paths["poses"]``:

    >>> paths = fetch_dataset_paths("VIA_multiple-crabs_5-frames_labels.csv")
    >>> bboxes_path = paths["bboxes"]


    See Also
    --------
    fetch_dataset

    """
    available_data_files = list_datasets()
    if filename not in available_data_files:
        raise log_error(
            ValueError,
            f"File '{filename}' is not in the registry. "
            f"Valid filenames are: {available_data_files}",
        )

    frame_file_name = metadata[filename]["frame"]["file_name"]
    video_file_name = metadata[filename]["video"]["file_name"]
    paths_dict = {
        "frame": None
        if not frame_file_name
        else Path(
            SAMPLE_DATA.fetch(f"frames/{frame_file_name}", progressbar=True)
        ),
        "video": None
        if (not video_file_name) or not (with_video)
        else Path(
            SAMPLE_DATA.fetch(f"videos/{video_file_name}", progressbar=True)
        ),
    }
    # Add trajectory data
    # Assume "poses" if not of type "bboxes"
    data_type = "bboxes" if metadata[filename]["type"] == "bboxes" else "poses"
    paths_dict[data_type] = Path(
        SAMPLE_DATA.fetch(f"{data_type}/{filename}", progressbar=True)
    )
    return paths_dict


def fetch_dataset(
    filename: str,
    with_video: bool = False,
) -> xarray.Dataset:
    """Load a sample dataset.

    The data are downloaded from the ``movement`` data repository to the user's
    local machine upon first use and are stored in a local cache directory.
    This function returns the data as an xarray Dataset.
    If there are any associated frames or videos, these files are also
    downloaded and the paths are stored as dataset attributes.

    Parameters
    ----------
    filename : str
        Name of the file to fetch.
    with_video : bool, optional
        Whether to download the associated video file (if available). If set
        to False, the "video" entry in the returned dictionary will be None.
        Defaults to False.

    Returns
    -------
    ds : xarray.Dataset
        Data contained in the fetched sample file.

    Examples
    --------
    Fetch a sample dataset and get the paths to the associated frame and video
    files:

    >>> from movement.sample_data import fetch_dataset
    >>> ds = fetch_dataset(
        "DLC_single-mouse_EPM.predictions.h5", with_video=True
    )
    >>> frame_path = ds.video_path
    >>> video_path = ds.frame_path

    See Also
    --------
    fetch_dataset_paths

    """
    file_paths = fetch_dataset_paths(filename, with_video=with_video)

    for key, load_module in zip(
        ["poses", "bboxes"], [load_poses, load_bboxes], strict=False
    ):
        if file_paths.get(key):
            ds = load_module.from_file(
                file_paths[key],
                source_software=metadata[filename]["source_software"],
                fps=metadata[filename]["fps"],
            )

    ds.attrs["frame_path"] = file_paths["frame"]
    ds.attrs["video_path"] = file_paths["video"]

    return ds
