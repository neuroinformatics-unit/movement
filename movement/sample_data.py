"""Module for fetching and loading sample datasets.

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

from movement.io import load_poses
from movement.logging import log_error, log_warning

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


def _download_metadata_file(file_name: str, data_dir: Path = DATA_DIR) -> Path:
    """Download the metadata yaml file.

    This function downloads the yaml file containing sample metadata from
    the *movement* data repository and saves it in the specified directory
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


def _fetch_metadata(file_name: str, data_dir: Path = DATA_DIR) -> list[dict]:
    """Download the metadata yaml file and load it as a list of dictionaries.

    Parameters
    ----------
    file_name : str
        Name of the metadata file to fetch.
    data_dir : pathlib.Path, optional
        Directory to store the metadata file in. Defaults to
        the constant ``DATA_DIR``. Can be overridden for testing purposes.

    Returns
    -------
    list[dict]
        A list of dictionaries containing metadata for each sample file.

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


def _generate_file_registry(metadata: list[dict]) -> dict[str, str]:
    """Generate a file registry based on the contents of the metadata.

    This includes files containing pose data, frames, or entire videos.

    Parameters
    ----------
    metadata : list of dict
        List of dictionaries containing metadata for each sample dataset.

    Returns
    -------
    dict
        Dictionary mapping file paths to their SHA-256 checksums.

    """
    poses_registry = {
        "poses/" + ds["file_name"]: ds["sha256sum"] for ds in metadata
    }

    ds_with_frames = [ds for ds in metadata if ds["frame"]["file_name"]]
    frames_registry = {
        "frames" + ds["frame"]["file_name"]: ds["frame"]["sha256sum"]
        for ds in ds_with_frames
    }

    ds_with_videos = [ds for ds in metadata if ds["video"]["file_name"]]
    videos_registry = {
        "videos" + ds["video"]["file_name"]: ds["video"]["sha256sum"]
        for ds in ds_with_videos
    }
    return {**poses_registry, **frames_registry, **videos_registry}


metadata = _fetch_metadata("metadata.yaml")
file_registry = _generate_file_registry(metadata)

# Create a download manager for the pose data
SAMPLE_DATA = pooch.create(
    path=DATA_DIR,
    base_url=f"{DATA_URL}/",
    retry_if_failed=0,
    registry=file_registry,
)


def list_sample_data() -> list[str]:
    """Find available sample pose data in the *movement* data repository.

    Returns
    -------
    filenames : list of str
    List of filenames for available pose data.

    """
    return [
        f.split("/")[-1]
        for f in SAMPLE_DATA.registry
        if f.startswith("poses/")
    ]


def fetch_sample_data_path(filename: str) -> Path:
    """Download sample pose data and return its local filepath.

    The data are downloaded from the *movement* data repository to the user's
    local machine upon first use and are stored in a local cache directory.
    The function returns the path to the downloaded file,
    not the contents of the file itself.

    Parameters
    ----------
    filename : str
        Name of the file to fetch.

    Returns
    -------
    path : pathlib.Path
        Path to the downloaded file.

    """
    try:
        return Path(SAMPLE_DATA.fetch(f"poses/{filename}", progressbar=True))
    except ValueError as error:
        raise log_error(
            ValueError,
            f"File '{filename}' is not in the registry. Valid "
            f"filenames are: {list_sample_data()}",
        ) from error


def fetch_sample_data(
    filename: str,
) -> xarray.Dataset:
    """Download sample pose data and load it as an xarray Dataset.

    The data are downloaded from the *movement* data repository to the user's
    local machine upon first use and are stored in a local cache directory.
    This function returns the pose data as an xarray Dataset.

    Parameters
    ----------
    filename : str
        Name of the file to fetch.

    Returns
    -------
    ds : xarray.Dataset
        Pose data contained in the fetched sample file.

    """
    file_path = fetch_sample_data_path(filename)
    file_metadata = next(
        file for file in metadata if file["file_name"] == filename
    )

    ds = load_poses.from_file(
        file_path,
        source_software=file_metadata["source_software"],
        fps=file_metadata["fps"],
    )
    return ds
