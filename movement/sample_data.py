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
    """Download the yaml file containing sample metadata from the *movement*
    data repository and save it in the specified directory with a temporary
    filename - temp_{file_name} - to avoid overwriting any existing files.

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
    """Download the yaml file containing metadata from the *movement* sample
    data repository and load it as a list of dictionaries.

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

    with open(local_file_path, "r") as metadata_file:
        metadata = yaml.safe_load(metadata_file)
    return metadata


metadata = _fetch_metadata("poses_files_metadata.yaml")

# Create a download manager for the pose data
SAMPLE_DATA = pooch.create(
    path=DATA_DIR / "poses",
    base_url=f"{DATA_URL}/poses/",
    retry_if_failed=0,
    registry={file["file_name"]: file["sha256sum"] for file in metadata},
)


def list_sample_data() -> list[str]:
    """Find available sample pose data in the *movement* data repository.

    Returns
    -------
    filenames : list of str
        List of filenames for available pose data."""
    return list(SAMPLE_DATA.registry.keys())


def fetch_sample_data_path(filename: str) -> Path:
    """Download sample pose data from the *movement* data repository and return
    its local filepath.

    The data are downloaded to the user's local machine the first time they are
    used and are stored in a local cache directory. The function returns the
    path to the downloaded file, not the contents of the file itself.

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
        return Path(SAMPLE_DATA.fetch(filename, progressbar=True))
    except ValueError:
        raise log_error(
            ValueError,
            f"File '{filename}' is not in the registry. Valid "
            f"filenames are: {list_sample_data()}",
        )


def fetch_sample_data(
    filename: str,
) -> xarray.Dataset:
    """Download and return sample pose data from the *movement* data
    repository.

    The data are downloaded to the user's local machine the first time they are
    used and are stored in a local cache directory. Returns sample pose data as
    an xarray Dataset.

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
