"""Module for fetching and loading sample datasets.

This module provides functions for fetching and loading sample data used in
tests, examples, and tutorials. The data are stored in a remote repository
on GIN and are downloaded to the user's local machine the first time they
are used.
"""

from pathlib import Path

import pooch
import requests.exceptions
import xarray
import yaml

from movement.io import load_poses

# URL to the remote data repository on GIN
# noinspection PyInterpreter
DATA_URL = (
    "https://gin.g-node.org/neuroinformatics/movement-test-data/raw/master"
)

# Save data in ¬/.movement/data
DATA_DIR = Path("~", ".movement", "data").expanduser()
# Create the folder if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Try to fetch the newest sample metadata
def fetch_metadata(file_name):
    temp_file = f"temp_{file_name}"

    if Path(DATA_DIR / file_name).is_file():
        Path.replace(DATA_DIR / file_name, DATA_DIR / temp_file)

    metadata_pooch = pooch.create(
        path=DATA_DIR,
        base_url=f"{DATA_URL}",
        registry={file_name: None},
    )

    try:
        md_path = Path(metadata_pooch.fetch(file_name, progressbar=True))
    except requests.exceptions.ConnectionError:
        if Path(DATA_DIR / temp_file).is_file():
            Path.rename(DATA_DIR / temp_file, DATA_DIR / file_name)

        raise Exception(
            "An error occurred while downloading the sample metadata file."
        )

    return md_path


metadata_path = fetch_metadata("poses_files_metadata.yaml")

with open(metadata_path, "r") as metadata_file:
    metadata = yaml.safe_load(metadata_file)

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
    return Path(SAMPLE_DATA.fetch(filename, progressbar=True))


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

    if file_metadata["source_software"] == "SLEAP":
        ds = load_poses.from_sleap_file(file_path, fps=file_metadata["fps"])
    elif file_metadata["source_software"] == "DeepLabCut":
        ds = load_poses.from_dlc_file(file_path, fps=file_metadata["fps"])
    elif file_metadata["source_software"] == "LightningPose":
        ds = load_poses.from_lp_file(file_path, fps=file_metadata["fps"])
    return ds
