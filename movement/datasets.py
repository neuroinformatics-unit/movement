"""Module for fetching and loading datasets.

This module provides functions for fetching and loading data used in tests,
examples, and tutorials. The data are stored in a remote repository on GIN
and are downloaded to the user's local machine the first time they are used.
"""

from pathlib import Path

import pooch
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

# Fetch newest sample metadata
Path.unlink(DATA_DIR / "poses_files_metadata.yaml", missing_ok=True)
METADATA_POOCH = pooch.create(
    path=DATA_DIR,
    base_url=f"{DATA_URL}",
    registry={"poses_files_metadata.yaml": None},
)
METADATA_PATH = Path(
    METADATA_POOCH.fetch("poses_files_metadata.yaml", progressbar=True)
)

with open(METADATA_PATH, "r") as sample_info:
    metadata = yaml.safe_load(sample_info)

sample_registry = {file["file_name"]: file["sha256sum"] for file in metadata}

# Create a download manager for the pose data
POSE_DATA = pooch.create(
    path=DATA_DIR / "poses",
    base_url=f"{DATA_URL}/poses/",
    retry_if_failed=0,
    registry=sample_registry,
)


def list_pose_data() -> list[str]:
    """Find available sample pose data in the *movement* data repository.

    Returns
    -------
    filenames : list of str
        List of filenames for available pose data."""
    return list(POSE_DATA.registry.keys())


def fetch_pose_data_path(filename: str) -> Path:
    """Fetch sample pose data from the *movement* data repository.

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
    return Path(POSE_DATA.fetch(filename, progressbar=True))


def fetch_pose_data(filename: str) -> xarray.Dataset:
    """Fetch sample pose data from the *movement* data repository.

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

    file_path = fetch_pose_data_path(filename)
    if filename.startswith("SLEAP"):
        ds = load_poses.from_sleap_file(file_path)
    elif filename.startswith("DLC"):
        ds = load_poses.from_dlc_file(file_path)
    return ds
