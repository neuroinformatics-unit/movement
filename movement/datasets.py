"""Module for fetching and loading datasets.

This module provides functions for fetching and loading data used in tests,
examples, and tutorials. The data are stored in a remote repository on GIN
and are downloaded to the user's local machine the first time they are used.
"""

from pathlib import Path

import pooch

# URL to the remote data repository on GIN
DATA_URL = (
    "https://gin.g-node.org/neuroinformatics/movement-test-data/raw/master"
)

# Save data in ¬/.movement/data
DATA_DIR = Path("~", ".movement", "data").expanduser()
# Create the folder if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create a download manager for the pose data
POSE_DATA = pooch.create(
    path=DATA_DIR / "poses",
    base_url=f"{DATA_URL}/poses/",
    retry_if_failed=0,
    registry={
        "DLC_single-wasp.predictions.h5": "931dddb6ef5e08db6054d3757a441ee31dd0d9ff5a10802ad8405d6c4e7e274e",  # noqa: E501
        "DLC_single-wasp.predictions.csv": "9b194cc930c2e2e0d33c816320d029f889b306d53ff2fe95ff408e99c9cdea23",  # noqa: E501
        "DLC_single-mouse_EPM.predictions.h5": "0ddc2b08c9401435929783b22ea31b3673ceb80c3a02c5f3531bb1cfd78deea5",  # noqa: E501
        "SLEAP_single-mouse_EPM.predictions.slp": "6ede1837dc8a66e5615ff1dc8d2d590727fa78e9b4d9391bad9e9ef0922445fe",  # noqa: E501
        "SLEAP_two-mice_social-interaction.predictions.slp": "45881affde9704c045e70b8d4b3f6bbb8d9bd8ef9f4cdea6d173cfe35857549b",  # noqa: E501
    },
)


def fetch_pose_data(filename: str) -> Path:
    """Fetch sample pose data from the remote repository.

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
