"""Module for fetching and loading datasets.

This module provides functions for fetching and loading data used in tests,
examples, and tutorials. The data are stored in a remote repository on GIN
and are downloaded to the user's local machine the first time they are used.
"""

from pathlib import Path

import pooch

# URL to the remote data repository on GIN
# noinspection PyInterpreter
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
        "DLC_two-mice.predictions.csv": "6f152cf7ce1ea2a3099384fa1655abde8963f5153a02f0ca643e3e8e97f63cbf",  # noqa: E501
        "SLEAP_single-mouse_EPM.analysis.h5": "0df0a09c2493a1d9964ba98cbf751eda62743f1d688ae82b6df7b0f77169ed47",  # noqa: E501
        "SLEAP_single-mouse_EPM.predictions.slp": "ca620db6123635761ddf69947f72f653d14a59137b355bd2d8f7c2f1be67e474",  # noqa: E501
        "SLEAP_two-mice_social-interaction.analysis.h5": "f7f1e59d4b2c34712089f8aaf2390272291d93e6991c1abe32d9ce798a6234f9",  # noqa: E501
        "SLEAP_two-mice_social-interaction.predictions.slp": "45881affde9704c045e70b8d4b3f6bbb8d9bd8ef9f4cdea6d173cfe35857549b",  # noqa: E501
        "SLEAP_three-mice_Aeon_proofread.analysis.h5": "82ebd281c406a61536092863bc51d1a5c7c10316275119f7daf01c1ff33eac2a",  # noqa: E501
        "SLEAP_three-mice_Aeon_proofread.predictions.slp": "7b7436a52dfd5f4d80d7c66919ad1a1732e5435fe33faf9011ec5f7b7074e788",  # noqa: E501
    },
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
