from pathlib import Path
from typing import Optional, Union

import pandas as pd

from movement.io.validators import DeeplabcutPosesFile

# TODO:
#  - store poses in a custom Trajectory class instead of DataFrame
#  - add support for other file formats (e.g. .csv)


def from_dlc_h5(file_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """Load pose estimation results from a Deeplabcut (DLC) HDF5 (.h5) file.

    Parameters
    ----------
    file_path : pathlib Path or str
        Path to the file containing the DLC poses.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the DLC poses

    Examples
    --------
    >>> from movement.io import load_poses
    >>> poses = load_poses.from_dlc_h5("path/to/file.h5")
    """

    # Validate the input data
    dlc_poses_file = DeeplabcutPosesFile(file_path=file_path)

    # Load the DLC poses
    try:
        df = pd.read_hdf(dlc_poses_file.file_path)
        df = pd.DataFrame(df)
    except (OSError, TypeError, ValueError):
        raise OSError(
            f"Could not read from {file_path}. "
            "Please check that the file is a valid, readable "
            "HDF5 file containing a dataframe."
            ""
        )
    return df
