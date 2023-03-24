from pathlib import Path
from typing import Optional, Union

import pandas as pd

from movement.utils import validate_dataframe, validate_file_path

# TODO:
#  - store poses in a custom Trajectory class instead of DataFrame
#  - add support for other file formats (e.g. .csv)


def from_dlc_h5(
    filepath: Union[Path, str], key: Optional[str] = "df_with_missing"
) -> Optional[pd.DataFrame]:
    """Load pose estimation results from a Deeplabcut (DLC) HDF5 (.h5) file.

    Parameters
    ----------
    filepath : pathlib Path or str
        Path to the file containing the DLC poses.
    key : str or None, optional
        Key to the dataframe in the HDF5 file.
        The default is "df_with_missing".
        If the key is not found, or if it is set to None,
        the function will try to find and load a single dataframe
        in the file.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the DLC poses

    Examples
    --------
    >>> from movement.io import load_poses
    >>> poses = load_poses.from_dlc_h5("path/to/file.h5")
    """

    filepath = validate_file_path(filepath, suffix=".h5")

    # Ensure key is a string or None
    if key is not None:
        if not isinstance(key, str):
            raise TypeError(
                f"Key must be a string or None. "
                f"Received {type(key)} instead."
            )

    # Load the DLC poses
    try:
        df = pd.read_hdf(filepath, key=key)
    except (OSError, TypeError, ValueError):
        raise OSError(
            f"Could not read from {filepath}. "
            "Please check that the file is a valid, readable "
            "HDF5 file containing a dataframe."
            ""
        )

    dlc_poses_df = validate_dataframe(df)
    return dlc_poses_df
