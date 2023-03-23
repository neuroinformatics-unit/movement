import warnings
from pathlib import Path
from typing import Optional, Union

import h5py
import pandas as pd

from movement.utils import validate_dataframe, validate_file_path

# TODO:
#  - store tracks in a custom Trajectory class instead of DataFrame
#  - add support for other file formats (e.g. .csv)


def from_dlc_h5(
    filepath: Union[Path, str], key: Optional[str] = "df_with_missing"
) -> Optional[pd.DataFrame]:
    """Load pose estimation results from a Deeplabcut (DLC) HDF5 (.h5) file.

    Parameters
    ----------
    filepath : pathlib Path or str
        Path to the file containing the DLC tracks.
    key : str or None, optional
        Key to the dataframe in the HDF5 file.
        The default is "df_with_missing".
        If the key is not found, or if it is set to None,
        the function will try to find and load a single dataframe
        in the file.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the DLC tracks

    Examples
    --------
    >>> from movement.io import load_poses
    >>> tracks = load_poses.from_dlc_h5("path/to/file.h5")
    """

    filepath = validate_file_path(filepath, suffix=".h5")

    # Ensure key is a string or None
    if key is not None:
        if not isinstance(key, str):
            raise TypeError(
                f"Key must be a string or None. "
                f"Received {type(key)} instead."
            )

    # Check if the key is in the file
    if key is not None:
        with h5py.File(filepath, "r") as h5_file:
            if key not in h5_file:
                warnings.warn(
                    f"Key '{key}' not found in file: {filepath}. "
                    "Will try to find a single dataframe in the file."
                )
                key = None

    # Load the DLC tracks
    try:
        df = pd.read_hdf(filepath, key=key)
    except (OSError, TypeError, ValueError):
        raise OSError(
            f"Could not read from {filepath}. "
            "Please check that the file is a valid, readable "
            "HDF5 file containing a dataframe."
            ""
        )

    return validate_dataframe(df)
