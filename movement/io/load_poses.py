import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from movement.io.validators import DeeplabcutPosesFile
from movement.log_config import configure_logging

# initialize logger
configure_logging()
logger = logging.getLogger(__name__)

# TODO:
#  - store poses in a custom Trajectory class instead of DataFrame


def from_dlc(file_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """Load pose estimation results from a Deeplabcut (DLC) files.
    Files must be in .h5 format or .csv format.

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
    >>> poses = load_poses.from_dlc("path/to/file.h5")
    """

    # Validate the input file path
    dlc_poses_file = DeeplabcutPosesFile(file_path=file_path)  # type: ignore
    file_suffix = dlc_poses_file.file_path.suffix

    # Load the DLC poses
    try:
        if file_suffix == ".csv":
            df = pd.read_csv(dlc_poses_file.file_path)
        else:  # file can only be .h5 at this point
            df = pd.read_hdf(dlc_poses_file.file_path)
            # above line does not necessarily return a DataFrame
            df = pd.DataFrame(df)
    except (OSError, TypeError, ValueError) as e:
        error_msg = (
            f"Could not load poses from {file_path}. "
            "Please check that the file is valid and readable."
        )
        logger.error(error_msg)
        raise OSError from e
    logger.info(f"Loaded poses from {file_path}")
    return df
