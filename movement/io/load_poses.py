import logging
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd

from movement.io.validators import DeepLabCutPosesFile, SleapAnalysisFile

# get logger
logger = logging.getLogger(__name__)


def from_dlc(file_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """Load pose estimation results from a DeepLabCut (DLC) files.
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
    dlc_poses_file = DeepLabCutPosesFile(path=file_path)  # type: ignore
    file_suffix = dlc_poses_file.path.suffix

    # Load the DLC poses
    try:
        if file_suffix == ".csv":
            df = _parse_dlc_csv_to_dataframe(dlc_poses_file.path)
        else:  # file can only be .h5 at this point
            df = pd.read_hdf(dlc_poses_file.path)
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


def _parse_dlc_csv_to_dataframe(file_path: Path) -> pd.DataFrame:
    """If poses are loaded from a DeepLabCut.csv file, the resulting DataFrame
    lacks the multi-index columns that are present in the .h5 file. This
    function parses the csv file to a DataFrame with multi-index columns.

    Parameters
    ----------
    file_path : pathlib Path
        Path to the file containing the DLC poses, in .csv format.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the DLC poses, with multi-index columns.
    """

    possible_level_names = ["scorer", "bodyparts", "coords", "individual"]
    with open(file_path, "r") as f:
        # if line starts with a possible level name, split it into a list
        # of strings, and add it to the list of header lines
        header_lines = [
            line.strip().split(",")
            for line in f.readlines()
            if line.split(",")[0] in possible_level_names
        ]

    # Form multi-index column names from the header lines
    level_names = [line[0] for line in header_lines]
    column_tuples = list(zip(*[line[1:] for line in header_lines]))
    columns = pd.MultiIndex.from_tuples(column_tuples, names=level_names)

    # Import the DLC poses as a DataFrame
    df = pd.read_csv(
        file_path, skiprows=len(header_lines), index_col=0, names=columns
    )
    df.columns.rename(level_names, inplace=True)
    return df


def from_sleap(file_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """Load pose estimation results from a SLEAP analysis file.

    Parameters
    ----------
    file_path : pathlib Path or str
        Path to the SLEAP analysis file (see Notes).

    Returns
    -------
    pandas DataFrame
        DataFrame containing the SLEAP poses

    Notes
    -----
    SLEAP analysis files have the suffix ".h5", not ".slp".
    The SLEAP inference procedure normally produces a file containing the
    predictions, e.g. "myproject.predictions.slp". This can be converted
    to an analysis file using the command line tool `sleap-convert` and
    using the "--format analysis" option. Alternatively, choose
    “Export Analysis HDF5…” from the “File” menu of the SLEAP GUI [1]_.

    movement expects the tracks to be proofread before loading them.
    There should be as many tracks as there are instances (animals) in the
    video, without identity switches. Follow the Sleap guide for
    tracking and proofreading [2]_.

    References
    ----------
    .. [1] https://sleap.ai/tutorials/analysis.html
    .. [2] https://sleap.ai/guides/proofreading.html
    """

    # Validate the input file
    sleap_poses_file = SleapAnalysisFile(path=file_path)  # type: ignore

    # Load the SLEAP poses
    with h5py.File(sleap_poses_file.path, "r") as f:
        # Initialise a dict to hold the data
        # First, load the tracks
        poses = {"tracks": f["tracks"][:].T}
        n_frames, n_nodes, n_dims, n_tracks = poses["tracks"].shape
        # Load the track occupancy matrix
        poses["track_occupancy"] = f["track_occupancy"][:].astype(bool)
        # Read the names of the nodes (body parts) and the tracks
        poses["node_names"] = [n.decode() for n in f["node_names"][:]]
        poses["track_names"] = [n.decode() for n in f["track_names"][:]]

        # If present, read the point scores into an array
        # of shape = (n_frames, n_nodes, n_tracks)
        if "point_scores" in f.keys():
            poses["point_scores"] = f["point_scores"][:].T
        # else, create an array of NaNs
        else:
            poses["point_scores"] = np.nans(
                (n_frames, n_nodes, n_tracks), dtype=np.float32
            )

        # Describe the values in the data dictionary we just created.
        for key, value in poses.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: {value.dtype} array of shape {value.shape}")
            else:
                print(f"{key}: {value}")

        # Build a DLC-style multi-index dataframe
        # SLEAP doesn't have the concept of "scorer",
        # so we use the filename as the scorer name
        df = pd.DataFrame(
            poses["tracks"].reshape(n_frames, n_tracks * n_nodes * n_dims),
            columns=pd.MultiIndex.from_product(
                [
                    [sleap_poses_file.path.stem],
                    poses["track_names"],
                    poses["node_names"],
                    ["x", "y"],
                ],
                names=["scorer", "individual", "bodyparts", "coords"],
            ),
            index=pd.Index(range(n_frames)),
        )

        logger.info(f"Loaded poses from {file_path}")
        # Save the dataframe as csv
        df.to_csv(sleap_poses_file.path.parent / "sleap_poses.csv")
        return df


if __name__ == "__main__":
    from movement.datasets import fetch_pose_data_path

    dlc_file = fetch_pose_data_path("DLC_single-mouse_EPM.predictions.h5")
    sleap_file = fetch_pose_data_path(
        "SLEAP_two-mice_social-interaction.analysis.h5"
    )

    dlc_df = from_dlc(dlc_file)
    sleap_df = from_sleap(sleap_file)
