import logging
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
from sleap_io.io.slp import read_labels

from movement.io.validators import DeepLabCutPosesFile

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
    dlc_poses_file = DeepLabCutPosesFile(file_path=file_path)  # type: ignore
    file_suffix = dlc_poses_file.file_path.suffix

    # Load the DLC poses
    try:
        if file_suffix == ".csv":
            df = _parse_dlc_csv_to_dataframe(dlc_poses_file.file_path)
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

    possible_level_names = ["scorer", "individuals", "bodyparts", "coords"]
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


def from_sleap(file_path: Union[Path, str]) -> dict:
    """Load pose tracking data from a SLEAP labels file.

    Parameters
    ----------
    file_path : pathlib Path or str
        Path to the file containing the SLEAP predictions, either in ".slp"
        or ".h5" (analysis) format. See Notes for more information.

    Returns
    -------
    dict
        Dictionary containing `pose_tracks`, `node_names` and `track_names`.
        - `pose_tracks` is an array containing the predicted poses.
        Shape: (n_frames, n_tracks, n_nodes, n_dims). The last axis
        contains the spatial coordinates "x" and "y", as well as the
        point-wise confidence values.
        - `node_names` is a list of the node names.
        - `track_names` is a list of the track names.

    Notes
    -----
    The SLEAP inference procedure normally produces a file suffixed with ".slp"
    containing the predictions, e.g. "myproject.predictions.slp".
    This can be converted to an ".h5" (analysis) file using the command line
    tool `sleap-convert` with the "--format analysis" option enabled,
    or alternatively by choosing “Export Analysis HDF5…” from the “File” menu
    of the SLEAP GUI [1]_.

    This function will only the predicted instances in the ".slp" file,
    not the user-labeled ones.

    movement expects the tracks to be proofread before loading them.
    There should be as many tracks as there are instances (animals) in the
    video, without identity switches. Follow the SLEAP guide for
    tracking and proofreading [2]_.

    References
    ----------
    .. [1] https://sleap.ai/tutorials/analysis.html
    .. [2] https://sleap.ai/guides/proofreading.html

    Examples
    --------
    >>> from movement.io import load_poses
    >>> poses = load_poses.from_sleap("path/to/labels.predictions.slp")
    """

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if file_path.suffix == ".h5":
        # Load the SLEAP predictions from an analysis file
        poses = _load_sleap_analysis_file(file_path)
    elif file_path.suffix == ".slp":
        # Load the SLEAP predictions from a labels file
        poses = _load_sleap_labels_file(file_path)
    else:
        error_msg = (
            f"Expected file suffix to be '.h5' or '.slp', "
            f"but got '{file_path.suffix}'. Make sure the file is "
            "a SLEAP labels file with suffix '.slp' or SLEAP analysis "
            "file with suffix '.h5'."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    n_frames, n_tracks, n_nodes, n_dims = poses["tracks"].shape
    logger.info(f"Loaded poses from {file_path}.")
    logger.debug(
        f"Shape: ({n_frames} frames, {n_tracks} tracks, "
        f"{n_nodes} nodes, {n_dims - 1} spatial coords "
        "+ 1 confidence score)"
    )
    logger.info(f"Track names: {poses['track_names']}")
    logger.info(f"Node names: {poses['node_names']}")
    return poses


def _load_sleap_analysis_file(file_path: Path) -> dict:
    """Load pose tracking data from a SLEAP analysis file.

    Parameters
    ----------
    file_path : pathlib Path
        Path to the file containing the SLEAP predictions, in ".h5" format.

    Returns
    -------
    dict
        Dictionary containing `pose_tracks`, `node_names` and `track_names`.
    """

    # Load the SLEAP poses
    with h5py.File(file_path, "r") as f:
        # First, load and reshape the pose tracks
        tracks = f["tracks"][:].T
        n_frames, n_nodes, n_dims, n_tracks = tracks.shape
        tracks = tracks.reshape((n_frames, n_tracks, n_nodes, n_dims))

        # If present, read the point-wise confidence scores
        # and add them to the "tracks" array
        confidence = np.full(
            (n_frames, n_tracks, n_nodes, 3), np.nan, dtype="float32"
        )
        if "point_scores" in f.keys():
            confidence = f["point_scores"][:].T
            confidence = confidence.reshape((n_frames, n_tracks, n_nodes))
        tracks = np.concatenate(
            [tracks, confidence[:, :, :, np.newaxis]], axis=3
        )

        # Create the dictionary to be returned
        poses = {
            "tracks": tracks,
            "node_names": [n.decode() for n in f["node_names"][:]],
            "track_names": [n.decode() for n in f["track_names"][:]],
        }
    return poses


def _load_sleap_labels_file(file_path: Path) -> dict:
    """Load pose tracking data from a SLEAP labels file.

    Parameters
    ----------
    file_path : pathlib Path
        Path to the file containing the SLEAP predictions, in ".slp" format.

    Returns
    -------
    dict
        Dictionary containing `pose_tracks`, `node_names` and `track_names`.
    """
    labels = read_labels(file_path.as_posix())
    poses = {
        "tracks": labels.numpy(return_confidence=True),
        "node_names": [node.name for node in labels.skeletons[0].nodes],
        "track_names": [track.name for track in labels.tracks],
    }
    # return_confidence=True adds the point-wise confidence scores
    # as an extra coord dimension to the "tracks" array

    return poses
