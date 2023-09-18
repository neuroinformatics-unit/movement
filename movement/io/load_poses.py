import logging
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sleap_io.io.slp import read_labels

from movement.io.poses_accessor import PosesAccessor
from movement.io.validators import (
    ValidFile,
    ValidHDF5,
    ValidPosesCSV,
    ValidPoseTracks,
)
from movement.logging import log_error

logger = logging.getLogger(__name__)


def from_dlc_df(df: pd.DataFrame, fps: Optional[float] = None) -> xr.Dataset:
    """Create an xarray.Dataset from a DeepLabCut-style pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the pose tracks and confidence scores. Must
        be formatted as in DeepLabCut output files (see Notes).
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        Dataset containing the pose tracks, confidence scores, and metadata.

    Notes
    -----
    The DataFrame must have a multi-index column with the following levels:
    "scorer", ("individuals"), "bodyparts", "coords". The "individuals"
    level may be omitted if there is only one individual in the video.
    The "coords" level contains the spatial coordinates "x", "y",
    as well as "likelihood" (point-wise confidence scores).
    The row index corresponds to the frame number.

    See Also
    --------
    movement.io.load_poses.from_dlc_file : Load pose tracks directly from file.
    """

    # read names of individuals and keypoints from the DataFrame
    if "individuals" in df.columns.names:
        individual_names = (
            df.columns.get_level_values("individuals").unique().to_list()
        )
    else:
        individual_names = ["individual_0"]

    keypoint_names = (
        df.columns.get_level_values("bodyparts").unique().to_list()
    )

    # reshape the data into (n_frames, n_individuals, n_keypoints, 3)
    # where the last axis contains "x", "y", "likelihood"
    tracks_with_scores = df.to_numpy().reshape(
        (-1, len(individual_names), len(keypoint_names), 3)
    )

    valid_data = ValidPoseTracks(
        tracks_array=tracks_with_scores[:, :, :, :-1],
        scores_array=tracks_with_scores[:, :, :, -1],
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
    )
    return _from_valid_data(valid_data)


def from_sleap_file(
    file_path: Union[Path, str], fps: Optional[float] = None
) -> xr.Dataset:
    """Load pose tracking data from a SLEAP file into an xarray Dataset.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the SLEAP predictions in ".h5"
        (analysis) format. Alternatively, an ".slp" (labels) file can
        also be supplied (but this feature is experimental, see Notes).
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        Dataset containing the pose tracks, confidence scores, and metadata.

    Notes
    -----
    The SLEAP predictions are normally saved in ".slp" files, e.g.
    "v1.predictions.slp". An analysis file, suffixed with ".h5" can be exported
    from the ".slp" file, using either the command line tool `sleap-convert`
    (with the "--format analysis" option enabled) or the SLEAP GUI (Choose
    "Export Analysis HDF5…" from the "File" menu) [1]_. This is the
    preferred format for loading pose tracks from SLEAP into *movement*.

    You can also try directly loading the ".slp" file, but this feature is
    experimental and doesnot work in all cases. If the ".slp" file contains
    both user-labeled and predicted instances, only the predicted ones will be
    loaded. If there are multiple videos in the file, only the first one will
    be used.

    *movement* expects the tracks to be assigned and proofread before loading
    them, meaning each track is interpreted as a single individual/animal.
    Follow the SLEAP guide for tracking and proofreading [2]_.

    References
    ----------
    .. [1] https://sleap.ai/tutorials/analysis.html
    .. [2] https://sleap.ai/guides/proofreading.html

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_sleap_file("path/to/file.analysis.h5", fps=30)
    """

    file = ValidFile(
        file_path,
        expected_permission="r",
        expected_suffix=[".h5", ".slp"],
    )

    # Load and validate data
    if file.path.suffix == ".h5":
        valid_data = _load_from_sleap_analysis_file(file.path, fps=fps)
    else:  # file.path.suffix == ".slp"
        valid_data = _load_from_sleap_labels_file(file.path, fps=fps)
    logger.debug(f"Validated pose tracks from {file.path}.")

    # Initialize an xarray dataset from the dictionary
    ds = _from_valid_data(valid_data)

    # Add metadata as attrs
    ds.attrs["source_software"] = "SLEAP"
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded pose tracks from {file.path}:")
    logger.info(ds)
    return ds


def from_dlc_file(
    file_path: Union[Path, str], fps: Optional[float] = None
) -> xr.Dataset:
    """Load pose tracking data from a DeepLabCut (DLC) output file
    into an xarray Dataset.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the DLC predicted poses, either in ".h5"
        or ".csv" format.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        Dataset containing the pose tracks, confidence scores, and metadata.

    See Also
    --------
    movement.io.load_poses.from_dlc_df : Load pose tracks from a DataFrame.

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_dlc_file("path/to/file.h5", fps=30)
    """

    file = ValidFile(
        file_path,
        expected_permission="r",
        expected_suffix=[".csv", ".h5"],
    )

    # Load the DLC poses into a DataFrame
    if file.path.suffix == ".csv":
        df = _parse_dlc_csv_to_df(file.path)
    else:  # file.path.suffix == ".h5"
        df = _load_df_from_dlc_h5(file.path)

    logger.debug(f"Loaded poses from {file.path} into a DataFrame.")
    # Convert the DataFrame to an xarray dataset
    ds = from_dlc_df(df=df, fps=fps)

    # Add metadata as attrs
    ds.attrs["source_software"] = "DeepLabCut"
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded pose tracks from {file.path}:")
    logger.info(ds)
    return ds


def _load_from_sleap_analysis_file(
    file_path: Path, fps: Optional[float]
) -> ValidPoseTracks:
    """Load and validate pose tracks and confidence scores from a SLEAP
    analysis file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the SLEAP analysis file containing predicted pose tracks.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame units.

    Returns
    -------
    movement.io.tracks_validators.ValidPoseTracks
        The validated pose tracks and confidence scores.
    """

    file = ValidHDF5(file_path, expected_datasets=["tracks"])

    with h5py.File(file.path, "r") as f:
        # transpose to shape: (n_frames, n_tracks, n_keypoints, n_space)
        tracks = f["tracks"][:].transpose((3, 0, 2, 1))
        # Create an array of NaNs for the confidence scores
        scores = np.full(tracks.shape[:-1], np.nan, dtype="float32")
        # If present, read the point-wise scores,
        # and transpose to shape: (n_frames, n_tracks, n_keypoints)
        if "point_scores" in f.keys():
            scores = f["point_scores"][:].transpose((2, 0, 1))

        return ValidPoseTracks(
            tracks_array=tracks,
            scores_array=scores,
            individual_names=[n.decode() for n in f["track_names"][:]],
            keypoint_names=[n.decode() for n in f["node_names"][:]],
            fps=fps,
        )


def _load_from_sleap_labels_file(
    file_path: Path, fps: Optional[float]
) -> ValidPoseTracks:
    """Load and validate pose tracks and confidence scores from a SLEAP
    labels file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the SLEAP labels file containing predicted pose tracks.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame units.

    Returns
    -------
    movement.io.tracks_validators.ValidPoseTracks
        The validated pose tracks and confidence scores.
    """

    file = ValidHDF5(file_path, expected_datasets=["pred_points", "metadata"])

    labels = read_labels(file.path.as_posix())
    tracks_with_scores = labels.numpy(untracked=False, return_confidence=True)

    return ValidPoseTracks(
        tracks_array=tracks_with_scores[:, :, :, :-1],
        scores_array=tracks_with_scores[:, :, :, -1],
        individual_names=[track.name for track in labels.tracks],
        keypoint_names=[kp.name for kp in labels.skeletons[0].nodes],
        fps=fps,
    )


def _parse_dlc_csv_to_df(file_path: Path) -> pd.DataFrame:
    """If poses are loaded from a DeepLabCut .csv file, the DataFrame
    lacks the multi-index columns that are present in the .h5 file. This
    function parses the csv file to a pandas DataFrame with multi-index
    columns, i.e. the same format as in the .h5 file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the DeepLabCut-style CSV file.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style DataFrame with multi-index columns.
    """

    file = ValidPosesCSV(file_path)

    possible_level_names = ["scorer", "individuals", "bodyparts", "coords"]
    with open(file.path, "r") as f:
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
        file.path,
        skiprows=len(header_lines),
        index_col=0,
        names=np.array(columns),
    )
    df.columns.rename(level_names, inplace=True)
    return df


def _load_df_from_dlc_h5(file_path: Path) -> pd.DataFrame:
    """Load pose tracks and likelihood scores from a DeepLabCut .h5 file
    into a pandas DataFrame.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the DeepLabCut-style HDF5 file containing pose tracks.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style Dataframe.
    """

    file = ValidHDF5(file_path, expected_datasets=["df_with_missing"])

    try:
        # pd.read_hdf does not always return a DataFrame
        df = pd.DataFrame(pd.read_hdf(file.path, key="df_with_missing"))
    except Exception as error:
        raise log_error(error, f"Could not load a dataframe from {file.path}.")
    return df


def _from_valid_data(data: ValidPoseTracks) -> xr.Dataset:
    """Convert already validated pose tracking data to an xarray Dataset.

    Parameters
    ----------
    data : movement.io.tracks_validators.ValidPoseTracks
        The validated data object.

    Returns
    -------
    xarray.Dataset
        Dataset containing the pose tracks, confidence scores, and metadata.
    """

    n_frames = data.tracks_array.shape[0]
    n_space = data.tracks_array.shape[-1]

    # Create the time coordinate, depending on the value of fps
    time_coords = np.arange(n_frames, dtype=int)
    time_unit = "frames"
    if data.fps is not None:
        time_coords = time_coords / data.fps
        time_unit = "seconds"

    DIM_NAMES = PosesAccessor.dim_names
    # Convert data to an xarray.Dataset
    return xr.Dataset(
        data_vars={
            "pose_tracks": xr.DataArray(data.tracks_array, dims=DIM_NAMES),
            "confidence": xr.DataArray(data.scores_array, dims=DIM_NAMES[:-1]),
        },
        coords={
            DIM_NAMES[0]: time_coords,
            DIM_NAMES[1]: data.individual_names,
            DIM_NAMES[2]: data.keypoint_names,
            DIM_NAMES[3]: ["x", "y", "z"][:n_space],
        },
        attrs={
            "fps": data.fps,
            "time_unit": time_unit,
            "source_software": None,
            "source_file": None,
        },
    )
