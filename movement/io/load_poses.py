"""Functions for loading pose tracking data from various frameworks."""

import ast
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sleap_io.io.slp import read_labels
from sleap_io.model.labels import Labels

from movement import MovementDataset
from movement.utils.logging import log_error, log_warning
from movement.validators.datasets import ValidBboxesDataset, ValidPosesDataset
from movement.validators.files import (
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
    ValidVIAtracksCSV,
)

logger = logging.getLogger(__name__)


def poses_from_numpy(
    position_array: np.ndarray,
    confidence_array: np.ndarray | None = None,
    individual_names: list[str] | None = None,
    keypoint_names: list[str] | None = None,
    fps: float | None = None,
    source_software: str | None = None,
) -> xr.Dataset:
    """Create a ``movement`` pose dataset from NumPy arrays.

    Parameters
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_keypoints, n_space)
        containing the poses. It will be converted to a
        :py:class:`xarray.DataArray` object named "position".
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals, n_keypoints) containing
        the point-wise confidence scores. It will be converted to a
        :py:class:`xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "individual_0",
        "individual_1", etc.
    keypoint_names : list of str, optional
        List of unique names for the keypoints in the skeleton. If None
        (default), the keypoints will be named "keypoint_0", "keypoint_1",
        etc.
    fps : float, optional
        Frames per second of the video. Defaults to None, in which case
        the time coordinates will be in frame numbers.
    source_software : str, optional
        Name of the pose estimation software from which the data originate.
        Defaults to None.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Examples
    --------
    Create random position data for two individuals, ``Alice`` and ``Bob``,
    with three keypoints each: ``snout``, ``centre``, and ``tail_base``.
    These are tracked in 2D space over 100 frames, at 30 fps.
    The confidence scores are set to 1 for all points.

    >>> import numpy as np
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_numpy(
    ...     position_array=np.random.rand((100, 2, 3, 2)),
    ...     confidence_array=np.ones((100, 2, 3)),
    ...     individual_names=["Alice", "Bob"],
    ...     keypoint_names=["snout", "centre", "tail_base"],
    ...     fps=30,
    ... )

    """
    valid_pose_data = ValidPosesDataset(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software=source_software,
    )
    return _poses_ds_from_valid_data(valid_pose_data)


def bboxes_from_numpy(
    position_array: np.ndarray,
    shape_array: np.ndarray,
    confidence_array: np.ndarray | None = None,
    individual_names: list[str] | None = None,
    fps: float | None = None,
    source_software: str | None = None,
) -> xr.Dataset:
    """Create a ``movement`` bounding boxes dataset from NumPy arrays.

    Parameters
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_space)
        containing the poses. It will be converted to a
        :py:class:`xarray.DataArray` object named "position".
    shape_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_space)
        containing the poses. It will be converted to a
        :py:class:`xarray.DataArray` object named "position".
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals) containing
        the point-wise confidence scores. It will be converted to a
        :py:class:`xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "id_0",
        "id_1", etc.
    fps : float, optional
        Frames per second of the video. Defaults to None, in which case
        the time coordinates will be in frame numbers.
    source_software : str, optional
        Name of the pose estimation software from which the data originate.
        Defaults to None.

    Returns
    -------
    xarray.Dataset
        ``movement`` bounding boxes dataset containing the boxes tracks,
        boxes shapes, confidence scores and associated metadata.

    Examples
    --------
    Create random position data for two individuals, ``Alice`` and ``Bob``,
    with three keypoints each: ``snout``, ``centre``, and ``tail_base``.
    These are tracked in 2D space over 100 frames, at 30 fps.
    The confidence scores are set to 1 for all points.

    >>> import numpy as np
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_numpy(
    ...     position_array=np.random.rand((100, 2, 3, 2)),
    ...     confidence_array=np.ones((100, 2, 3)),
    ...     individual_names=["Alice", "Bob"],
    ...     keypoint_names=["snout", "centre", "tail_base"],
    ...     fps=30,
    ... )

    """
    valid_bboxes_data = ValidBboxesDataset(
        position_array=position_array,
        shape_array=shape_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        fps=fps,
        source_software=source_software,
    )
    return _bboxes_ds_from_valid_data(valid_bboxes_data)


def from_file(
    file_path: Path | str,
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` dataset from any supported file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing predicted poses. The file format must
        be among those supported by the ``from_dlc_file()``,
        ``from_slp_file()`` or ``from_lp_file()`` functions. One of these
        these functions will be called internally, based on
        the value of ``source_software``.
    source_software : "DeepLabCut", "SLEAP", "LightningPose" or "VIA-tracks".
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    See Also
    --------
    movement.io.load_poses.from_dlc_file
    movement.io.load_poses.from_sleap_file
    movement.io.load_poses.from_lp_file
    movement.io.load_poses.from_via_tracks_file

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_file(
    ...     "path/to/file.h5", source_software="DeepLabCut", fps=30
    ... )

    """
    if source_software == "DeepLabCut":
        return from_dlc_file(file_path, fps)
    elif source_software == "SLEAP":
        return from_sleap_file(file_path, fps)
    elif source_software == "LightningPose":
        return from_lp_file(file_path, fps)
    elif source_software == "VIA-tracks":
        return from_via_tracks_file(file_path, fps)
    else:
        raise log_error(
            ValueError, f"Unsupported source software: {source_software}"
        )


def from_dlc_style_df(
    df: pd.DataFrame,
    fps: float | None = None,
    source_software: Literal["DeepLabCut", "LightningPose"] = "DeepLabCut",
) -> xr.Dataset:
    """Create a ``movement`` dataset from a DeepLabCut-style DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the pose tracks and confidence scores. Must
        be formatted as in DeepLabCut output files (see Notes).
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.
    source_software : str, optional
        Name of the pose estimation software from which the data originate.
        Defaults to "DeepLabCut", but it can also be "LightningPose"
        (because they the same DataFrame format).

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

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
    movement.io.load_poses.from_dlc_file

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

    return poses_from_numpy(
        position_array=tracks_with_scores[:, :, :, :-1],
        confidence_array=tracks_with_scores[:, :, :, -1],
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software=source_software,
    )


def from_sleap_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a SLEAP file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the SLEAP predictions in .h5
        (analysis) format. Alternatively, a .slp (labels) file can
        also be supplied (but this feature is experimental, see Notes).
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    The SLEAP predictions are normally saved in .slp files, e.g.
    "v1.predictions.slp". An analysis file, suffixed with ".h5" can be exported
    from the .slp file, using either the command line tool `sleap-convert`
    (with the "--format analysis" option enabled) or the SLEAP GUI (Choose
    "Export Analysis HDF5…" from the "File" menu) [1]_. This is the
    preferred format for loading pose tracks from SLEAP into *movement*.

    You can also directly load the .slp file. However, if the file contains
    multiple videos, only the pose tracks from the first video will be loaded.
    If the file contains a mix of user-labelled and predicted instances, user
    labels are prioritised over predicted instances to mirror SLEAP's approach
    when exporting .h5 analysis files [2]_.

    *movement* expects the tracks to be assigned and proofread before loading
    them, meaning each track is interpreted as a single individual. If
    no tracks are found in the file, *movement* assumes that this is a
    single-individual track, and will assign a default individual name.
    If multiple instances without tracks are present in a frame, the last
    instance is selected [2]_.
    Follow the SLEAP guide for tracking and proofreading [3]_.

    References
    ----------
    .. [1] https://sleap.ai/tutorials/analysis.html
    .. [2] https://github.com/talmolab/sleap/blob/v1.3.3/sleap/info/write_tracking_h5.py#L59
    .. [3] https://sleap.ai/guides/proofreading.html

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
        ds = _ds_from_sleap_analysis_file(file.path, fps=fps)
    else:  # file.path.suffix == ".slp"
        ds = _ds_from_sleap_labels_file(file.path, fps=fps)

    # Add metadata as attrs
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded pose tracks from {file.path}:")
    logger.info(ds)
    return ds


def from_lp_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a LightningPose file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the predicted poses, in .csv format.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_lp_file("path/to/file.csv", fps=30)

    """
    return _ds_from_lp_or_dlc_file(
        file_path=file_path, source_software="LightningPose", fps=fps
    )


def from_dlc_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a DeepLabCut file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the predicted poses, either in .h5
        or .csv format.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    See Also
    --------
    movement.io.load_poses.from_dlc_style_df

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_dlc_file("path/to/file.h5", fps=30)

    """
    return _ds_from_lp_or_dlc_file(
        file_path=file_path, source_software="DeepLabCut", fps=fps
    )


def from_via_tracks_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Load VIA tracks file into an xarray Dataset.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the VIA tracks, in .csv format.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        Dataset containing the bounding boxes' tracks, confidence scores, and
        metadata.

    Notes
    -----
    TODO: csv files, confidence scores and include references

    References
    ----------
    .. [1] https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_via_tracks_file("path/to/file.csv", fps=30)

    """
    # Validate the file
    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=[".csv"]
    )
    # Validate specific VIA file
    # validate specific csv file (checks header)
    # TODO: more checks! e.g. if shape is "rect"?
    via_file = ValidVIAtracksCSV(file.path)
    logger.debug(f"Validated VIA tracks csv file {via_file.path}.")

    # Extract numpy arrays to a dict
    bboxes_arrays = _numpy_arrays_from_via_tracks_file(via_file.path)

    # Create a dataset from numpy arrays
    # (it creates a ValidBboxesDataset in between)
    ds = bboxes_from_numpy(
        position_array=bboxes_arrays["position_array"],
        shape_array=bboxes_arrays["shape_array"],
        confidence_array=bboxes_arrays["confidence_array"],
        individual_names=bboxes_arrays["individual_names"],
        fps=fps,
        source_software="VIA-tracks",
    )
    logger.debug(f"Validated bounding boxes' tracks from {via_file.path}.")

    # Add metadata as attrs
    ds.attrs["source_software"] = "VIA-tracks"
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded bounding boxes' tracks from {via_file.path}:")
    logger.info(ds)
    return ds


def _ds_from_lp_or_dlc_file(
    file_path: Path | str,
    source_software: Literal["LightningPose", "DeepLabCut"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` dataset from a LightningPose or DeepLabCut file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the predicted poses, either in .h5
        or .csv format.
    source_software : {'LightningPose', 'DeepLabCut'}
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    expected_suffix = [".csv"]
    if source_software == "DeepLabCut":
        expected_suffix.append(".h5")

    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=expected_suffix
    )

    # Load the DeepLabCut poses into a DataFrame
    if file.path.suffix == ".csv":
        df = _df_from_dlc_csv(file.path)
    else:  # file.path.suffix == ".h5"
        df = _df_from_dlc_h5(file.path)

    logger.debug(f"Loaded poses from {file.path} into a DataFrame.")
    # Convert the DataFrame to an xarray dataset
    ds = from_dlc_style_df(df=df, fps=fps, source_software=source_software)

    # Add metadata as attrs
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded pose tracks from {file.path}:")
    logger.info(ds)
    return ds


def _ds_from_sleap_analysis_file(
    file_path: Path, fps: float | None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a SLEAP analysis (.h5) file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the SLEAP analysis file containing predicted pose tracks.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame units.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    file = ValidHDF5(file_path, expected_datasets=["tracks"])

    with h5py.File(file.path, "r") as f:
        # transpose to shape: (n_frames, n_tracks, n_keypoints, n_space)
        tracks = f["tracks"][:].transpose((3, 0, 2, 1))
        # Create an array of NaNs for the confidence scores
        scores = np.full(tracks.shape[:-1], np.nan)
        individual_names = [n.decode() for n in f["track_names"][:]] or None
        if individual_names is None:
            log_warning(
                f"Could not find SLEAP Track in {file.path}. "
                "Assuming single-individual dataset and assigning "
                "default individual name."
            )
        # If present, read the point-wise scores,
        # and transpose to shape: (n_frames, n_tracks, n_keypoints)
        if "point_scores" in f:
            scores = f["point_scores"][:].transpose((2, 0, 1))
        return poses_from_numpy(
            position_array=tracks.astype(np.float32),
            confidence_array=scores.astype(np.float32),
            individual_names=individual_names,
            keypoint_names=[n.decode() for n in f["node_names"][:]],
            fps=fps,
            source_software="SLEAP",
        )


def _ds_from_sleap_labels_file(
    file_path: Path, fps: float | None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a SLEAP labels (.slp) file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the SLEAP labels file containing predicted pose tracks.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame units.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    file = ValidHDF5(file_path, expected_datasets=["pred_points", "metadata"])
    labels = read_labels(file.path.as_posix())
    tracks_with_scores = _sleap_labels_to_numpy(labels)
    individual_names = [track.name for track in labels.tracks] or None
    if individual_names is None:
        log_warning(
            f"Could not find SLEAP Track in {file.path}. "
            "Assuming single-individual dataset and assigning "
            "default individual name."
        )
    return poses_from_numpy(
        position_array=tracks_with_scores[:, :, :, :-1],
        confidence_array=tracks_with_scores[:, :, :, -1],
        individual_names=individual_names,
        keypoint_names=[kp.name for kp in labels.skeletons[0].nodes],
        fps=fps,
        source_software="SLEAP",
    )


def _numpy_arrays_from_via_tracks_file(file_path: Path) -> dict:
    """Load and validate data from a VIA tracks file.

    Return a ValidBboxesDataset instance.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the VIA tracks file containing bounding boxes' tracks.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the `time` coordinates will be in frame units.

    Returns
    -------
    movement.io.tracks_validators.ValidPosesDataset
        The validated bounding boxes' tracks and confidence scores.

    """
    # # validate specific csv file (checks header)
    # # TODO: more checks! e.g. if shape is "rect"?
    # file = ValidVIAtracksCSV(file_path)
    # file.path.as_posix()

    # Read file as a dataframe with columns the desired data
    # TODO: add confidence with nans if not provided
    df = _load_df_from_via_tracks_file(file_path)

    # ---------------------------------------------------------
    # Compute position_array and shape_array
    list_unique_bbox_IDs = sorted(df.ID.unique().tolist())
    list_unique_frames = sorted(df.frame_number.unique().tolist())
    # assert 0 not in list_unique_bbox_IDs

    list_centroid_arrays, list_shape_arrays = [], []
    for bbox_id in list_unique_bbox_IDs:
        # Get subset dataframe for one bbox (frane_number, x, y, w, h, ID)
        df_one_bbox = df.loc[df["ID"] == bbox_id]
        df_one_bbox = df_one_bbox.sort_values(by="frame_number")

        # Drop rows with same frame_number and ID
        # (if manual annotation, sometimes the same ID appears >1 in a frame)
        if len(df_one_bbox.frame_number.unique()) != len(df_one_bbox):
            print(f"ID {bbox_id} appears more than once in a frame")
            print("Dropping duplicates")
            df_one_bbox = df_one_bbox.drop_duplicates(
                subset=["frame_number", "ID"],  # they may differ in x,y,w,h
                keep="first",  # or last?
            )

        # Reindex based on full set of unique frames
        # (otherwise only the ones for this bbox are in the df)
        df_one_bbox_reindexed = (
            df_one_bbox.set_index("frame_number")
            .reindex(list_unique_frames)
            .reset_index()
        )

        # Convert to numpy arrays
        list_centroid_arrays.append(
            df_one_bbox_reindexed[["x", "y"]].to_numpy()
        )
        list_shape_arrays.append(df_one_bbox_reindexed[["w", "h"]].to_numpy())
        # TODO: add confidence array

    # Concatenate centroid arrays and shape arrays for all IDs
    centroid_array = np.stack(list_centroid_arrays, axis=1)
    shape_array = np.stack(list_shape_arrays, axis=1)

    # ---------------------------------------------------------
    # Return dict of arrays
    # - position_array: (n_frames, n_individual_names, n_space) ----> x, y
    # - shape_array: (n_frames, n_individual_names, n_space)
    # ----> width, height
    # - IDs: list of unique IDs
    # - TODO: confidence_array: (n_frames, n_individuals, n_keypoints)
    # TODO: review ID as string; what is more consistent with what we have?
    return {
        "position_array": centroid_array,
        "shape_array": shape_array,
        "individual_names": ["a", "b", "c"],  # individual_names,
        "confidence_array": np.zeros((2, 2, 2, 2)),  # TODO
        # "fps"=fps,
        # "source_software"="VIA-tracks",
    }


def _load_df_from_via_tracks_file(file_path: Path) -> pd.DataFrame:
    """Load a VIA tracks file into a (restructured) pandas DataFrame."""
    # Read file as dataframe
    df_file = pd.read_csv(file_path, sep=",", header=0)

    # Extract frame number
    # TODO return numpy array?
    # TODO: improve
    list_frame_numbers = _extract_frame_number_from_via_tracks_df(df_file)

    # Extract x,y,w,h of bboxes as numpy arrays
    # TODO: extract confidence if exists
    bbox_xy_array = _via_attribute_column_to_numpy(
        df_file, "region_shape_attributes", ["x", "y"], float
    )
    bbox_wh_array = _via_attribute_column_to_numpy(
        df_file, "region_shape_attributes", ["width", "height"], float
    )
    bbox_ID_array = _via_attribute_column_to_numpy(
        df_file, "region_attributes", ["track"], int
    )

    # Make a dataframe with restructured data
    # TODO: add confidence if exists, otherwise nans
    return pd.DataFrame(
        {
            "frame_number": list_frame_numbers,  # make this a numpy array too?
            "x": bbox_xy_array[:, 0],
            "y": bbox_xy_array[:, 1],
            "w": bbox_wh_array[:, 0],
            "h": bbox_wh_array[:, 1],
            "ID": bbox_ID_array[:, :],
        }
    )


def _extract_frame_number_from_via_tracks_df(df):
    # Check if frame number is defined as file_attributes, else get from
    # filename
    # return as numpy array?
    # improve this;

    # Extract frame number from file_attributes if exists
    _via_attribute_column_to_numpy(
        df,
        via_attribute_column_name="file_attributes",
        list_attributes=["frame"],  # make this an input with a default?
        type_fn=int,
    )

    # Else extract from filename
    # frame number is between "_" and ".", led by at least one zero, followed
    # by extension
    pattern = r"_(0\d*)\.\w+$"  # before: r"_(0\d*)\."

    list_frame_numbers = [
        int(re.search(pattern, f).group(1))  # type: ignore
        if re.search(pattern, f)
        else np.nan
        for f in df["filename"]
    ]

    return list_frame_numbers


def _via_attribute_column_to_numpy(
    df: pd.DataFrame,
    via_attribute_column_name: str,
    list_attributes: list[str],
    cast_fn: Callable = float,
) -> np.ndarray:
    """Convert values from VIA attribute-type column to a float numpy array.

    Arrays will have at least 1 column (no 0-rank numpy arrays).
    If several values, these are passed as columns
    axis 0 is rows of df

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data.
    via_attribute_column_name : str
        The name of the column that holds values as literal dictionaries.
    list_attributes : list[str]
        The list of keys whose values we want to extract from the literal
        dictionaries.
    cast_fn : type, optional
        The type function to cast the values to, by default float.

    Returns
    -------
    np.ndarray
        A numpy array of floats representing the extracted attribute values.

    """
    # initialise list
    list_bbox_attr = []

    # iterate through rows
    for _, row in df.iterrows():
        # extract attributes from the column of interest
        list_bbox_attr.append(
            tuple(
                cast_fn(ast.literal_eval(row[via_attribute_column_name])[reg])
                for reg in list_attributes
            )
        )

    # convert to numpy array
    bbox_attr_array = np.array(list_bbox_attr)

    return bbox_attr_array


##################################


def _sleap_labels_to_numpy(labels: Labels) -> np.ndarray:
    """Convert a SLEAP ``Labels`` object to a NumPy array.

    The output array contains pose tracks and point-wise confidence scores.

    Parameters
    ----------
    labels : Labels
        A SLEAP `Labels` object.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing pose tracks and confidence scores.

    Notes
    -----
    This function only considers SLEAP instances in the first
    video of the SLEAP `Labels` object. User-labelled instances are
    prioritised over predicted instances, mirroring SLEAP's approach
    when exporting .h5 analysis files [1]_.

    This function is adapted from `Labels.numpy()` from the
    `sleap_io` package [2]_.

    References
    ----------
    .. [1] https://github.com/talmolab/sleap/blob/v1.3.3/sleap/info/write_tracking_h5.py#L59
    .. [2] https://github.com/talmolab/sleap-io

    """
    # Select frames from the first video only
    lfs = [lf for lf in labels.labeled_frames if lf.video == labels.videos[0]]
    # Figure out frame index range
    frame_idxs = [lf.frame_idx for lf in lfs]
    first_frame = min(0, min(frame_idxs))
    last_frame = max(0, max(frame_idxs))

    n_tracks = len(labels.tracks) or 1  # If no tracks, assume 1 individual
    individuals = labels.tracks or [None]
    skeleton = labels.skeletons[-1]  # Assume project only uses last skeleton
    n_nodes = len(skeleton.nodes)
    n_frames = int(last_frame - first_frame + 1)
    tracks = np.full((n_frames, n_tracks, n_nodes, 3), np.nan, dtype="float32")

    for lf in lfs:
        i = int(lf.frame_idx - first_frame)
        user_instances = lf.user_instances
        predicted_instances = lf.predicted_instances
        for j, ind in enumerate(individuals):
            user_track_instances = [
                inst for inst in user_instances if inst.track == ind
            ]
            predicted_track_instances = [
                inst for inst in predicted_instances if inst.track == ind
            ]
            # Use user-labelled instance if available
            if user_track_instances:
                inst = user_track_instances[-1]
                tracks[i, j] = np.hstack(
                    (inst.numpy(), np.full((n_nodes, 1), np.nan))
                )
            elif predicted_track_instances:
                inst = predicted_track_instances[-1]
                tracks[i, j] = inst.numpy(scores=True)
    return tracks


def _df_from_dlc_csv(file_path: Path) -> pd.DataFrame:
    """Create a DeepLabCut-style DataFrame from a .csv file.

    If poses are loaded from a DeepLabCut-style .csv file, the DataFrame
    lacks the multi-index columns that are present in the .h5 file. This
    function parses the .csv file to DataFrame with multi-index columns,
    i.e. the same format as in the .h5 file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the DeepLabCut-style .csv file containing pose tracks.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style DataFrame with multi-index columns.

    """
    file = ValidDeepLabCutCSV(file_path)

    possible_level_names = ["scorer", "individuals", "bodyparts", "coords"]
    with open(file.path) as f:
        # if line starts with a possible level name, split it into a list
        # of strings, and add it to the list of header lines
        header_lines = [
            line.strip().split(",")
            for line in f.readlines()
            if line.split(",")[0] in possible_level_names
        ]

    # Form multi-index column names from the header lines
    level_names = [line[0] for line in header_lines]
    column_tuples = list(
        zip(*[line[1:] for line in header_lines], strict=False)
    )
    columns = pd.MultiIndex.from_tuples(column_tuples, names=level_names)

    # Import the DeepLabCut poses as a DataFrame
    df = pd.read_csv(
        file.path,
        skiprows=len(header_lines),
        index_col=0,
        names=np.array(columns),
    )
    df.columns.rename(level_names, inplace=True)
    return df


def _df_from_dlc_h5(file_path: Path) -> pd.DataFrame:
    """Create a DeepLabCut-style DataFrame from a .h5 file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the DeepLabCut-style HDF5 file containing pose tracks.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style DataFrame with multi-index columns.

    """
    file = ValidHDF5(file_path, expected_datasets=["df_with_missing"])
    # pd.read_hdf does not always return a DataFrame but we assume it does
    # in this case (since we know what's in the "df_with_missing" dataset)
    df = pd.DataFrame(pd.read_hdf(file.path, key="df_with_missing"))
    return df


def _poses_ds_from_valid_data(data: ValidPosesDataset) -> xr.Dataset:
    """Create a ``movement`` dataset from validated pose tracking data.

    Parameters
    ----------
    data : movement.io.tracks_validators.ValidPosesDataset
        The validated data object.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    n_frames = data.position_array.shape[0]
    n_space = data.position_array.shape[-1]

    # Create the time coordinate, depending on the value of fps
    time_coords = np.arange(n_frames, dtype=int)
    time_unit = "frames"
    if data.fps is not None:
        time_coords = time_coords / data.fps
        time_unit = "seconds"

    DIM_NAMES = MovementDataset.dim_names
    # Convert data to an xarray.Dataset
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(data.position_array, dims=DIM_NAMES),
            "confidence": xr.DataArray(
                data.confidence_array, dims=DIM_NAMES[:-1]
            ),
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
            "source_software": data.source_software,
            "source_file": None,
        },
    )


############################
# From valid dataset structure to xr.dataset
def _bboxes_ds_from_valid_data(data: ValidBboxesDataset) -> xr.Dataset:
    """Convert already validated bboxes tracking data to an xarray Dataset.

    Parameters
    ----------
    data : movement.io.tracks_validators.ValidPosesDataset
        The validated data object.

    Returns
    -------
    xarray.Dataset
        Dataset containing the pose tracks, confidence scores, and metadata.

    """
    # TODO: a lot of common code with pose ds, can we combine/refactor?

    n_frames = data.position_array.shape[0]
    n_space = data.position_array.shape[-1]

    # Create the time coordinate, depending on the value of fps
    time_coords = np.arange(n_frames, dtype=int)
    time_unit = "frames"
    if data.fps is not None:
        time_coords = time_coords / data.fps
        time_unit = "seconds"

    DIM_NAMES = MovementDataset.dim_names

    # Convert data to an xarray.Dataset
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(data.position_array, dims=DIM_NAMES),
            "shape": xr.DataArray(data.shape_array, dims=DIM_NAMES[:-1]),
            "confidence": xr.DataArray(
                data.confidence_array, dims=DIM_NAMES[:-1]
            ),
        },
        coords={
            DIM_NAMES[0]: time_coords,
            DIM_NAMES[1]: data.individual_names,
            DIM_NAMES[3]: ["x", "y", "z"][:n_space],  # TODO: w, h?
        },
        attrs={
            "fps": data.fps,
            "time_unit": time_unit,
            "source_software": data.source_software,
            "source_file": None,
        },
    )
