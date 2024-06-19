"""Functions for loading bounding boxes tracking data."""

import ast
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from movement import MovementDataset
from movement.utils.logging import log_error  # , log_warning
from movement.validators.datasets import ValidBboxesDataset
from movement.validators.files import ValidFile, ValidVIAtracksCSV

logger = logging.getLogger(__name__)


def from_file(
    file_path: Path | str,
    source_software: Literal["VIA-tracks"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` dataset from a supported tracked bboxes file.

    At the moment, we only support VIA-tracks files.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing predicted poses. The file format must
        be among those supported by the ``from_dlc_file()``,
        ``from_slp_file()`` or ``from_lp_file()`` functions. One of these
        these functions will be called internally, based on
        the value of ``source_software``.
    source_software : "VIA-tracks".
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the tracked bounding boxes, their
        confidence scores, and associated metadata.

    See Also
    --------
    movement.io.load_poses.from_via_tracks_file

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_file(
    ...     "path/to/file.csv", source_software="VIA-tracks", fps=30
    ... )

    """
    if source_software == "VIA-tracks":
        return from_via_tracks_file(file_path, fps)
    else:
        raise log_error(
            ValueError, f"Unsupported source software: {source_software}"
        )


def from_numpy(
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
    return _ds_from_valid_data(valid_bboxes_data)


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
    via_file = ValidVIAtracksCSV(file.path)
    logger.debug(f"Validated VIA tracks csv file {via_file.path}.")

    # Extract numpy arrays to a dict
    bboxes_arrays = _numpy_arrays_from_via_tracks_file(via_file.path)

    # Create a dataset from numpy arrays
    # (it creates a ValidBboxesDataset in between)
    # TODO
    ds = from_numpy(
        position_array=bboxes_arrays["position_array"],
        shape_array=bboxes_arrays["shape_array"],
        confidence_array=bboxes_arrays["confidence_array"],
        individual_names=bboxes_arrays["individual_names"],  # could be None
        fps=fps,
        source_software="VIA-tracks",
    )
    # logger.debug(f"Validated bounding boxes' tracks from {via_file.path}.")

    # Add metadata as attrs
    ds.attrs["source_software"] = "VIA-tracks"
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded bounding boxes' tracks from {via_file.path}:")
    logger.info(ds)
    return ds


def _numpy_arrays_from_via_tracks_file(file_path: Path) -> dict:
    """Extract numpy arrays from a VIA tracks file.

    The numpy arrays are:
    - position_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_space)
        containing the poses. It will be converted to a
        :py:class:`xarray.DataArray` object named "position".
    - shape_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_space)
        containing the poses. It will be converted to a
        :py:class:`xarray.DataArray` object named "position".
    - confidence_array : np.ndarray
        Array of shape (n_frames, n_individuals) containing
        the point-wise confidence scores. It will be converted to a
        :py:class:`xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    - individual_names : list of str
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "id_0",
        "id_1", etc.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the VIA tracks file containing bounding boxes' tracks.

    Returns
    -------
    dict
        The validated bounding boxes' tracks and confidence scores.

    """
    # Read file as dataframe
    df_file = pd.read_csv(file_path, sep=",", header=0)

    # Extract 2D dataframe from input dataframe
    df = _reformat_via_tracks_df(df_file)

    # Compute indices of df where ID switches
    bool_ID_diff_from_prev = df["ID"].ne(df["ID"].shift())  # pandas series
    idcs_ID_switch = (
        bool_ID_diff_from_prev.loc[lambda x: x].index[1:].to_numpy()
    )

    # Stack position, shape and confidence arrays
    map_key_to_columns = {
        "position_array": ["x", "y"],
        "shape_array": ["w", "h"],
        "confidence_array": ["confidence"],
    }
    array_dict = {}
    for key in map_key_to_columns:
        list_arrays = np.split(
            df[map_key_to_columns[key]].to_numpy(),
            idcs_ID_switch,  # along axis=0
        )

        array_dict[key] = np.stack(list_arrays)

    # Add remaining arrays
    array_dict["individual_names"] = df["ID"].unique()
    array_dict["frame_array"] = df["frame_number"].unique()

    return array_dict


def _reformat_via_tracks_df(df: pd.DataFrame) -> pd.DataFrame:
    # Extract 2D arrays from input dataframe
    # TODO: return as dict instead
    (
        bbox_position_array,
        bbox_shape_array,
        bbox_ID_array,
        bbox_confidence_array,
        frame_array,
    ) = _extract_2d_arrays_from_via_tracks_df(df)

    # Make a 2D dataframe
    df = pd.DataFrame(
        {
            "frame_number": frame_array[:, 0],
            "x": bbox_position_array[:, 0],
            "y": bbox_position_array[:, 1],
            "w": bbox_shape_array[:, 0],
            "h": bbox_shape_array[:, 1],
            "confidence": bbox_confidence_array[:, 0],
            "ID": bbox_ID_array[:, 0],
        }
    )

    # Important!
    # Sort dataframe by ID and frame number
    df = df.sort_values(by=["ID", "frame_number"]).reset_index(drop=True)

    # Compute desired index: all combinations of ID and frame number
    multi_index = pd.MultiIndex.from_product(
        [df["ID"].unique(), df["frame_number"].unique()],
        names=["ID", "frame_number"],
    )

    # Set index to ID, frame number, fill in values with nans and reset index
    df = (
        df.set_index(["ID", "frame_number"]).reindex(multi_index).reset_index()
    )
    return df


def _extract_2d_arrays_from_via_tracks_df(df: pd.DataFrame) -> tuple:
    # frame number array
    frame_array = _extract_frame_number_from_via_tracks_df(df)  # 2D

    # position 2D array
    # rows: frames
    # columns: x,y
    bbox_position_array = _via_attribute_column_to_numpy(
        df, "region_shape_attributes", ["x", "y"], float
    )

    # shape 2D array
    bbox_shape_array = _via_attribute_column_to_numpy(
        df, "region_shape_attributes", ["width", "height"], float
    )

    # track 2D array
    bbox_ID_array = _via_attribute_column_to_numpy(
        df, "region_attributes", ["track"], int
    )

    # confidence 2D array
    region_attributes_dicts = [
        ast.literal_eval(d) for d in df.region_attributes
    ]
    if all(["confidence" in d for d in region_attributes_dicts]):
        bbox_confidence_array = _via_attribute_column_to_numpy(
            df, "region_attributes", ["confidence"], float
        )
    else:
        bbox_confidence_array = np.full(frame_array.shape, np.nan)

    # TODO: return as dict instead
    return (
        bbox_position_array,
        bbox_shape_array,
        bbox_ID_array,
        bbox_confidence_array,
        frame_array,
    )


# TODO! check this
def _extract_frame_number_from_via_tracks_df(df) -> np.ndarray:
    """_summary_.

    Parameters
    ----------
    df : _type_
        _description_

    Returns
    -------
    np.ndarray
        A 2D numpy array containing the frame numbers.

    """
    # Check if frame number is defined as file_attributes, else get from
    # filename
    # return as numpy array?
    # improve this;

    # Extract frame number from file_attributes if exists
    # Extract list of file attributes (dicts)
    file_attributes_dicts = [ast.literal_eval(d) for d in df.file_attributes]

    if all(["frame" in d for d in file_attributes_dicts]):
        frame_array = _via_attribute_column_to_numpy(
            df,
            via_column_name="file_attributes",
            list_attributes=["frame"],  # make this an input with a default?
            cast_fn=int,
        )
    else:
        # Else extract from filename
        # frame number is between "_" and ".", led by at least one zero,
        # followed by the file extension
        pattern = r"_(0\d*)\.\w+$"  # before: r"_(0\d*)\."

        list_frame_numbers = [
            int(re.search(pattern, f).group(1))  # type: ignore
            if re.search(pattern, f)
            else np.nan
            for f in df["filename"]
        ]

        frame_array = np.array(list_frame_numbers).reshape(-1, 1)

    return frame_array


def _via_attribute_column_to_numpy(
    df: pd.DataFrame,
    via_column_name: str,
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
        The pandas DataFrame containing the data from the VIA file.
    via_column_name : str
        The name of a column in the VIA file whose values are literal
        dictionaries (e.g. `file_attributes`, `region_shape_attributes`
        or `region_attributes`).
    list_attributes : list[str]
        The list of keys whose values we want to extract from the literal
        dictionaries in the `via_column_name` column.
    cast_fn : type, optional
        The type function to cast the values to. By default `float`.

    Returns
    -------
    np.ndarray
        A numpy array representing the extracted attribute values.

    """
    # initialise list
    list_bbox_attr = []

    # iterate through rows
    for _, row in df.iterrows():
        # extract attributes from the column of interest
        list_bbox_attr.append(
            tuple(
                cast_fn(ast.literal_eval(row[via_column_name])[reg])
                for reg in list_attributes
            )
        )

    # convert to numpy array
    bbox_attr_array = np.array(list_bbox_attr)

    return bbox_attr_array


# TODO
# From valid dataset structure to xr.dataset
def _ds_from_valid_data(data: ValidBboxesDataset) -> xr.Dataset:
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
