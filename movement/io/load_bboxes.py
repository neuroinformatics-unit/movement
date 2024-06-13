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
    # validate specific csv file (checks header)
    # TODO: more checks! e.g. if shape is "rect"?
    via_file = ValidVIAtracksCSV(file.path)
    logger.debug(f"Validated VIA tracks csv file {via_file.path}.")

    # Extract numpy arrays to a dict
    # TODO: get from notebook
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
    logger.debug(f"Validated bounding boxes' tracks from {via_file.path}.")

    # Add metadata as attrs
    ds.attrs["source_software"] = "VIA-tracks"
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded bounding boxes' tracks from {via_file.path}:")
    logger.info(ds)
    return ds


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
    # Read file as a dataframe with columns the desired data
    # TODO: add confidence with nans if not provided
    df = _load_df_from_via_tracks_file(file_path)

    # Extract numpy arrays -- see notebook
    # ---------------------------------------------------------
    # Compute position_array and shape_array
    list_unique_bbox_IDs = sorted(df.ID.unique().tolist())
    list_unique_frames = sorted(df.frame_number.unique().tolist())

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
    return {
        "position_array": centroid_array,
        "shape_array": shape_array,
        "individual_names": ["a", "b", "c"],  # could be None!,
        "confidence_array": np.zeros((2, 2, 2, 2)),  # TODO
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
