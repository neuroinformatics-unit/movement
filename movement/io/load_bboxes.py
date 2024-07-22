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
from movement.utils.logging import log_error
from movement.validators.datasets import ValidBboxesDataset
from movement.validators.files import ValidFile, ValidVIATracksCSV

logger = logging.getLogger(__name__)


def from_numpy(
    position_array: np.ndarray,
    shape_array: np.ndarray,
    confidence_array: np.ndarray | None = None,
    individual_names: list[str] | None = None,
    frame_array: np.ndarray | None = None,
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
        :py:class:`xarray.DataArray` object named "shape".
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals) containing
        the point-wise confidence scores. It will be converted to a
        :py:class:`xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "id_0",
        "id_1", etc.
    frame_array : np.ndarray, optional
        Array of shape (n_frames, 1) containing the frame numbers. If no frame
        numbers are supplied (default), the frames will be numbered
        based on the position_array data and starting from 0.
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
    Create random position data for two bounding boxes, ``id_0`` and ``id_1``,
    with the same width (40 pixels) and height (30 pixels). These are tracked
    in 2D space for 100 frames, which are numbered from the start frame 1200
    to the end frame 1299. The confidence score for all bounding boxes is set
    to 0.5.

    >>> import numpy as np
    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_numpy(
    ...     position_array=np.random.rand((100, 2, 2)),
    ...     shape_array=np.ones((100, 2, 2)) * [40, 30],
    ...     confidence_array=np.ones((100, 2)) * 0.5,
    ...     individual_names=["id_0", "id_1"],
    ...     frame_array=np.arange(1200, 1300).reshape(-1, 1),
    ... )

    """
    valid_bboxes_data = ValidBboxesDataset(
        position_array=position_array,
        shape_array=shape_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        frame_array=frame_array,
        fps=fps,
        source_software=source_software,
    )
    return _ds_from_valid_data(valid_bboxes_data)


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
        Path to the file containing the tracked bounding boxes. Currently
        only VIA-tracks .csv files are supported.
    source_software : "VIA-tracks".
        The source software of the file. Currently only "VIA-tracks
        is supported.
    fps : float, optional
        The number of frames per second in the video. If provided, the
        ``time`` coordinates will be in seconds. If None (default), the
        ``time`` coordinates will be in frame numbers. If no frame numbers
        are provided in the file, frame numbers will be assigned based on
        the position data and starting from 0.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the position, shape, and confidence
        scores of the tracked bounding boxes, and associated metadata.

    See Also
    --------
    movement.io.load_bboxes.from_via_tracks_file

    Examples
    --------
    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_file(
    ...     "path/to/file.csv", source_software="VIA-tracks", fps=30
    ... )

    """
    if source_software == "VIA-tracks":
        return from_via_tracks_file(file_path, fps)
    else:
        raise log_error(
            ValueError, f"Unsupported source software: {source_software}"
        )


def from_via_tracks_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a VIA tracks .csv file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the VIA file with the tracked bounding boxes, in .csv format.
        For more information on the VIA tracks file format, see the VIA
        tutorial for tracking [1]_.
    fps : float, optional
        The number of frames per second in the video.  If provided, the
        ``time`` coordinates will be in seconds. If None (default), the
        ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` bounding boxes dataset containing the boxes tracks,
        boxes shapes, confidence scores and associated metadata.

    Notes
    -----
    For each bounding box, the ID specified in the "track" field of the VIA
    file is expressed as "individual_name" in the xarray.Dataset. The
    individual names follow the format `id_<N>`, with N being the bounding box
    ID.

    References
    ----------
    .. [1] https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html

    Examples
    --------
    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_via_tracks_file("path/to/file.csv", fps=30)

    """
    # General file validation
    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=[".csv"]
    )

    # Specific VIA-file validation
    via_file = ValidVIATracksCSV(file.path)
    logger.debug(f"Validated VIA tracks csv file {via_file.path}.")

    # Create an xarray.Dataset from the data
    bboxes_arrays = _numpy_arrays_from_via_tracks_file(via_file.path)
    ds = from_numpy(
        position_array=bboxes_arrays["position_array"],
        shape_array=bboxes_arrays["shape_array"],
        confidence_array=bboxes_arrays["confidence_array"],
        individual_names=[
            f"id_{id}" for id in bboxes_arrays["ID_array"].squeeze()
        ],
        frame_array=bboxes_arrays["frame_array"],
        fps=fps,
        source_software="VIA-tracks",
    )  # it validates the dataset via ValidBboxesDataset

    # Add metadata as attributes
    ds.attrs["source_software"] = "VIA-tracks"
    ds.attrs["source_file"] = file.path.as_posix()

    logger.info(f"Loaded bounding boxes' tracks from {via_file.path}:")
    logger.info(ds)
    return ds


def _numpy_arrays_from_via_tracks_file(file_path: Path) -> dict:
    """Extract numpy arrays from the input VIA tracks file.

    The extracted numpy arrays are:
    - position_array (n_frames, n_individuals, n_space):
        contains the trajectory of the bounding boxes' centroids.
    - shape_array (n_frames, n_individuals, n_space):
        contains the shape of the bounding boxes (width and height).
    - confidence_array (n_frames, n_individuals):
        contains the confidence score for each bounding box.
        If no confidence scores are provided, they are set to an array of NaNs.
    - ID_array (n_individuals, 1):
        contains the integer IDs of the tracked bounding boxes.
    - frame_array (n_frames, 1):
        contains the frame numbers.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the VIA tracks file containing the bounding boxes' tracks.

    Returns
    -------
    dict
        The validated bounding boxes' arrays.

    """
    # Extract 2D dataframe from input data
    # (sort data by ID and frame number and fill with nans)
    df = _df_from_via_tracks_file(file_path)

    # Compute indices of the rows where the IDs switch
    bool_ID_diff_from_prev = df["ID"].ne(df["ID"].shift())  # pandas series
    indices_ID_switch = (
        bool_ID_diff_from_prev.loc[lambda x: x].index[1:].to_numpy()
    )

    # Stack position, shape and confidence arrays along ID axis
    map_key_to_columns = {
        "position_array": ["x", "y"],
        "shape_array": ["w", "h"],
        "confidence_array": ["confidence"],
    }
    array_dict = {}
    for key in map_key_to_columns:
        list_arrays = np.split(
            df[map_key_to_columns[key]].to_numpy(),
            indices_ID_switch,  # indices along axis=0
        )

        array_dict[key] = np.stack(list_arrays, axis=1).squeeze()

    # Add remaining arrays to dict
    array_dict["ID_array"] = df["ID"].unique().reshape(-1, 1)
    array_dict["frame_array"] = df["frame_number"].unique().reshape(-1, 1)

    return array_dict


def _df_from_via_tracks_file(file_path: Path) -> pd.DataFrame:
    """Load VIA tracks file as a dataframe.

    Read the VIA tracks file as a pandas dataframe with columns:
    - ID: the integer ID of the tracked bounding box.
    - frame_number: the frame number of the tracked bounding box.
    - x: the x-coordinate of the tracked bounding box centroid.
    - y: the y-coordinate of the tracked bounding box centroid.
    - w: the width of the tracked bounding box.
    - h: the height of the tracked bounding box.
    - confidence: the confidence score of the tracked bounding box.

    The dataframe is sorted by ID and frame number, and for each ID,
    empty frames are filled in with NaNs.
    """
    # Read VIA tracks file as a pandas dataframe
    df_file = pd.read_csv(file_path, sep=",", header=0)

    # Format to a 2D dataframe
    df = pd.DataFrame(
        {
            "ID": _via_attribute_column_to_numpy(
                df_file, "region_attributes", ["track"], int
            ).squeeze(),
            "frame_number": _extract_frame_number_from_via_tracks_df(
                df_file
            ).squeeze(),
            "x": _via_attribute_column_to_numpy(
                df_file, "region_shape_attributes", ["x"], float
            ).squeeze(),
            "y": _via_attribute_column_to_numpy(
                df_file, "region_shape_attributes", ["y"], float
            ).squeeze(),
            "w": _via_attribute_column_to_numpy(
                df_file, "region_shape_attributes", ["width"], float
            ).squeeze(),
            "h": _via_attribute_column_to_numpy(
                df_file, "region_shape_attributes", ["height"], float
            ).squeeze(),
            "confidence": _extract_confidence_from_via_tracks_df(
                df_file
            ).squeeze(),
        }
    )

    # Sort dataframe by ID and frame number
    df = df.sort_values(by=["ID", "frame_number"]).reset_index(drop=True)

    # Fill in empty frames with nans
    multi_index = pd.MultiIndex.from_product(
        [df["ID"].unique(), df["frame_number"].unique()],
        names=["ID", "frame_number"],
    )  # desired index: all combinations of ID and frame number

    # Set index to (ID, frame number), fill in values with nans and
    # reset to original index
    df = (
        df.set_index(["ID", "frame_number"]).reindex(multi_index).reset_index()
    )
    return df


def _extract_confidence_from_via_tracks_df(df) -> np.ndarray:
    """Extract confidence scores from the VIA tracks input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The VIA tracks input dataframe is the one obtained from
        `df = pd.read_csv(file_path, sep=",", header=0)`.

    Returns
    -------
    np.ndarray
        A numpy array of size (n_bboxes, 1) containing the bounding boxes
        confidence scores.

    """
    region_attributes_dicts = [
        ast.literal_eval(d) for d in df.region_attributes
    ]

    # Check if confidence is defined as a region attribute, else set to NaN
    if all(["confidence" in d for d in region_attributes_dicts]):
        bbox_confidence = _via_attribute_column_to_numpy(
            df, "region_attributes", ["confidence"], float
        )
    else:
        bbox_confidence = np.full((df.shape[0], 1), np.nan)

    return bbox_confidence


def _extract_frame_number_from_via_tracks_df(df) -> np.ndarray:
    """Extract frame numbers from the VIA tracks input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The VIA tracks input dataframe is the one obtained from
        `df = pd.read_csv(file_path, sep=",", header=0)`.

    Returns
    -------
    np.ndarray
        A numpy array of size (n_frames, 1) containing the frame numbers.
        In the VIA tracks file, the frame number is expected to be defined as a
        'file_attribute' , or encoded in the filename as an integer number led
        by at least one zero, between "_" and ".", followed by the file
        extension.

    """
    # Extract frame number from file_attributes if exists
    file_attributes_dicts = [ast.literal_eval(d) for d in df.file_attributes]
    if all(["frame" in d for d in file_attributes_dicts]):
        frame_array = _via_attribute_column_to_numpy(
            df,
            via_column_name="file_attributes",
            list_keys=["frame"],
            cast_fn=int,
        )
    # Else extract from filename
    else:
        pattern = r"_(0\d*)\.\w+$"
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
    list_keys: list[str],
    cast_fn: Callable = float,
) -> np.ndarray:
    """Convert values from VIA attribute-type column to a numpy array.

    In the VIA tracks file, the attribute-type columns are the columns
    whose name includes the word `attributes` (i.e. `file_attributes`,
    `region_shape_attributes` or `region_attributes`). These columns hold
    dictionary data.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame containing the data from the VIA file.
        This is the dataframe obtained from running
        `df = pd.read_csv(file_path, sep=",", header=0)`.
    via_column_name : str
        The name of a column in the VIA file whose values are literal
        dictionaries (i.e. `file_attributes`, `region_shape_attributes`
        or `region_attributes`).
    list_keys : list[str]
        The list of keys whose values we want to extract from the literal
        dictionaries in the `via_column_name` column.
    cast_fn : type, optional
        The type function to cast the values to. By default `float`.

    Returns
    -------
    np.ndarray
        A numpy array holding the extracted values. The rows (axis=0) are the
        rows of the input dataframe. The columns (axis=1) follow the order of
        the `list_attributes`.  Arrays will have at least 1 column
        (no 0-rank numpy arrays).

    """
    list_bbox_attr = []
    for _, row in df.iterrows():
        row_dict_data = ast.literal_eval(row[via_column_name])
        list_bbox_attr.append(
            tuple(cast_fn(row_dict_data[reg]) for reg in list_keys)
        )

    bbox_attr_array = np.array(list_bbox_attr)

    return bbox_attr_array


def _ds_from_valid_data(data: ValidBboxesDataset) -> xr.Dataset:
    """Convert a validated bboxes dataset to an xarray Dataset.

    Parameters
    ----------
    data : movement.io.tracks_validators.ValidPosesDataset
        The validated data object.

    Returns
    -------
    bounding boxes dataset containing the boxes tracks,
        boxes shapes, confidence scores and associated metadata.

    """
    # Create the time coordinate
    time_coords = data.frame_array
    time_unit = "frames"
    # if fps is provided: time_coords is expressed in seconds.
    if data.fps is not None:
        # Compute frames from the start (first frame is frame 0).
        # Ignoring type error because `data.frame_array` is not None after
        # ValidBboxesDataset.__attrs_post_init__()
        time_coords = np.arange(data.frame_array.shape[0], dtype=int)  # type: ignore
        time_coords = time_coords / data.fps
        time_unit = "seconds"

    # Convert data to an xarray.Dataset
    # ('time', 'individuals', 'space')
    DIM_NAMES = tuple(a for a in MovementDataset.dim_names if a != "keypoints")
    n_space = data.position_array.shape[-1]
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(data.position_array, dims=DIM_NAMES),
            "shape": xr.DataArray(data.shape_array, dims=DIM_NAMES),
            "confidence": xr.DataArray(
                data.confidence_array, dims=DIM_NAMES[:-1]
            ),
        },
        # Ignoring type error because `time_coords`
        # (which is a function of `data.frame_array`)
        # cannot be None after
        # ValidBboxesDataset.__attrs_post_init__()
        coords={
            DIM_NAMES[0]: time_coords.squeeze(),  # type: ignore
            DIM_NAMES[1]: data.individual_names,
            DIM_NAMES[2]: ["x", "y", "z"][:n_space],
        },
        attrs={
            "fps": data.fps,
            "time_unit": time_unit,
            "source_software": data.source_software,
            "source_file": None,
            "ds_type": "bboxes",
        },
    )
