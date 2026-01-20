"""Load bounding boxes tracking data into ``movement``."""

import ast
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from movement.utils.logging import logger
from movement.validators.datasets import ValidBboxesInputs
from movement.validators.files import (
    DEFAULT_FRAME_REGEXP,
    ValidFile,
    ValidVIATracksCSV,
)


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
        Array of shape (n_frames, n_space, n_individuals)
        containing the tracks of the bounding box centroids.
        It will be converted to a :class:`xarray.DataArray` object
        named "position".
    shape_array : np.ndarray
        Array of shape (n_frames, n_space, n_individuals)
        containing the shape of the bounding boxes. The shape of a bounding
        box is its width (extent along the x-axis of the image) and height
        (extent along the y-axis of the image). It will be converted to a
        :class:`xarray.DataArray` object named "shape".
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals) containing
        the confidence scores of the bounding boxes. If None (default), the
        confidence scores are set to an array of NaNs. It will be converted
        to a :class:`xarray.DataArray` object named "confidence".
    individual_names : list of str, optional
        List of individual names for the tracked bounding boxes in the video.
        If None (default), bounding boxes are assigned names based on the size
        of the ``position_array``. The names will be in the format of
        ``id_<N>``, where <N>  is an integer from 0 to
        ``position_array.shape[-1]-1`` (i.e., "id_0", "id_1"...).
    frame_array : np.ndarray, optional
        Array of shape (n_frames, 1) containing the frame numbers for which
        bounding boxes are defined. If None (default), frame numbers will
        be assigned based on the first dimension of the ``position_array``,
        starting from 0. If a specific array of frame numbers is provided,
        these need to be integers sorted in increasing order.
    fps : float, optional
        The video sampling rate. If None (default), the ``time`` coordinates
        of the resulting ``movement`` dataset will be in frame numbers. If
        ``fps`` is provided, the ``time`` coordinates  will be in seconds. If
        the ``time`` coordinates are in seconds, they will indicate the
        elapsed time from the capture of the first frame (assumed to be frame
        0).
    source_software : str, optional
        Name of the software that generated the data. Defaults to None.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the position, shape, and confidence
        scores of the tracked bounding boxes, and any associated metadata.

    Examples
    --------
    Create random position data for two bounding boxes, ``id_0`` and ``id_1``,
    with the same width (40 pixels) and height (30 pixels). These are tracked
    in 2D space for 100 frames, which are numbered from the start frame 1200
    to the end frame 1299. The confidence score for all bounding boxes is set
    to 0.5.

    >>> import numpy as np
    >>> from movement.io import load_bboxes
    >>> rng = np.random.default_rng(seed=42)
    >>> ds = load_bboxes.from_numpy(
    ...     position_array=rng.random((100, 2, 2)),
    ...     shape_array=np.ones((100, 2, 2)) * [40, 30],
    ...     confidence_array=np.ones((100, 2)) * 0.5,
    ...     individual_names=["id_0", "id_1"],
    ...     frame_array=np.arange(1200, 1300).reshape(-1, 1),
    ... )

    Create a dataset with the same data as above, but with the time
    coordinates in seconds. We use a video sampling rate of 60 fps. The time
    coordinates in the resulting dataset will indicate the elapsed time from
    the capture of the 0th frame. So for the frames 1200, 1201, 1203,... 1299
    the corresponding time coordinates in seconds will be 20, 20.0167,
    20.033,... 21.65 s.

    >>> ds = load_bboxes.from_numpy(
    ...     position_array=rng.random((100, 2, 2)),
    ...     shape_array=np.ones((100, 2, 2)) * [40, 30],
    ...     confidence_array=np.ones((100, 2)) * 0.5,
    ...     individual_names=["id_0", "id_1"],
    ...     frame_array=np.arange(1200, 1300).reshape(-1, 1),
    ...     fps=60,
    ... )

    Create a dataset with the same data as above, but express the time
    coordinate in frames, and assume the first tracked frame is frame 0.
    To do this, we simply omit the ``frame_array`` input argument.

    >>> ds = load_bboxes.from_numpy(
    ...     position_array=rng.random((100, 2, 2)),
    ...     shape_array=np.ones((100, 2, 2)) * [40, 30],
    ...     confidence_array=np.ones((100, 2)) * 0.5,
    ...     individual_names=["id_0", "id_1"],
    ... )

    Create a dataset with the same data as above, but express the time
    coordinate in seconds, and assume the first tracked frame is captured
    at time = 0 seconds. To do this, we omit the ``frame_array`` input argument
    and pass an ``fps`` value.

    >>> ds = load_bboxes.from_numpy(
    ...     position_array=rng.random((100, 2, 2)),
    ...     shape_array=np.ones((100, 2, 2)) * [40, 30],
    ...     confidence_array=np.ones((100, 2)) * 0.5,
    ...     individual_names=["id_0", "id_1"],
    ...     fps=60,
    ... )

    """
    valid_bboxes_inputs = ValidBboxesInputs(
        position_array=position_array,
        shape_array=shape_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        frame_array=frame_array,
        fps=fps,
        source_software=source_software,
    )
    return valid_bboxes_inputs.to_dataset()


def from_file(
    file_path: Path | str,
    source_software: Literal["VIA-tracks"],
    fps: float | None = None,
    use_frame_numbers_from_file: bool = False,
    frame_regexp: str = DEFAULT_FRAME_REGEXP,
) -> xr.Dataset:
    """Create a ``movement`` bounding boxes dataset from a supported file.

    At the moment, we only support VIA tracks .csv files.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the tracked bounding boxes. Currently
        only VIA tracks .csv files are supported.
    source_software : "VIA-tracks".
        The source software of the file. Currently only files from the
        VIA 2.0.12 annotator [1]_ ("VIA-tracks") are supported.
        See .
    fps : float, optional
        The video sampling rate. If None (default), the ``time`` coordinates
        of the resulting ``movement`` dataset will be in frame numbers. If
        ``fps`` is provided, the ``time`` coordinates  will be in seconds. If
        the ``time`` coordinates are in seconds, they will indicate the
        elapsed time from the capture of the first frame (assumed to be frame
        0).
    use_frame_numbers_from_file : bool, optional
        If True, the frame numbers in the resulting dataset are
        the same as the ones specified for each tracked bounding box in the
        input file. This may be useful if the bounding boxes are tracked for a
        subset of frames in a video, but you want to maintain the start of the
        full video as the time origin. If False (default), the frame numbers
        in the VIA tracks .csv file are instead mapped to a 0-based sequence of
        consecutive integers.
    frame_regexp : str, optional
        Regular expression pattern to extract the frame number from the frame
        filename. By default, the frame number is expected to be encoded in
        the filename as an integer number led by at least one zero, followed
        by the file extension. Only used if ``use_frame_numbers_from_file`` is
        True.


    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the position, shape, and confidence
        scores of the tracked bounding boxes, and any associated metadata.

    See Also
    --------
    movement.io.load_bboxes.from_via_tracks_file

    References
    ----------
    .. [1] https://www.robots.ox.ac.uk/~vgg/software/via/

    Examples
    --------
    Create a dataset from the VIA tracks .csv file at "path/to/file.csv", with
    the time coordinates in seconds, and assuming t = 0 seconds corresponds to
    the first tracked frame in the file.

    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_file(
    >>>     "path/to/file.csv",
    >>>     source_software="VIA-tracks",
    >>>     fps=30,
    >>> )

    """
    if source_software == "VIA-tracks":
        return from_via_tracks_file(
            file_path,
            fps,
            use_frame_numbers_from_file=use_frame_numbers_from_file,
            frame_regexp=frame_regexp,
        )
    else:
        raise logger.error(
            ValueError(f"Unsupported source software: {source_software}")
        )


def from_via_tracks_file(
    file_path: Path | str,
    fps: float | None = None,
    use_frame_numbers_from_file: bool = False,
    frame_regexp: str = DEFAULT_FRAME_REGEXP,
) -> xr.Dataset:
    """Create a ``movement`` dataset from a VIA tracks .csv file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the VIA tracks .csv file with the tracked bounding boxes.
        For more information on the VIA tracks .csv file format, see the VIA
        tutorial for tracking [1]_.
    fps : float, optional
        The video sampling rate. If None (default), the ``time`` coordinates
        of the resulting ``movement`` dataset will be in frame numbers. If
        ``fps`` is provided, the ``time`` coordinates  will be in seconds. If
        the ``time`` coordinates are in seconds, they will indicate the
        elapsed time from the capture of the first frame (assumed to be frame
        0).
    use_frame_numbers_from_file : bool, optional
        If True, the frame numbers in the resulting dataset are
        the same as the ones in the VIA tracks .csv file. This may be useful if
        the bounding boxes are tracked for a subset of frames in a video,
        but you want to maintain the start of the full video as the time
        origin. If False (default), the frame numbers in the VIA tracks .csv
        file are instead mapped to a 0-based sequence of consecutive integers.
    frame_regexp : str, optional
        Regular expression pattern to extract the frame number from the frame
        filename. By default, the frame number is expected to be encoded in
        the filename as an integer number led by at least one zero, followed
        by the file extension. Only used if ``use_frame_numbers_from_file`` is
        True.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the position, shape, and confidence
        scores of the tracked bounding boxes, and any associated metadata.

    Notes
    -----
    Note that the x,y coordinates in the input VIA tracks .csv file
    represent the the top-left corner of each bounding box. Instead the
    corresponding ``movement`` dataset holds in its ``position`` array the
    centroid of each bounding box.

    Additionally, the bounding boxes IDs specified in the "track" field of
    the VIA tracks .csv file are mapped to the ``individuals`` dimension in the
    ``movement`` dataset. The individual names follow the format ``id_<N>``,
    with N being the bounding box ID.

    References
    ----------
    .. [1] https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html

    Examples
    --------
    Create a dataset from the VIA tracks .csv file at "path/to/file.csv", with
    the time coordinates in frames, and setting the first tracked frame in the
    file as frame 0.

    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_via_tracks_file(
    ...     "path/to/file.csv",
    ... )

    Create a dataset from the VIA tracks .csv file at "path/to/file.csv", with
    the time coordinates in seconds, and assuming t = 0 seconds corresponds to
    the first tracked frame in the file.

    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_via_tracks_file(
    ...     "path/to/file.csv",
    ...     fps=30,
    ... )

    Create a dataset from the VIA tracks .csv file at "path/to/file.csv", with
    the time coordinates in frames, and using the same frame numbers as
    in the VIA tracks .csv file.

    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_via_tracks_file(
    ...     "path/to/file.csv",
    ...     use_frame_numbers_from_file=True.
    ... )

    Create a dataset from the VIA tracks .csv file at "path/to/file.csv", with
    the time coordinates in seconds, and assuming t = 0 seconds corresponds to
    the 0th frame in the full video.

    >>> from movement.io import load_bboxes
    >>> ds = load_bboxes.from_via_tracks_file(
    ...     "path/to/file.csv",
    ...     fps=30,
    ...     use_frame_numbers_from_file=True,
    ... )


    """
    # General file validation
    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=[".csv"]
    )

    # Specific VIA-tracks .csv file validation
    via_file = ValidVIATracksCSV(file.path, frame_regexp=frame_regexp)
    logger.info(f"Validated VIA tracks .csv file {via_file.path}.")

    # Create an xarray.Dataset from the data
    bboxes_arrays = _numpy_arrays_from_via_tracks_file(
        via_file.path, via_file.frame_regexp
    )
    ds = from_numpy(
        position_array=bboxes_arrays["position_array"],
        shape_array=bboxes_arrays["shape_array"],
        confidence_array=bboxes_arrays["confidence_array"],
        individual_names=[
            f"id_{id.item()}" for id in bboxes_arrays["ID_array"]
        ],
        frame_array=(
            bboxes_arrays["frame_array"]
            if use_frame_numbers_from_file
            else None
        ),
        fps=fps,
        source_software="VIA-tracks",
    )  # it validates the dataset via ValidBboxesInputs

    # Add metadata as attributes
    ds.attrs["source_software"] = "VIA-tracks"
    ds.attrs["source_file"] = via_file.path.as_posix()

    logger.info(f"Loaded bounding boxes tracks from {via_file.path}:\n{ds}")
    return ds


def _pre_parse_df_columns(
    df: pd.DataFrame, frame_regexp: str = DEFAULT_FRAME_REGEXP
) -> pd.DataFrame:
    """Pre-parse dataframe columns to improve performance.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to pre-parse.
    frame_regexp : str, optional
        The regular expression to extract the frame number from the filename.

    """
    # Do I need to copy?
    # df = df.copy()

    # Parse region_shape_attributes: x, y, width, height
    if "region_shape_attributes" in df.columns:
        df_dicts = df["region_shape_attributes"].apply(ast.literal_eval)
        df["region_shape_attributes_x"] = df_dicts.apply(lambda d: d.get("x"))
        df["region_shape_attributes_y"] = df_dicts.apply(lambda d: d.get("y"))
        df["region_shape_attributes_width"] = df_dicts.apply(
            lambda d: d.get("width")
        )
        df["region_shape_attributes_height"] = df_dicts.apply(
            lambda d: d.get("height")
        )
        df = df.drop(columns=["region_shape_attributes"])

    # Parse region_attributes: track, confidence
    if "region_attributes" in df.columns:
        df_dicts = df["region_attributes"].apply(ast.literal_eval)
        df["region_attributes_track"] = df_dicts.apply(
            lambda d: d.get("track")
        )
        df["region_attributes_confidence"] = df_dicts.apply(
            lambda d: d.get("confidence")
        )
        df = df.drop(columns=["region_attributes"])

    # Parse file_attributes: frame may be in file_attributes or in filename
    if "file_attributes" in df.columns:
        # Convert file_attributes data to dictionaries
        df_dicts = df["file_attributes"].apply(ast.literal_eval)

        # Check if frame is in file_attributes for all files
        if all(["frame" in d for d in df_dicts]):
            df["frame"] = df_dicts.apply(lambda d: d.get("frame"))
        # Else extract frame number from filename
        else:
            df["frame"] = df["filename"].str.extract(
                frame_regexp, expand=False
            )
            # df["frame"] = df["filename"].apply(
            #     lambda f: int(re.search(frame_regexp, f).group(1))
            #     if re.search(frame_regexp, f)
            #     else np.nan
            # )
        df = df.drop(columns=["file_attributes"])

    return df


def _numpy_arrays_from_via_tracks_file(
    file_path: Path, frame_regexp: str = DEFAULT_FRAME_REGEXP
) -> dict:
    """Extract numpy arrays from the input VIA tracks .csv file.

    The extracted numpy arrays are returned in a dictionary with the following
    keys:

    - position_array (n_frames, n_space, n_individuals):
        contains the trajectories of the bounding box centroids.
    - shape_array (n_frames, n_space, n_individuals):
        contains the shape of the bounding boxes (width and height).
    - confidence_array (n_frames, n_individuals):
        contains the confidence score of each bounding box.
        If no confidence scores are provided, they are set to an array of NaNs.
    - ID_array (n_individuals, 1):
        contains the integer IDs of the tracked bounding boxes.
    - frame_array (n_frames, 1):
        contains the frame numbers.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the VIA tracks .csv file containing the bounding box tracks.

    frame_regexp : str
        Regular expression pattern to extract the frame number from the frame
        filename. By default, the frame number is expected to be encoded in
        the filename as an integer number led by at least one zero, followed
        by the file extension.

    Returns
    -------
    dict
        The validated bounding boxes arrays.

    """
    # Extract 2D dataframe from input data
    # (sort data by ID and frame number, and
    # fill empty frame-ID pairs with nans)
    df = _df_from_via_tracks_file(file_path, frame_regexp)

    # Compute indices of the rows where the IDs switch
    bool_id_diff_from_prev = df["ID"].ne(df["ID"].shift())  # pandas series
    indices_id_switch = (
        bool_id_diff_from_prev.loc[lambda x: x].index[1:].to_numpy()
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
            indices_id_switch,  # indices along axis=0
        )
        array_dict[key] = np.stack(list_arrays, axis=-1)

        # squeeze only last dimension if it is 1
        if array_dict[key].shape[1] == 1:
            array_dict[key] = array_dict[key].squeeze(axis=1)

    # Transform position_array to represent centroid of bbox,
    # rather than top-left corner
    # (top left corner: corner of the bbox with minimum x and y coordinates)
    array_dict["position_array"] += array_dict["shape_array"] / 2

    # Add remaining arrays to dict
    array_dict["ID_array"] = df["ID"].unique().reshape(-1, 1)
    array_dict["frame_array"] = df["frame_number"].unique().reshape(-1, 1)

    return array_dict


def _df_from_via_tracks_file(
    file_path: Path, frame_regexp: str = DEFAULT_FRAME_REGEXP
) -> pd.DataFrame:
    """Load VIA tracks .csv file as a dataframe.

    Read the VIA tracks .csv file as a pandas dataframe with columns:
    - ID: the integer ID of the tracked bounding box.
    - frame_number: the frame number of the tracked bounding box.
    - x: the x-coordinate of the tracked bounding box's top-left corner.
    - y: the y-coordinate of the tracked bounding box's top-left corner.
    - w: the width of the tracked bounding box.
    - h: the height of the tracked bounding box.
    - confidence: the confidence score of the tracked bounding box.

    The dataframe is sorted by ID and frame number, and for each ID,
    empty frames are filled in with NaNs. The coordinates of the bboxes
    are assumed to be in the image coordinate system (i.e., the top-left
    corner of a bbox is its corner with minimum x and y coordinates).

    The frame number is extracted from the filename using the provided
    regexp if it is not defined as a 'file_attribute' in the VIA tracks .csv
    file.
    """
    # Read VIA tracks .csv file as a pandas dataframe
    df_input = pd.read_csv(file_path, sep=",", header=0)
    # df_file = pd.read_parquet(file_path)  # engine

    # Pre-parse dataframe if not already
    if "region_shape_attributes_x" not in df_input.columns:
        logger.info(
            "Converting to optimized format "
            "(this may take a few minutes for large files)..."
        )
        df_input = _pre_parse_df_columns(df_input, frame_regexp)
        logger.info("Conversion complete.")

    # Map columns to desired names
    map_input_to_output_cols = {
        "region_attributes_track": "ID",
        "frame": "frame_number",
        "region_shape_attributes_x": "x",
        "region_shape_attributes_y": "y",
        "region_shape_attributes_width": "w",
        "region_shape_attributes_height": "h",
    }
    df = (
        df_input[list(map_input_to_output_cols.keys())]
        .rename(columns=map_input_to_output_cols)
        .copy()
    )

    # Apply type conversions
    df["ID"] = df["ID"].astype(int)
    df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce")
    df[["x", "y", "w", "h"]] = df[["x", "y", "w", "h"]].astype(float)

    # Handle confidence column and fill with nan if empty
    df["confidence"] = df_input.get("region_attributes_confidence", np.nan)

    # ----
    # Check if reindexing is needed
    if len(df) == len(df["ID"].unique()) * len(df["frame_number"].unique()):
        # Data is complete, just sort
        df = df.sort_values(by=["ID", "frame_number"], axis=0)
    else:
        # Define desired index: all combinations of ID and frame number
        multi_index = pd.MultiIndex.from_product(
            [df["ID"].unique().tolist(), df["frame_number"].unique().tolist()],
            # these unique lists may not be sorted!
            names=["ID", "frame_number"],
        )

        # Set index to (ID, frame number), fill in values with nans,
        # sort by ID and frame_number and reset to new index
        df = (
            df.set_index(["ID", "frame_number"])
            .reindex(multi_index)
            .reset_index()
        )
    # ----------
    return df
