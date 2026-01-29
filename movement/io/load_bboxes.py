"""Load bounding boxes tracking data into ``movement``."""

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
    bboxes_arrays = _numpy_arrays_from_valid_file_object(via_file)
    ds = from_numpy(
        position_array=bboxes_arrays["position_array"],
        shape_array=bboxes_arrays["shape_array"],
        confidence_array=bboxes_arrays["confidence_array"],
        individual_names=[
            f"id_{id}" for id in bboxes_arrays["ID_array"].flatten()
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


def _numpy_arrays_from_valid_file_object(
    valid_via_file: ValidVIATracksCSV,
) -> dict:
    """Extract numpy arrays from VIA tracks file object.

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
    valid_via_file : ValidVIATracksCSV
        A validated VIA tracks file object.

    Returns
    -------
    dict
        The validated bounding boxes arrays.

    """
    # Extract 2D dataframe from input data
    # (sort data by ID and frame number, and
    # fill empty frame-ID pairs with nans)
    df = _parsed_df_from_valid_file_object(valid_via_file)

    # Extract numpy arrays
    n_individuals = df["ID"].nunique()
    n_frames = df["frame_number"].nunique()
    all_data = df[["x", "y", "w", "h", "confidence"]].to_numpy(copy=True)

    array_dict: dict[str, np.ndarray] = {}
    array_dict["position_array"] = (
        all_data[:, 0:2]  # x,y
        .reshape(n_individuals, n_frames, 2)
        .transpose(1, 2, 0)
    )
    array_dict["shape_array"] = (
        all_data[:, 2:4]  # w,h
        .reshape(n_individuals, n_frames, 2)
        .transpose(1, 2, 0)
    )
    array_dict["confidence_array"] = (
        all_data[:, 4]  # confidence
        .reshape(n_individuals, n_frames)
        .transpose()
    )

    # Transform position_array to represent centroid of bbox,
    # rather than top-left corner
    # (top left corner: corner of the bbox with minimum x and y coordinates)
    array_dict["position_array"] += array_dict["shape_array"] / 2

    # Add remaining arrays to dict
    array_dict["ID_array"] = df["ID"].unique().reshape(-1, 1)
    array_dict["frame_array"] = df["frame_number"].unique().reshape(-1, 1)

    return array_dict


def _parsed_df_from_valid_file_object(
    valid_via_file: ValidVIATracksCSV,
) -> pd.DataFrame:
    """Build a sorted DataFrame from a validated VIA file object.

    Creates a DataFrame with ID, frame_number, x, y, w, h, and confidence
    columns. Missing (ID, frame_number) combinations are filled with NaNs,
    and the result is sorted by ID and frame_number.

    Parameters
    ----------
    valid_via_file : ValidVIATracksCSV
        A validated VIA tracks file object.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with all ID/frame combinations.

    """
    # Build dataframe from file validator object data, then sort and reindex
    df = pd.DataFrame(
        {
            "ID": valid_via_file.ids,
            "frame_number": valid_via_file.frame_numbers,
            "x": np.array(valid_via_file.x, dtype=np.float32),
            "y": np.array(valid_via_file.y, dtype=np.float32),
            "w": np.array(valid_via_file.w, dtype=np.float32),
            "h": np.array(valid_via_file.h, dtype=np.float32),
            "confidence": np.array(
                valid_via_file.confidence_values, dtype=np.float32
            ),
        }
    )

    # If every ID is defined for every frame:
    # just sort and reindex (does not add rows)
    if len(df) == df["ID"].nunique() * df["frame_number"].nunique():
        df = df.sort_values(
            by=["ID", "frame_number"],
            axis=0,
        ).reset_index(drop=True)

    # If some combinations of ID and frame number are missing:
    # fill with nan
    else:
        # Desired index: all combinations of ID and frame number
        multi_index = pd.MultiIndex.from_product(
            [df["ID"].unique().tolist(), df["frame_number"].unique().tolist()],
            # these unique lists may not be sorted!
            names=["ID", "frame_number"],
        )

        # Set index to (ID, frame number), fill in values with nans,
        # sort by ID and frame_number, and reset to new index
        df = (
            df.set_index(["ID", "frame_number"])
            .reindex(multi_index)  # fills missing rows with nan
            .sort_values(by=["ID", "frame_number"], axis=0)
            .reset_index()
        )
    return df
