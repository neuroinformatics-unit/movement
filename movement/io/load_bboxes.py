"""Load bounding boxes tracking data into ``movement``."""

import warnings
from pathlib import Path
from typing import Literal, cast

import numpy as np
import xarray as xr

from movement.io.load import register_loader
from movement.utils.logging import logger
from movement.validators.datasets import ValidBboxesInputs
from movement.validators.files import DEFAULT_FRAME_REGEXP, ValidVIATracksCSV


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
    position_array
        Array of shape (n_frames, n_space, n_individuals)
        containing the tracks of the bounding box centroids.
        It will be converted to a :class:`xarray.DataArray` object
        named "position".
    shape_array
        Array of shape (n_frames, n_space, n_individuals)
        containing the shape of the bounding boxes. The shape of a bounding
        box is its width (extent along the x-axis of the image) and height
        (extent along the y-axis of the image). It will be converted to a
        :class:`xarray.DataArray` object named "shape".
    confidence_array
        Array of shape (n_frames, n_individuals) containing
        the confidence scores of the bounding boxes. If None (default), the
        confidence scores are set to an array of NaNs. It will be converted
        to a :class:`xarray.DataArray` object named "confidence".
    individual_names
        List of individual names for the tracked bounding boxes in the video.
        If None (default), bounding boxes are assigned names based on the size
        of the ``position_array``. The names will be in the format of
        ``id_<N>``, where <N>  is an integer from 0 to
        ``position_array.shape[-1]-1`` (i.e., "id_0", "id_1"...).
    frame_array
        Array of shape (n_frames, 1) containing the frame numbers for which
        bounding boxes are defined. If None (default), frame numbers will
        be assigned based on the first dimension of the ``position_array``,
        starting from 0. If a specific array of frame numbers is provided,
        these need to be integers sorted in increasing order.
    fps
        The video sampling rate. If None (default), the ``time`` coordinates
        of the resulting ``movement`` dataset will be in frame numbers. If
        ``fps`` is provided, the ``time`` coordinates  will be in seconds. If
        the ``time`` coordinates are in seconds, they will indicate the
        elapsed time from the capture of the first frame (assumed to be frame
        0).
    source_software
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
    file: Path | str,
    source_software: Literal["VIA-tracks"],
    fps: float | None = None,
    use_frame_numbers_from_file: bool = False,
    frame_regexp: str = DEFAULT_FRAME_REGEXP,
) -> xr.Dataset:
    """Create a ``movement`` bounding boxes dataset from a supported file.

    .. deprecated:: 0.14.0
        This function is deprecated and will be removed in a future release.
        Use :func:`movement.io.load_dataset<movement.io.load.load_dataset>`
        instead.

    At the moment, we only support VIA tracks .csv files.

    Parameters
    ----------
    file
        Path to the file containing the tracked bounding boxes. Currently
        only VIA tracks .csv files are supported.
    source_software
        The source software of the file. Currently only files from the
        VIA 2.0.12 annotator [1]_ ("VIA-tracks") are supported.
        See .
    fps
        The video sampling rate. If None (default), the ``time`` coordinates
        of the resulting ``movement`` dataset will be in frame numbers. If
        ``fps`` is provided, the ``time`` coordinates  will be in seconds. If
        the ``time`` coordinates are in seconds, they will indicate the
        elapsed time from the capture of the first frame (assumed to be frame
        0).
    use_frame_numbers_from_file
        If True, the frame numbers in the resulting dataset are
        the same as the ones specified for each tracked bounding box in the
        input file. This may be useful if the bounding boxes are tracked for a
        subset of frames in a video, but you want to maintain the start of the
        full video as the time origin. If False (default), the frame numbers
        in the VIA tracks .csv file are instead mapped to a 0-based sequence of
        consecutive integers.
    frame_regexp
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
    warnings.warn(
        "The function `movement.io.load_bboxes.from_file` is deprecated"
        " and will be removed in a future release. "
        "Please use `movement.io.load_dataset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if source_software == "VIA-tracks":
        return from_via_tracks_file(
            file,
            fps,
            use_frame_numbers_from_file=use_frame_numbers_from_file,
            frame_regexp=frame_regexp,
        )
    else:
        valid_options = ["VIA-tracks"]
        raise logger.error(
            ValueError(
                f"Unsupported source software: '{source_software}'. "
                f"Supported options are: {', '.join(valid_options)}"
            )
        )


@register_loader("VIA-tracks", file_validators=[ValidVIATracksCSV])
def from_via_tracks_file(
    file: str | Path,
    fps: float | None = None,
    use_frame_numbers_from_file: bool = False,
    frame_regexp: str = DEFAULT_FRAME_REGEXP,
) -> xr.Dataset:
    """Create a ``movement`` dataset from a VIA tracks .csv file.

    Parameters
    ----------
    file
        Path to the VIA tracks .csv file with the tracked bounding boxes.
        For more information on the VIA tracks .csv file format, see the VIA
        tutorial for tracking [1]_.
    fps
        The video sampling rate. If None (default), the ``time`` coordinates
        of the resulting ``movement`` dataset will be in frame numbers. If
        ``fps`` is provided, the ``time`` coordinates  will be in seconds. If
        the ``time`` coordinates are in seconds, they will indicate the
        elapsed time from the capture of the first frame (assumed to be frame
        0).
    use_frame_numbers_from_file
        If True, the frame numbers in the resulting dataset are
        the same as the ones in the VIA tracks .csv file. This may be useful if
        the bounding boxes are tracked for a subset of frames in a video,
        but you want to maintain the start of the full video as the time
        origin. If False (default), the frame numbers in the VIA tracks .csv
        file are instead mapped to a 0-based sequence of consecutive integers.
    frame_regexp
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
    # Note: the @register_loader decorator has already validated
    # `file` as a ValidVIATracksCSV, and forwarded `frame_regexp`
    # to it via _get_validator_kwargs.
    valid_file = cast("ValidVIATracksCSV", file)

    # Create an xarray.Dataset from the data
    bboxes_arrays = _numpy_arrays_from_valid_via_object(valid_file)
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
    file_path = valid_file.file
    ds.attrs["source_software"] = "VIA-tracks"
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded bounding boxes tracks from {file_path}:\n{ds}")
    return ds


def _numpy_arrays_from_valid_via_object(
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
    valid_via_file
        A validated VIA tracks file object.

    Returns
    -------
    dict
        The validated bounding boxes arrays.

    """
    # Get 1D data from the validator
    x = valid_via_file.x
    y = valid_via_file.y
    w = valid_via_file.w
    h = valid_via_file.h
    ids = valid_via_file.ids
    frame_numbers = valid_via_file.frame_numbers
    confidence = valid_via_file.confidence

    # Compute **sorted** unique IDs and frames
    unique_ids = np.unique(ids)
    unique_frames = np.unique(frame_numbers)

    # Map each observation's ID and frame to an index in the output arrays
    id_indices_per_obs = np.searchsorted(unique_ids, ids)
    frame_indices_per_obs = np.searchsorted(unique_frames, frame_numbers)

    # Initialise output dense arrays and fill with NaNs
    n_individuals = len(unique_ids)
    n_frames = len(unique_frames)
    position_array = np.full(
        (n_frames, 2, n_individuals), np.nan, dtype=np.float32
    )
    shape_array = np.full(
        (n_frames, 2, n_individuals), np.nan, dtype=np.float32
    )
    confidence_array = np.full(
        (n_frames, n_individuals), np.nan, dtype=np.float32
    )

    # Place sparse values directly into the output
    # Position = centroid = top-left corner + half the bbox size
    position_array[frame_indices_per_obs, 0, id_indices_per_obs] = x + w / 2
    position_array[frame_indices_per_obs, 1, id_indices_per_obs] = y + h / 2
    shape_array[frame_indices_per_obs, 0, id_indices_per_obs] = w
    shape_array[frame_indices_per_obs, 1, id_indices_per_obs] = h
    confidence_array[frame_indices_per_obs, id_indices_per_obs] = confidence

    return {
        "position_array": position_array,
        "shape_array": shape_array,
        "confidence_array": confidence_array,
        "ID_array": unique_ids.reshape(-1, 1),
        "frame_array": unique_frames.reshape(-1, 1),
    }
