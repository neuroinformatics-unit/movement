"""Load pose tracking data from various frameworks into ``movement``."""

import logging
from pathlib import Path
from typing import Literal, Union, Optional

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sleap_io.io.slp import read_labels
from sleap_io.model.labels import Labels

from movement.utils.logging import log_error, log_warning
from movement.validators.datasets import ValidPosesDataset
from movement.validators.files import (
    ValidAniposeCSV,
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
)

logger = logging.getLogger(__name__)


def from_numpy(
    position_array: np.ndarray,
    confidence_array: np.ndarray | None = None,
    individual_names: list[str] | None = None,
    keypoint_names: list[str] | None = None,
    fps: float | None = None,
    source_software: str | None = None,
) -> xr.Dataset:
    """Create a ``movement`` dataset from NumPy arrays.

    Parameters
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_space, n_keypoints, n_individuals)
        containing the poses. It will be converted to a
        :class:`xarray.DataArray` object named "position".
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_keypoints, n_individuals) containing
        the point-wise confidence scores. It will be converted to a
        :class:`xarray.DataArray` object named "confidence".
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
    >>> from movement.io import load_dataset
    >>> ds = load_dataset.from_numpy(
    ...     position_array=np.random.rand(100, 2, 3, 2),
    ...     confidence_array=np.ones((100, 3, 2)),
    ...     individual_names=["Alice", "Bob"],
    ...     keypoint_names=["snout", "centre", "tail_base"],
    ...     fps=30,
    ... )

    """
    valid_data = ValidPosesDataset(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software=source_software,
    )
    return _ds_from_valid_data(valid_data)


def from_file(
    file_path: Path | str,
    source_software: Literal[
        "DeepLabCut", "SLEAP", "LightningPose", "Anipose"
    ],
    fps: float | None = None,
    **kwargs,
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
    source_software : "DeepLabCut", "SLEAP", "LightningPose", or "Anipose"
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the software-specific
        loading functions that are listed under "See Also".

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    See Also
    --------
    movement.io.load_dataset.from_dlc_file
    movement.io.load_dataset.from_sleap_file
    movement.io.load_dataset.from_lp_file
    movement.io.load_dataset.from_anipose_file

    Examples
    --------
    >>> from movement.io import load_dataset
    >>> ds = load_dataset.from_file(
    ...     "path/to/file.h5", source_software="DeepLabCut", fps=30
    ... )

    """
    if source_software == "DeepLabCut":
        return from_dlc_file(file_path, fps)
    elif source_software == "SLEAP":
        return from_sleap_file(file_path, fps)
    elif source_software == "LightningPose":
        return from_lp_file(file_path, fps)
    elif source_software == "Anipose":
        return from_anipose_file(file_path, fps, **kwargs)
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
        the ``time`` coordinates will be in frame numbers.
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
    movement.io.load_dataset.from_dlc_file

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
    # reshape the data into (n_frames, 3, n_keypoints, n_individuals)
    # where the second axis contains "x", "y", "likelihood"
    tracks_with_scores = (
        df.to_numpy()
        .reshape((-1, len(individual_names), len(keypoint_names), 3))
        .transpose(0, 3, 2, 1)
    )
    return from_numpy(
        position_array=tracks_with_scores[:, :-1, :, :],
        confidence_array=tracks_with_scores[:, -1, :, :],
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
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    The SLEAP predictions are normally saved in .slp files, e.g.
    "v1.predictions.slp". An analysis file, suffixed with ".h5" can be exported
    from the SLEAP GUI (File > Export Analysis HDF5...). This is the
    recommended format for loading data into ``movement``.

    See Also
    --------
    movement.io.load_dataset.from_file : Load data from any supported file format.

    Examples
    --------
    >>> from movement.io import load_dataset
    >>> ds = load_dataset.from_sleap_file("path/to/file.analysis.h5", fps=30)

    """
    file = _validate_file_path(file_path, expected_suffix=[".h5", ".slp"])
    if file.suffix == ".h5":
        return _ds_from_sleap_analysis_file(file.path, fps)
    else:  # at this point it can only be .slp (because of validation)
        return _ds_from_sleap_labels_file(file.path, fps)


def from_lp_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a LightningPose file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the LightningPose predictions in .csv
        format.
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
    movement.io.load_dataset.from_file : Load data from any supported file format.

    Examples
    --------
    >>> from movement.io import load_dataset
    >>> ds = load_dataset.from_lp_file("path/to/file.csv", fps=30)

    """
    file = _validate_file_path(file_path, expected_suffix=[".csv"])
    return _ds_from_lp_or_dlc_file(file.path, "LightningPose", fps)


def from_dlc_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a DeepLabCut file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the DeepLabCut predictions in .h5
        or .csv format.
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
    movement.io.load_dataset.from_file : Load data from any supported file format.

    Examples
    --------
    >>> from movement.io import load_dataset
    >>> ds = load_dataset.from_dlc_file("path/to/file.h5", fps=30)

    """
    file = _validate_file_path(file_path, expected_suffix=[".h5", ".csv"])
    return _ds_from_lp_or_dlc_file(file.path, "DeepLabCut", fps)


def from_multiview_files(
    file_path_dict: dict[str, Path | str],
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` dataset from multiple camera views.

    Parameters
    ----------
    file_path_dict : dict
        Dictionary mapping camera view names to file paths. The file paths
        must be among those supported by the ``from_dlc_file()``,
        ``from_slp_file()`` or ``from_lp_file()`` functions.
    source_software : "DeepLabCut", "SLEAP", or "LightningPose"
        The source software of the files.
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
    movement.io.load_dataset.from_file : Load data from any supported file format.

    Examples
    --------
    >>> from movement.io import load_dataset
    >>> ds = load_dataset.from_multiview_files(
    ...     file_path_dict={
    ...         "front": "path/to/front.csv",
    ...         "side": "path/to/side.csv",
    ...     },
    ...     source_software="DeepLabCut",
    ...     fps=30,
    ... )

    """
    if source_software == "DeepLabCut":
        return _ds_from_multiview_dlc_files(file_path_dict, fps)
    elif source_software == "SLEAP":
        return _ds_from_multiview_sleap_files(file_path_dict, fps)
    elif source_software == "LightningPose":
        return _ds_from_multiview_lp_files(file_path_dict, fps)
    else:
        raise log_error(
            ValueError, f"Unsupported source software: {source_software}"
        )


def _ds_from_lp_or_dlc_file(
    file_path: Path | str,
    source_software: Literal["LightningPose", "DeepLabCut"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` dataset from a LightningPose or DeepLabCut file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the predictions.
    source_software : "LightningPose" or "DeepLabCut"
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    file = Path(file_path)
    if file.suffix == ".csv":
        df = _df_from_dlc_csv(file)
    else:  # at this point it can only be .h5 (because of validation)
        df = _df_from_dlc_h5(file)
    return from_dlc_style_df(df, fps, source_software)


def _ds_from_sleap_analysis_file(
    file_path: Path, fps: float | None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a SLEAP analysis file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the file containing the SLEAP predictions in .h5
        (analysis) format.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    with h5py.File(file_path, "r") as f:
        # read names of individuals and keypoints
        individual_names = f["track_occupancy"].dtype.names
        keypoint_names = f["point_scores"].dtype.names

        # read pose tracks and confidence scores
        tracks = f["tracks"][:]  # shape: (n_frames, n_individuals, n_keypoints, 2)
        scores = f["point_scores"][:]  # shape: (n_frames, n_individuals, n_keypoints)

        # reshape the data into (n_frames, 2, n_keypoints, n_individuals)
        tracks = tracks.transpose(0, 3, 2, 1)
        scores = scores.transpose(0, 2, 1)

        return from_numpy(
            position_array=tracks,
            confidence_array=scores,
            individual_names=individual_names,
            keypoint_names=keypoint_names,
            fps=fps,
            source_software="SLEAP",
        )


def _ds_from_sleap_labels_file(
    file_path: Path, fps: float | None
) -> xr.Dataset:
    """Create a ``movement`` dataset from a SLEAP labels file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the file containing the SLEAP predictions in .slp
        (labels) format.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    This function is experimental and may not work with all SLEAP labels
    files. It is recommended to export the predictions as an analysis file
    (File > Export Analysis HDF5...) and use
    :func:`movement.io.load_dataset.from_sleap_file` instead.

    """
    labels = read_labels(file_path)
    return _sleap_labels_to_numpy(labels)


def _sleap_labels_to_numpy(labels: Labels) -> np.ndarray:
    """Convert SLEAP labels to a ``movement`` dataset.

    Parameters
    ----------
    labels : sleap_io.model.labels.Labels
        SLEAP labels object.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    # read names of individuals and keypoints
    individual_names = [track.name for track in labels.tracks]
    keypoint_names = [node.name for node in labels.skeleton.nodes]

    # read pose tracks and confidence scores
    tracks = np.zeros(
        (len(labels), len(individual_names), len(keypoint_names), 2)
    )
    scores = np.zeros((len(labels), len(individual_names), len(keypoint_names)))

    for frame_idx, frame in enumerate(labels):
        for instance in frame.instances:
            track_idx = individual_names.index(instance.track.name)
            for point_idx, point in enumerate(instance.points):
                keypoint_idx = keypoint_names.index(point.node.name)
                tracks[frame_idx, track_idx, keypoint_idx] = point.coordinates
                scores[frame_idx, track_idx, keypoint_idx] = point.confidence

    # reshape the data into (n_frames, 2, n_keypoints, n_individuals)
    tracks = tracks.transpose(0, 3, 2, 1)
    scores = scores.transpose(0, 2, 1)

    return from_numpy(
        position_array=tracks,
        confidence_array=scores,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=None,  # fps is not available in SLEAP labels files
        source_software="SLEAP",
    )


def _df_from_dlc_csv(file_path: Path) -> pd.DataFrame:
    """Create a DeepLabCut-style DataFrame from a .csv file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the file containing the DeepLabCut predictions in .csv
        format.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style DataFrame containing the pose tracks and
        confidence scores.

    """
    df = pd.read_csv(file_path)
    return ValidDeepLabCutCSV(df)


def _df_from_dlc_h5(file_path: Path) -> pd.DataFrame:
    """Create a DeepLabCut-style DataFrame from a .h5 file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the file containing the DeepLabCut predictions in .h5
        format.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style DataFrame containing the pose tracks and
        confidence scores.

    """
    df = pd.read_hdf(file_path, key="df_with_missing")
    return ValidDeepLabCutCSV(df)


def _ds_from_valid_data(data: ValidPosesDataset) -> xr.Dataset:
    """Create a ``movement`` dataset from a validated dataset.

    Parameters
    ----------
    data : movement.validators.datasets.ValidPosesDataset
        Validated dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    # create the dataset
    ds = xr.Dataset(
        data_vars={
            "position": (
                ["time", "space", "keypoints", "individuals"],
                data.position_array,
            ),
            "confidence": (
                ["time", "keypoints", "individuals"],
                data.confidence_array,
            ),
        },
        coords={
            "time": np.arange(data.position_array.shape[0], dtype=int),
            "space": ["x", "y"],
            "keypoints": data.keypoint_names,
            "individuals": data.individual_names,
        },
        attrs={
            "fps": data.fps,
            "source_software": data.source_software,
        },
    )

    # add time coordinates in seconds if fps is provided
    if data.fps is not None:
        ds.coords["time"] = ds.coords["time"] / data.fps

    return ds


def from_anipose_style_df(
    df: pd.DataFrame,
    fps: float | None = None,
    individual_name: str = "individual_0",
) -> xr.Dataset:
    """Create a ``movement`` dataset from an Anipose-style DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the pose tracks and confidence scores. Must
        be formatted as in Anipose output files (see Notes).
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
    individual_name : str, optional
        Name of the individual in the video. Defaults to "individual_0".

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    The DataFrame must have a multi-index column with the following levels:
    "scorer", "bodyparts", "coords". The "coords" level contains the spatial
    coordinates "x", "y", "z", as well as "likelihood" (point-wise confidence
    scores). The row index corresponds to the frame number.

    See Also
    --------
    movement.io.load_dataset.from_anipose_file : Load data from an Anipose file.

    """
    # read names of keypoints from the DataFrame
    keypoint_names = (
        df.columns.get_level_values("bodyparts").unique().to_list()
    )
    # reshape the data into (n_frames, 3, n_keypoints, 1)
    # where the second axis contains "x", "y", "z"
    tracks_with_scores = (
        df.to_numpy()
        .reshape((-1, len(keypoint_names), 4))
        .transpose(0, 2, 1, np.newaxis)
    )
    return from_numpy(
        position_array=tracks_with_scores[:, :-1, :, :],
        confidence_array=tracks_with_scores[:, -1, :, :],
        individual_names=[individual_name],
        keypoint_names=keypoint_names,
        fps=fps,
        source_software="Anipose",
    )


def from_anipose_file(
    file_path: Union[str, Path],
    fps: Optional[float] = None,
    individual_name: Optional[str] = None,
) -> MovementDataset:
    """Load pose tracks from an Anipose CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the Anipose CSV file.
    fps : float, optional
        Frames per second. If not provided, will be inferred from the data.
    individual_name : str, optional
        Name of the individual. If not provided, will be inferred from the data.

    Returns
    -------
    MovementDataset
        A movement dataset containing the pose tracks.

    See Also
    --------
    from_anipose_style_df : Load pose tracks from an Anipose-style DataFrame.
    """
    file = _validate_file_path(file_path, expected_suffix=[".csv"])
    df = pd.read_csv(file.path)
    
    return from_anipose_style_df(df, fps, individual_name) 