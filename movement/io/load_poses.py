"""Load pose tracking data from various frameworks into ``movement``."""

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import pynwb
import xarray as xr
from sleap_io.io.slp import read_labels
from sleap_io.model.labels import Labels

from movement.utils.logging import logger
from movement.validators.datasets import ValidPosesDataset
from movement.validators.files import (
    ValidAniposeCSV,
    ValidDeepLabCutCSV,
    ValidFile,
    ValidHDF5,
    ValidNWBFile,
)


def from_numpy(
    position_array: np.ndarray,
    confidence_array: np.ndarray | None = None,
    individual_names: list[str] | None = None,
    keypoint_names: list[str] | None = None,
    fps: float | None = None,
    source_software: str | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from NumPy arrays.

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
        (default), the individuals will be named "id_0", "id_1", etc.
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
    >>> rng = np.random.default_rng(seed=42)
    >>> ds = load_poses.from_numpy(
    ...     position_array=rng.random((100, 2, 3, 2)),
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
        "DeepLabCut",
        "SLEAP",
        "LightningPose",
        "Anipose",
        "NWB",
    ],
    fps: float | None = None,
    **kwargs,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from any supported file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing predicted poses. The file format must
        be among those supported by the ``from_dlc_file()``,
        ``from_slp_file()`` or ``from_lp_file()`` functions. One of these
        these functions will be called internally, based on
        the value of ``source_software``.
    source_software : {"DeepLabCut", "SLEAP", "LightningPose", "Anipose", \
        "NWB"}
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
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
    movement.io.load_poses.from_dlc_file
    movement.io.load_poses.from_sleap_file
    movement.io.load_poses.from_lp_file
    movement.io.load_poses.from_anipose_file
    movement.io.load_poses.from_nwb_file

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
    elif source_software == "Anipose":
        return from_anipose_file(file_path, fps, **kwargs)
    elif source_software == "NWB":
        if fps is not None:
            logger.warning(
                "The fps argument is ignored when loading from an NWB file. "
                "The frame rate will be directly read or estimated from "
                "metadata in the file."
            )
        return from_nwb_file(file_path, **kwargs)
    else:
        raise logger.error(
            ValueError(f"Unsupported source software: {source_software}")
        )


def from_dlc_style_df(
    df: pd.DataFrame,
    fps: float | None = None,
    source_software: Literal["DeepLabCut", "LightningPose"] = "DeepLabCut",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a DeepLabCut-style DataFrame.

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
    """Create a ``movement`` poses dataset from a SLEAP file.

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
    logger.info(f"Loaded pose tracks from {file.path}:\n{ds}")
    return ds


def from_lp_file(
    file_path: Path | str, fps: float | None = None
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a LightningPose file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the predicted poses, in .csv format.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

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
    """Create a ``movement`` poses dataset from a DeepLabCut file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the predicted poses, either in .h5
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
    movement.io.load_poses.from_dlc_style_df

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_dlc_file("path/to/file.h5", fps=30)

    """
    return _ds_from_lp_or_dlc_file(
        file_path=file_path, source_software="DeepLabCut", fps=fps
    )


def from_multiview_files(
    file_path_dict: dict[str, Path | str],
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    fps: float | None = None,
) -> xr.Dataset:
    """Load and merge pose tracking data from multiple views (cameras).

    Parameters
    ----------
    file_path_dict : dict[str, Union[Path, str]]
        A dict whose keys are the view names and values are the paths to load.
    source_software : {'LightningPose', 'SLEAP', 'DeepLabCut'}
        The source software of the file.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata, with an additional ``views`` dimension.

    """
    views_list = list(file_path_dict.keys())
    new_coord_views = xr.DataArray(views_list, dims="view")
    dataset_list = [
        from_file(f, source_software=source_software, fps=fps)
        for f in file_path_dict.values()
    ]
    return xr.concat(dataset_list, dim=new_coord_views)


def _ds_from_lp_or_dlc_file(
    file_path: Path | str,
    source_software: Literal["LightningPose", "DeepLabCut"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a LightningPose or DLC file.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file containing the predicted poses, either in .h5
        or .csv format.
    source_software : {'LightningPose', 'DeepLabCut'}
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
    expected_suffix = [".csv"]
    if source_software == "DeepLabCut":
        expected_suffix.append(".h5")
    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=expected_suffix
    )
    # Load the DeepLabCut poses into a DataFrame
    df = (
        _df_from_dlc_csv(file.path)
        if file.path.suffix == ".csv"
        else _df_from_dlc_h5(file.path)
    )
    logger.debug(f"Loaded poses from {file.path} into a DataFrame.")
    # Convert the DataFrame to an xarray dataset
    ds = from_dlc_style_df(df=df, fps=fps, source_software=source_software)
    # Add metadata as attrs
    ds.attrs["source_file"] = file.path.as_posix()
    logger.info(f"Loaded pose tracks from {file.path}:\n{ds}")
    return ds


def _ds_from_sleap_analysis_file(
    file_path: Path, fps: float | None
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a SLEAP analysis (.h5) file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the SLEAP analysis file containing predicted pose tracks.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    file = ValidHDF5(file_path, expected_datasets=["tracks"])
    with h5py.File(file.path, "r") as f:
        # Transpose to shape: (n_frames, n_space, n_keypoints, n_tracks)
        tracks = f["tracks"][:].transpose(3, 1, 2, 0)
        # Create an array of NaNs for the confidence scores
        scores = np.full(tracks.shape[:1] + tracks.shape[2:], np.nan)
        individual_names = [n.decode() for n in f["track_names"][:]] or None
        if individual_names is None:
            logger.warning(
                f"Could not find SLEAP Track in {file.path}. "
                "Assuming single-individual dataset and assigning "
                "default individual name."
            )
        # If present, read the point-wise scores,
        # and transpose to shape: (n_frames, n_keypoints, n_tracks)
        if "point_scores" in f:
            scores = f["point_scores"][:].T
        return from_numpy(
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
    """Create a ``movement`` poses dataset from a SLEAP labels (.slp) file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the SLEAP labels file containing predicted pose tracks.
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.

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
        logger.warning(
            f"Could not find SLEAP Track in {file.path}. "
            "Assuming single-individual dataset and assigning "
            "default individual name."
        )
    return from_numpy(
        position_array=tracks_with_scores[:, :-1, :, :],
        confidence_array=tracks_with_scores[:, -1, :, :],
        individual_names=individual_names,
        keypoint_names=[kp.name for kp in labels.skeletons[0].nodes],
        fps=fps,
        source_software="SLEAP",
    )


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
        A NumPy array containing pose tracks and confidence scores,
        with shape ``(n_frames, 3, n_nodes, n_tracks)``.

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
    tracks = np.full((n_frames, 3, n_nodes, n_tracks), np.nan, dtype="float32")

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
                tracks[i, ..., j] = np.hstack(
                    (inst.numpy(), np.full((n_nodes, 1), np.nan))
                ).T
            elif predicted_track_instances:
                inst = predicted_track_instances[-1]
                tracks[i, ..., j] = inst.numpy(scores=True).T
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


def _ds_from_valid_data(data: ValidPosesDataset) -> xr.Dataset:
    """Create a ``movement`` poses dataset from validated pose tracking data.

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
    n_space = data.position_array.shape[1]

    dataset_attrs: dict[str, str | float | None] = {
        "source_software": data.source_software,
        "ds_type": "poses",
    }
    # Create the time coordinate, depending on the value of fps
    if data.fps is not None:
        time_coords = np.arange(n_frames, dtype=float) / data.fps
        time_unit = "seconds"
        dataset_attrs["fps"] = data.fps
    else:
        time_coords = np.arange(n_frames, dtype=int)
        time_unit = "frames"

    dataset_attrs["time_unit"] = time_unit

    DIM_NAMES = ValidPosesDataset.DIM_NAMES
    # Convert data to an xarray.Dataset
    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(data.position_array, dims=DIM_NAMES),
            "confidence": xr.DataArray(
                data.confidence_array, dims=DIM_NAMES[:1] + DIM_NAMES[2:]
            ),
        },
        coords={
            DIM_NAMES[0]: time_coords,
            DIM_NAMES[1]: ["x", "y", "z"][:n_space],
            DIM_NAMES[2]: data.keypoint_names,
            DIM_NAMES[3]: data.individual_names,
        },
        attrs=dataset_attrs,
    )


def from_anipose_style_df(
    df: pd.DataFrame,
    fps: float | None = None,
    individual_name: str = "id_0",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an Anipose 3D dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Anipose triangulation dataframe
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.
    individual_name : str, optional
        Name of the individual, by default "id_0"

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.


    Notes
    -----
    Reshape dataframe with columns keypoint1_x, keypoint1_y, keypoint1_z,
    keypoint1_score,keypoint2_x, keypoint2_y, keypoint2_z,
    keypoint2_score...to array of positions with dimensions
    time, space, keypoints, individuals, and array of confidence (from scores)
    with dimensions time, keypoints, individuals.

    """
    keypoint_names = sorted(
        list(
            set(
                [
                    col.rsplit("_", 1)[0]
                    for col in df.columns
                    if any(col.endswith(f"_{s}") for s in ["x", "y", "z"])
                ]
            )
        )
    )

    n_frames = len(df)
    n_keypoints = len(keypoint_names)

    # Initialize arrays and fill
    position_array = np.zeros(
        (n_frames, 3, n_keypoints, 1)
    )  # 1 for single individual
    confidence_array = np.zeros((n_frames, n_keypoints, 1))
    for i, kp in enumerate(keypoint_names):
        for j, coord in enumerate(["x", "y", "z"]):
            position_array[:, j, i, 0] = df[f"{kp}_{coord}"]
        confidence_array[:, i, 0] = df[f"{kp}_score"]

    individual_names = [individual_name]

    return from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        source_software="Anipose",
        fps=fps,
    )


def from_anipose_file(
    file_path: Path | str,
    fps: float | None = None,
    individual_name: str = "id_0",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an Anipose 3D .csv file.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the Anipose triangulation .csv file
    fps : float, optional
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.
    individual_name : str, optional
        Name of the individual, by default "id_0"

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    We currently do not load all information, only x, y, z, and score
    (confidence) for each keypoint. Future versions will load n of cameras
    and error.

    """
    file = ValidFile(
        file_path,
        expected_permission="r",
        expected_suffix=[".csv"],
    )
    anipose_file = ValidAniposeCSV(file.path)
    anipose_df = pd.read_csv(anipose_file.path)

    return from_anipose_style_df(
        anipose_df, fps=fps, individual_name=individual_name
    )


def from_nwb_file(
    file: str | Path | pynwb.file.NWBFile,
    processing_module_key: str = "behavior",
    pose_estimation_key: str = "PoseEstimation",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an NWB file.

    The input can be a path to an NWB file on disk or a
    :class:`pynwb.file.NWBFile` object.
    The data will be extracted from the NWB file's
    :class:`pynwb.base.ProcessingModule`
    (specified by ``processing_module_key``) that contains the
    ``ndx_pose.PoseEstimation`` object (specified by ``pose_estimation_key``)
    formatted according to the ``ndx-pose`` NWB extension [1]_.

    Parameters
    ----------
    file : str | Path | pynwb.file.NWBFile
        Path to the NWB file on disk (ending in ".nwb"),
        or an NWBFile object.
    processing_module_key : str, optional
        Name of the :class:`pynwb.base.ProcessingModule` in the NWB file that
        contains the pose estimation data. Default is "behavior".
    pose_estimation_key: str, optional
        Name of the ``ndx_pose.PoseEstimation`` object in the processing
        module (specified by ``processing_module_key``).
        Default is "PoseEstimation".

    Returns
    -------
    xarray.Dataset
        A single-individual ``movement`` dataset containing the pose tracks,
        confidence scores, and associated metadata.

    References
    ----------
    .. [1] https://github.com/rly/ndx-pose

    Examples
    --------
    Open an NWB file and load pose tracks from the
    :class:`pynwb.file.NWBFile` object:

    >>> import pynwb
    >>> import xarray as xr
    >>> from movement.io import load_poses
    >>> with pynwb.NWBHDF5IO("path/to/file.nwb", mode="r") as io:
    ...     nwb_file = io.read()
    ...     ds = load_poses.from_nwb_file(nwb_file)

    Or, directly load pose tracks from an NWB file on disk:

    >>> ds = load_poses.from_nwb_file("path/to/file.nwb")

    Load two single-individual datasets from two NWB files and merge them
    into a multi-individual dataset:

    >>> ds_singles = [
    ...     load_poses.from_nwb_file(f) for f in ["id1.nwb", "id2.nwb"]
    ... ]
    >>> ds_multi = xr.merge(ds_singles)

    """
    file = ValidNWBFile(file).file
    if isinstance(file, Path):
        with pynwb.NWBHDF5IO(file, mode="r") as io:
            nwbfile_object = io.read()
            ds = _ds_from_nwb_object(
                nwbfile_object, processing_module_key, pose_estimation_key
            )
            ds.attrs["source_file"] = file
    else:  # file is an NWBFile object
        ds = _ds_from_nwb_object(
            file, processing_module_key, pose_estimation_key
        )
    return ds


def _ds_from_nwb_object(
    nwb_file: pynwb.file.NWBFile,
    processing_module_key: str = "behavior",
    pose_estimation_key: str = "PoseEstimation",
) -> xr.Dataset:
    """Extract a ``movement`` poses dataset from an NWBFile object.

    Parameters
    ----------
    nwb_file : pynwb.file.NWBFile
        An NWBFile object.
    processing_module_key : str, optional
        Name of the :class:`pynwb.base.ProcessingModule` in the NWB file that
        contains the pose estimation data. Default is "behavior".
    pose_estimation_key: str, optional
        Name of the ``ndx_pose.PoseEstimation`` object in the processing
        module (specified by ``processing_module_key``).
        Default is "PoseEstimation".

    Returns
    -------
    xarray.Dataset
        A single-individual ``movement`` poses dataset

    """
    pose_estimation = nwb_file.processing[processing_module_key][
        pose_estimation_key
    ]
    source_software = pose_estimation.source_software
    pose_estimation_series = pose_estimation.pose_estimation_series
    single_keypoint_datasets = []
    for keypoint, pes in pose_estimation_series.items():
        # Extract position and confidence data for each keypoint
        position_data = np.asarray(pes.data)  # shape: (n_frames, n_space)
        confidence_data = (  # shape: (n_frames,)
            np.asarray(pes.confidence)
            if getattr(pes, "confidence", None) is not None
            else np.full(position_data.shape[0], np.nan)
        )
        # Compute fps from time differences between timestamps
        # if rate is not available
        fps = pes.rate or float(np.nanmedian(1 / np.diff(pes.timestamps)))
        single_keypoint_datasets.append(
            # create movement dataset with 1 keypoint and 1 individual
            from_numpy(
                position_data[:, :, np.newaxis, np.newaxis],
                confidence_data[:, np.newaxis, np.newaxis],
                individual_names=[nwb_file.identifier],
                keypoint_names=[keypoint],
                fps=round(fps, 6),
                source_software=source_software,
            )
        )
    return xr.merge(single_keypoint_datasets)
