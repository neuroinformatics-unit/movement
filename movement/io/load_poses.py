"""Load pose tracking data from various frameworks into ``movement``."""

import warnings
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
import pandas as pd
import pynwb
import xarray as xr
from sleap_io.io.slp import read_labels
from sleap_io.model.labels import Labels

from movement.io.load import register_loader
from movement.utils.logging import logger
from movement.validators.datasets import ValidPosesInputs
from movement.validators.files import (
    ValidAniposeCSV,
    ValidBVHFile,
    ValidCOCOJSON,
    ValidDeepLabCutCSV,
    ValidDeepLabCutH5,
    ValidFile,
    ValidNWBFile,
    ValidSleapAnalysis,
    ValidSleapLabels,
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
    position_array
        Array of shape (n_frames, n_space, n_keypoints, n_individuals)
        containing the poses. It will be converted to a
        :class:`xarray.DataArray` object named "position".
    confidence_array
        Array of shape (n_frames, n_keypoints, n_individuals) containing
        the point-wise confidence scores. It will be converted to a
        :class:`xarray.DataArray` object named "confidence".
        If None (default), the scores will be set to an array of NaNs.
    individual_names
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "id_0", "id_1", etc.
    keypoint_names
        List of unique names for the keypoints in the skeleton. If None
        (default), the keypoints will be named "keypoint_0", "keypoint_1",
        etc.
    fps
        Frames per second of the video. Defaults to None, in which case
        the time coordinates will be in frame numbers.
    source_software
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
    valid_poses_inputs = ValidPosesInputs(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software=source_software,
    )
    return valid_poses_inputs.to_dataset()


def from_file(
    file: Path | str,
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

    .. deprecated:: 0.14.0
        This function is deprecated and will be removed in a future release.
        Use :func:`movement.io.load_dataset<movement.io.load.load_dataset>`
        instead.

    Parameters
    ----------
    file
        Path to the file containing predicted poses. The file format must
        be among those supported by the ``from_dlc_file()``,
        ``from_slp_file()`` or ``from_lp_file()`` functions. One of these
        these functions will be called internally, based on
        the value of ``source_software``.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
        This argument is ignored when ``source_software`` is "NWB", as the
        frame rate will be directly read or estimated from metadata in
        the NWB file.
    **kwargs
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
    warnings.warn(
        "The function `movement.io.load_poses.from_file` is deprecated"
        " and will be removed in a future release. "
        "Please use `movement.io.load_dataset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if source_software == "DeepLabCut":
        return from_dlc_file(file, fps)
    elif source_software == "SLEAP":
        return from_sleap_file(file, fps)
    elif source_software == "LightningPose":
        return from_lp_file(file, fps)
    elif source_software == "Anipose":
        return from_anipose_file(file, fps, **kwargs)
    elif source_software == "NWB":
        if fps is not None:
            logger.warning(
                "The fps argument is ignored when loading from an NWB file. "
                "The frame rate will be directly read or estimated from "
                "metadata in the file."
            )
        return from_nwb_file(file, **kwargs)
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
    df
        DataFrame containing the pose tracks and confidence scores. Must
        be formatted as in DeepLabCut output files (see Notes).
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.
    source_software
        Name of the pose estimation software from which the data originate.
        Defaults to "DeepLabCut", but it can also be "LightningPose"
        (because they use the same DataFrame format).

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    Notes
    -----
    The DataFrame must have a multi-index column with the following levels:
    "scorer", ("individuals"), "bodyparts", "coords".
    The "individuals" level may be omitted if there is only one individual
    in the video.
    The "coords" level contains either:

    - the spatial coordinates "x", "y", and "likelihood"
      (point-wise confidence scores), or
    - the spatial coordinates "x", "y", and "z" (3D poses estimated by
      `triangulating 2D poses from multiple DeepLabCut output files\
      <https://deeplabcut.github.io/DeepLabCut/docs/Overviewof3D.html>`__).

    The row index corresponds to the frame number.

    See Also
    --------
    movement.io.load_poses.from_dlc_file

    """
    # Read names of individuals and keypoints from the DataFrame
    if "individuals" in df.columns.names:
        individual_names = (
            df.columns.get_level_values("individuals").unique().to_list()
        )
    else:
        individual_names = ["individual_0"]
    keypoint_names = (
        df.columns.get_level_values("bodyparts").unique().to_list()
    )
    # Extract position (and confidence if present)
    coord_names = df.columns.get_level_values("coords").unique().to_list()
    n_coords = len(coord_names)
    tracks = (
        df.to_numpy()
        .reshape((-1, len(individual_names), len(keypoint_names), n_coords))
        .transpose(0, 3, 2, 1)
    )
    if "likelihood" in coord_names:  # Coords: ['x', 'y', 'likelihood']
        likelihood_index = coord_names.index("likelihood")
        confidence_array = tracks[:, likelihood_index, :, :]
        pos_idx = [j for j in range(n_coords) if j != likelihood_index]
        position_array = tracks[:, pos_idx, :, :]
    else:  # Coords: ['x', 'y', 'z']
        position_array = tracks
        confidence_array = None
    return from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software=source_software,
    )


@register_loader(
    "SLEAP", file_validators=[ValidSleapLabels, ValidSleapAnalysis]
)
def from_sleap_file(file: str | Path, fps: float | None = None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a SLEAP file.

    Parameters
    ----------
    file
        Path to the file containing the SLEAP predictions in .h5
        (analysis) format. Alternatively, a .slp (labels) file can
        also be supplied (but this feature is experimental, see Notes).
    fps
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
    .. [1] https://docs.sleap.ai/latest/tutorial/exporting-the-results/#analysis-hdf5
    .. [2] https://github.com/talmolab/sleap/blob/v1.3.3/sleap/info/write_tracking_h5.py#L59
    .. [3] https://docs.sleap.ai/latest/guides/tracking-and-proofreading/

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_sleap_file("path/to/file.analysis.h5", fps=30)

    """
    valid_file = cast("ValidFile", file)
    file_path = valid_file.file
    if file_path.suffix == ".h5":
        ds = _ds_from_sleap_analysis_file(file_path, fps=fps)
    else:  # file.suffix == ".slp"
        ds = _ds_from_sleap_labels_file(file_path, fps=fps)
    # Add metadata as attrs
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded pose tracks from {file_path}:\n{ds}")
    return ds


@register_loader("LightningPose", file_validators=[ValidDeepLabCutCSV])
def from_lp_file(file: str | Path, fps: float | None = None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a LightningPose file.

    Parameters
    ----------
    file
        Path to the file containing the predicted poses, in .csv format.
    fps
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
    valid_file = cast("ValidFile", file)
    ds = _ds_from_lp_or_dlc_file(
        valid_file=valid_file, source_software="LightningPose", fps=fps
    )
    n_individuals = ds.sizes.get("individuals", 1)
    if n_individuals > 1:
        raise logger.error(
            ValueError(
                "LightningPose only supports single-individual datasets, "
                f"but the loaded dataset has {n_individuals} individuals. "
                "Did you mean to load from a DeepLabCut file instead?"
            )
        )
    return ds


@register_loader(
    "DeepLabCut", file_validators=[ValidDeepLabCutH5, ValidDeepLabCutCSV]
)
def from_dlc_file(file: str | Path, fps: float | None = None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a DeepLabCut file.

    Parameters
    ----------
    file
        Path to the file containing the predicted poses, either in .h5
        or .csv format.
    fps
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

    Notes
    -----
    In ``movement``, pose data can only be loaded if all individuals have
    the same set of keypoints (i.e., the same labeled body parts).
    While DeepLabCut supports assigning keypoints that are not shared across
    individuals (see the `DeepLabCut documentation for multi-animal projects
    <https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#b-configure-the-project>`_),
    this feature is not currently supported in ``movement``.

    """
    return _ds_from_lp_or_dlc_file(
        valid_file=cast("ValidFile", file),
        source_software="DeepLabCut",
        fps=fps,
    )


def from_multiview_files(
    file_dict: dict[str, Path | str],
    source_software: Literal["DeepLabCut", "SLEAP", "LightningPose"],
    fps: float | None = None,
) -> xr.Dataset:
    """Load and merge pose tracking data from multiple views (cameras).

    .. deprecated:: 0.14.0
        This function is deprecated and will be removed in a future release.
        Use :func:`movement.io.load_multiview_dataset<movement.io.\
        load.load_multiview_dataset>` instead.

    Parameters
    ----------
    file_dict
        A dict whose keys are the view names and values are the paths to load.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata, with an additional ``views`` dimension.

    """
    warnings.warn(
        "The function `movement.io.load_poses.from_multiview_files` is "
        "deprecated and will be removed in a future release. "
        "Please use `movement.io.load_multiview_dataset` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    views_list = list(file_dict.keys())
    new_coord_views = xr.DataArray(views_list, dims="view")
    dataset_list = [
        from_file(f, source_software=source_software, fps=fps)
        for f in file_dict.values()
    ]
    return xr.concat(dataset_list, dim=new_coord_views)


def _ds_from_lp_or_dlc_file(
    valid_file: ValidFile,
    source_software: Literal["LightningPose", "DeepLabCut"],
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a LightningPose or DLC file.

    Parameters
    ----------
    valid_file
        The validated LightningPose or DeepLabCut file object.
    source_software
        The source software of the file.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    # Load the DeepLabCut poses into a DataFrame
    file_path = valid_file.file
    df = (
        _df_from_dlc_csv(valid_file)
        if isinstance(valid_file, ValidDeepLabCutCSV)
        else pd.DataFrame(pd.read_hdf(file_path, key="df_with_missing"))
        # pd.read_hdf does not always return a DataFrame but we assume it does
        # in this case (since we know what's in the "df_with_missing" dataset)
    )
    logger.debug(f"Loaded poses from {file_path} into a DataFrame.")
    # Convert the DataFrame to an xarray dataset
    ds = from_dlc_style_df(df=df, fps=fps, source_software=source_software)
    # Add metadata as attrs
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded pose tracks from {file_path}:\n{ds}")
    return ds


def _ds_from_sleap_analysis_file(file: Path, fps: float | None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a SLEAP analysis (.h5) file.

    Parameters
    ----------
    file
        Path to the SLEAP analysis file containing predicted pose tracks.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    with h5py.File(file, "r") as f:
        # Transpose to shape: (n_frames, n_space, n_keypoints, n_tracks)
        tracks = f["tracks"][:].transpose(3, 1, 2, 0)
        # Create an array of NaNs for the confidence scores
        scores = np.full(tracks.shape[:1] + tracks.shape[2:], np.nan)
        individual_names = [n.decode() for n in f["track_names"][:]] or None
        if individual_names is None:
            logger.warning(
                f"Could not find SLEAP Track in {file}. "
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


def _ds_from_sleap_labels_file(file: Path, fps: float | None) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a SLEAP labels (.slp) file.

    Parameters
    ----------
    file
        Path to the SLEAP labels file containing predicted pose tracks.
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks, confidence scores,
        and associated metadata.

    """
    labels = read_labels(file.as_posix())
    tracks_with_scores = _sleap_labels_to_numpy(labels)
    individual_names = [track.name for track in labels.tracks] or None
    if individual_names is None:
        logger.warning(
            f"Could not find SLEAP Track in {file}. "
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
    labels
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


def _df_from_dlc_csv(valid_file: ValidDeepLabCutCSV) -> pd.DataFrame:
    """Create a DeepLabCut-style DataFrame from a .csv file.

    If poses are loaded from a DeepLabCut-style .csv file, the DataFrame
    lacks the multi-index columns that are present in the .h5 file. This
    function parses the .csv file to DataFrame with multi-index columns,
    i.e. the same format as in the .h5 file.

    Parameters
    ----------
    valid_file
        The validated DeepLabCut-style CSV file object.

    Returns
    -------
    pandas.DataFrame
        DeepLabCut-style DataFrame with multi-index columns.

    """
    # Deliberately avoid using pd.read_csv with index_col=0 here
    # and instead set the index after reading the CSV,
    # as in cases where the first data row is empty (e.g. "0,,,,,"),
    # pandas will misinterpret that value as the index name instead of a row.
    level_names = valid_file.level_names
    df = pd.read_csv(
        valid_file.file,
        header=list(range(len(level_names))),
    )
    df = df.set_index(df.columns[0])
    df.index.name = None
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=level_names)  # type: ignore[arg-type]
    return df


def from_anipose_style_df(
    df: pd.DataFrame,
    fps: float | None = None,
    individual_name: str = "id_0",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an Anipose 3D dataframe.

    Parameters
    ----------
    df
        Anipose triangulation dataframe
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.
    individual_name
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


@register_loader("Anipose", file_validators=[ValidAniposeCSV])
def from_anipose_file(
    file: str | Path,
    fps: float | None = None,
    individual_name: str = "id_0",
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from an Anipose 3D .csv file.

    Parameters
    ----------
    file
        Path to the Anipose triangulation .csv file
    fps
        The number of frames per second in the video. If None (default),
        the ``time`` coordinates will be in frame units.
    individual_name
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
    valid_file = cast("ValidFile", file)
    anipose_df = pd.read_csv(valid_file.file)
    return from_anipose_style_df(
        anipose_df, fps=fps, individual_name=individual_name
    )


@register_loader("NWB", file_validators=[ValidNWBFile])
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
    file
        Path to the NWB file on disk (ending in ".nwb"),
        or an NWBFile object.
    processing_module_key
        Name of the :class:`pynwb.base.ProcessingModule` in the NWB file that
        contains the pose estimation data. Default is "behavior".
    pose_estimation_key
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
    valid_file = cast("ValidFile", file)
    file_path_or_nwbfile_obj = valid_file.file
    if isinstance(file_path_or_nwbfile_obj, Path):
        with pynwb.NWBHDF5IO(file_path_or_nwbfile_obj, mode="r") as io:
            nwbfile_object = io.read()
            ds = _ds_from_nwb_object(
                nwbfile_object, processing_module_key, pose_estimation_key
            )
            ds.attrs["source_file"] = file_path_or_nwbfile_obj
    else:  # file is an NWBFile object
        ds = _ds_from_nwb_object(
            file_path_or_nwbfile_obj,
            processing_module_key,
            pose_estimation_key,
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
    nwb_file
        An NWBFile object.
    processing_module_key
        Name of the :class:`pynwb.base.ProcessingModule` in the NWB file that
        contains the pose estimation data. Default is "behavior".
    pose_estimation_key
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
    return xr.merge(
        single_keypoint_datasets, join="outer", compat="no_conflicts"
    )


@register_loader("COCO", file_validators=[ValidCOCOJSON])
def from_coco_file(
    file: str | Path,
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a COCO JSON file.

    The input file must follow the `COCO keypoint detection format
    <https://cocodataset.org/#format-data>`_, containing
    ``images``, ``annotations``, and ``categories`` sections.

    Parameters
    ----------
    file
        Path to the COCO keypoint annotation JSON file.
    fps
        The number of frames per second. If None (default),
        the ``time`` coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the pose tracks,
        confidence scores, and associated metadata.

    Notes
    -----
    Each image in the COCO file is treated as one time frame.
    Images are sorted by their ``id`` to establish a temporal
    order. Multiple annotations per image are treated as
    separate individuals. If annotations include a
    ``track_id`` field, it is used for consistent individual
    identity across frames; otherwise individuals are numbered
    per frame (``id_0``, ``id_1``, ...).

    The COCO visibility flag is mapped to confidence:
    ``0`` (not labelled) → ``NaN`` position and ``0``
    confidence, ``1`` (labelled but not visible) → actual
    position with confidence ``0.5``, ``2`` (labelled and
    visible) → actual position with confidence ``1.0``.

    If the annotation includes a ``score`` field, the
    per-keypoint confidence is multiplied by it.

    See Also
    --------
    movement.io.load_poses.from_numpy

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_coco_file("path/to/coco_keypoints.json", fps=30)

    """
    valid_file = cast("ValidCOCOJSON", file)
    file_path = valid_file.file
    data = valid_file.data
    ds = _ds_from_coco_data(data, fps=fps)
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded pose tracks from {file_path}:\n{ds}")
    return ds


def _coco_individual_mapping(
    annotations: list[dict],
    cat_id: int,
    anns_by_image: dict[int, list[dict]],
) -> tuple[list[str], dict[int, int] | None]:
    """Determine individual names and track_id mapping.

    Parameters
    ----------
    annotations
        List of COCO annotation dicts.
    cat_id
        Category ID to filter annotations.
    anns_by_image
        Annotations grouped by image_id.

    Returns
    -------
    tuple
        A tuple of (individual_names, track_id_to_idx).
        track_id_to_idx is None if track_id is not present.

    """
    has_track_id = any("track_id" in ann for ann in annotations)
    if has_track_id:
        track_ids = sorted(
            {
                ann["track_id"]
                for ann in annotations
                if ann.get("category_id") == cat_id
            }
        )
        names = [f"id_{tid}" for tid in track_ids]
        mapping = {tid: i for i, tid in enumerate(track_ids)}
        return names, mapping

    max_individuals = max(
        (len(v) for v in anns_by_image.values()),
        default=1,
    )
    return [f"id_{i}" for i in range(max_individuals)], None


def _coco_fill_arrays(
    anns_by_image: dict[int, list[dict]],
    image_id_to_frame: dict[int, int],
    track_id_to_idx: dict[int, int] | None,
    n_frames: int,
    n_keypoints: int,
    n_individuals: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Populate position and confidence arrays from COCO annotations.

    Parameters
    ----------
    anns_by_image
        Annotations grouped by image_id.
    image_id_to_frame
        Mapping from image_id to frame index.
    track_id_to_idx
        Mapping from track_id to individual index.
        None if track_id is not used.
    n_frames
        Number of frames.
    n_keypoints
        Number of keypoints per individual.
    n_individuals
        Number of individuals.

    Returns
    -------
    tuple
        A tuple of (position_array, confidence_array).

    """
    position = np.full((n_frames, 2, n_keypoints, n_individuals), np.nan)
    confidence = np.full((n_frames, n_keypoints, n_individuals), np.nan)

    for img_id, anns in anns_by_image.items():
        frame_idx = image_id_to_frame.get(img_id)
        if frame_idx is None:
            continue
        for j, ann in enumerate(anns):
            ind_idx = (
                track_id_to_idx[ann["track_id"]]
                if track_id_to_idx is not None
                else j
            )
            _coco_fill_keypoints(
                ann,
                ind_idx,
                frame_idx,
                n_keypoints,
                position,
                confidence,
            )
    return position, confidence


def _coco_fill_keypoints(
    ann: dict,
    ind_idx: int,
    frame_idx: int,
    n_keypoints: int,
    position: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Fill position/confidence for a single annotation.

    Parameters
    ----------
    ann
        A single COCO annotation dict.
    ind_idx
        Individual index.
    frame_idx
        Frame index.
    n_keypoints
        Number of keypoints.
    position
        Position array to fill in-place.
    confidence
        Confidence array to fill in-place.

    """
    kps = ann["keypoints"]
    score = ann.get("score", 1.0)
    for k in range(n_keypoints):
        x = kps[k * 3]
        y = kps[k * 3 + 1]
        v = kps[k * 3 + 2]
        if v == 0:
            position[frame_idx, :, k, ind_idx] = np.nan
            confidence[frame_idx, k, ind_idx] = 0.0
        else:
            position[frame_idx, 0, k, ind_idx] = x
            position[frame_idx, 1, k, ind_idx] = y
            vis_conf = v / 2.0
            confidence[frame_idx, k, ind_idx] = vis_conf * score


def _ds_from_coco_data(
    data: dict,
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from parsed COCO data.

    Parameters
    ----------
    data
        Parsed COCO JSON data as a dictionary.
    fps
        Frames per second. If None, time coordinates will
        be frame numbers.

    Returns
    -------
    xarray.Dataset
        A ``movement`` poses dataset.

    """
    first_cat = data["categories"][0]
    cat_id = first_cat["id"]
    keypoint_names = first_cat["keypoints"]
    n_keypoints = len(keypoint_names)

    images = sorted(data["images"], key=lambda x: x["id"])
    image_id_to_frame = {img["id"]: i for i, img in enumerate(images)}
    n_frames = len(images)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        if ann.get("category_id") != cat_id:
            continue
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    individual_names, track_id_to_idx = _coco_individual_mapping(
        data["annotations"], cat_id, anns_by_image
    )

    position_array, confidence_array = _coco_fill_arrays(
        anns_by_image,
        image_id_to_frame,
        track_id_to_idx,
        n_frames,
        n_keypoints,
        len(individual_names),
    )

    return from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=individual_names,
        keypoint_names=keypoint_names,
        fps=fps,
        source_software="COCO",
    )


@register_loader("BVH", file_validators=[ValidBVHFile])
def from_bvh_file(
    file: str | Path,
    fps: float | None = None,
) -> xr.Dataset:
    """Create a ``movement`` poses dataset from a BVH file.

    `BVH (Biovision Hierarchy)
    <https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html>`_
    is a text-based motion capture format that stores a
    skeleton hierarchy and per-frame joint rotations.

    This function parses the skeleton hierarchy and motion
    data, then computes 3D joint positions via forward
    kinematics.

    Parameters
    ----------
    file
        Path to the BVH file.
    fps
        The number of frames per second. If None (default)
        and the BVH file contains a ``Frame Time`` field,
        fps will be computed from it. Otherwise, the ``time``
        coordinates will be in frame numbers.

    Returns
    -------
    xarray.Dataset
        ``movement`` dataset containing the 3D pose tracks,
        confidence scores (set to ``NaN``), and associated
        metadata.

    Notes
    -----
    The BVH format stores joint rotations (Euler angles)
    rather than positions. This function computes absolute
    3D positions via forward kinematics by traversing the
    skeleton hierarchy, applying rotations and offsets at
    each joint.

    Only joint nodes (``ROOT`` and ``JOINT``) are included
    as keypoints. ``End Site`` nodes are excluded.

    See Also
    --------
    movement.io.load_poses.from_numpy

    Examples
    --------
    >>> from movement.io import load_poses
    >>> ds = load_poses.from_bvh_file("path/to/motion.bvh")

    """
    valid_file = cast("ValidBVHFile", file)
    file_path = valid_file.file

    hierarchy, motion_data, frame_time = _parse_bvh(file_path)
    # Compute fps from frame_time if not provided
    if fps is None and frame_time is not None and frame_time > 0:
        fps = round(1.0 / frame_time, 6)

    # Compute 3D positions via forward kinematics
    joint_names, position_array = _bvh_forward_kinematics(
        hierarchy, motion_data
    )

    n_frames = position_array.shape[0]
    n_keypoints = len(joint_names)
    # BVH has no confidence info; set to NaN
    confidence_array = np.full((n_frames, n_keypoints, 1), np.nan)
    # Add individuals dimension (single individual)
    position_array = position_array[..., np.newaxis]

    ds = from_numpy(
        position_array=position_array,
        confidence_array=confidence_array,
        individual_names=None,
        keypoint_names=joint_names,
        fps=fps,
        source_software="BVH",
    )
    ds.attrs["source_file"] = file_path.as_posix()
    logger.info(f"Loaded pose tracks from {file_path}:\n{ds}")
    return ds


def _parse_bvh(
    file_path: Path,
) -> tuple[list[dict], np.ndarray, float | None]:
    """Parse a BVH file into hierarchy and motion data.

    Parameters
    ----------
    file_path
        Path to the BVH file.

    Returns
    -------
    hierarchy
        List of joint dictionaries with keys: ``name``,
        ``offset``, ``channels``, ``children``,
        ``parent_index``, ``channel_offset``.
    motion_data
        2D array of shape ``(n_frames, n_channels)``
        containing the per-frame motion data values.
    frame_time
        The time between frames in seconds, or None.

    """
    with open(file_path) as f:
        lines = f.readlines()

    joints, motion_start = _parse_bvh_hierarchy(lines)
    motion_data, frame_time = _parse_bvh_motion(lines, motion_start)
    return joints, motion_data, frame_time


def _parse_bvh_hierarchy(
    lines: list[str],
) -> tuple[list[dict], int]:
    """Parse the HIERARCHY section of a BVH file.

    Parameters
    ----------
    lines
        All lines from the BVH file.

    Returns
    -------
    joints
        List of joint dictionaries.
    motion_line
        Line index where MOTION section starts.

    """
    joints: list[dict] = []
    stack: list[int] = []
    channel_offset = 0
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(("ROOT", "JOINT")):
            _add_bvh_joint(line, joints, stack)
        elif line.startswith("End Site"):
            i = _skip_end_site(lines, i)
        elif line.startswith("OFFSET") and stack:
            parts = line.split()
            joints[stack[-1]]["offset"] = np.array(
                [float(p) for p in parts[1:4]]
            )
        elif line.startswith("CHANNELS") and stack:
            parts = line.split()
            n_ch = int(parts[1])
            joints[stack[-1]]["channels"] = parts[2 : 2 + n_ch]
            joints[stack[-1]]["channel_offset"] = channel_offset
            channel_offset += n_ch
        elif line == "}":
            if stack:
                stack.pop()
        elif line.startswith("MOTION"):
            return joints, i
        i += 1

    return joints, i


def _add_bvh_joint(
    line: str,
    joints: list[dict],
    stack: list[int],
) -> None:
    """Add a ROOT or JOINT node to the joints list."""
    parts = line.split()
    name = parts[1]
    joint: dict = {
        "name": name,
        "offset": np.zeros(3),
        "channels": [],
        "children": [],
        "parent_index": (stack[-1] if stack else -1),
        "channel_offset": 0,
    }
    if stack:
        joints[stack[-1]]["children"].append(len(joints))
    joints.append(joint)
    stack.append(len(joints) - 1)


def _skip_end_site(lines: list[str], i: int) -> int:
    """Skip past an End Site block in BVH.

    Returns the line index of the closing brace.
    """
    i += 1
    brace_count = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if "{" in stripped:
            brace_count += 1
        if "}" in stripped:
            if brace_count <= 1:
                return i
            brace_count -= 1
        i += 1
    return i


def _parse_bvh_motion(
    lines: list[str],
    motion_start: int,
) -> tuple[np.ndarray, float | None]:
    """Parse the MOTION section of a BVH file.

    Parameters
    ----------
    lines
        All lines from the BVH file.
    motion_start
        Line index of the MOTION keyword.

    Returns
    -------
    motion_data
        Array of shape ``(n_frames, n_channels)``.
    frame_time
        Seconds per frame, or None.

    """
    i = motion_start + 1
    n_frames = 0
    frame_time: float | None = None

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Frames:"):
            n_frames = int(line.split(":")[1].strip())
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1].strip())
            i += 1
            break
        i += 1

    motion_rows = []
    for j in range(n_frames):
        if i + j < len(lines):
            row = lines[i + j].strip().split()
            motion_rows.append([float(v) for v in row])
    motion_data = np.array(motion_rows)
    return motion_data, frame_time


def _axis_rotation_matrix(axis: str, angle_rad: float) -> np.ndarray:
    """Return a 3×3 rotation matrix for a single axis.

    Parameters
    ----------
    axis
        One of ``"X"``, ``"Y"``, or ``"Z"``.
    angle_rad
        Rotation angle in radians.

    Returns
    -------
    numpy.ndarray
        3×3 rotation matrix.

    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    # Z
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler_to_rotation_matrix(
    angles: np.ndarray,
    order: str,
) -> np.ndarray:
    """Convert Euler angles (degrees) to a rotation matrix.

    Parameters
    ----------
    angles
        Array of 3 Euler angles in degrees.
    order
        Rotation order string, e.g. ``"ZXY"``.

    Returns
    -------
    numpy.ndarray
        3×3 rotation matrix.

    """
    rad = np.deg2rad(angles)
    axis_to_idx = {"X": 0, "Y": 1, "Z": 2}
    rot = np.eye(3)
    for axis_char in order:
        idx = axis_to_idx[axis_char]
        rot = rot @ _axis_rotation_matrix(axis_char, rad[idx])
    return rot


def _extract_bvh_channels(
    channels: list[str],
    frame_data: np.ndarray,
    ch_offset: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Extract translation, rotation, and order from channels.

    Parameters
    ----------
    channels
        List of channel names for a joint.
    frame_data
        1D array of all channel values for one frame.
    ch_offset
        Starting index into ``frame_data``.

    Returns
    -------
    translation
        3D translation vector.
    rotation_angles
        3D rotation angles in degrees.
    rotation_order
        String indicating rotation axis order.

    """
    channel_map = {
        "Xposition": ("t", 0),
        "Yposition": ("t", 1),
        "Zposition": ("t", 2),
        "Xrotation": ("r", 0),
        "Yrotation": ("r", 1),
        "Zrotation": ("r", 2),
    }
    translation = np.zeros(3)
    rotation_angles = np.zeros(3)
    rotation_order = ""
    for c_idx, ch_name in enumerate(channels):
        val = frame_data[ch_offset + c_idx]
        kind, axis_idx = channel_map[ch_name]
        if kind == "t":
            translation[axis_idx] = val
        else:
            rotation_angles[axis_idx] = val
            rotation_order += ch_name[0]
    return translation, rotation_angles, rotation_order


def _bvh_forward_kinematics(
    joints: list[dict],
    motion_data: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Compute 3D joint positions from BVH data.

    Parameters
    ----------
    joints
        List of joint dictionaries from ``_parse_bvh``.
    motion_data
        2D array of shape ``(n_frames, n_channels)``.

    Returns
    -------
    joint_names
        List of joint names.
    positions
        Array of shape ``(n_frames, 3, n_joints)``
        containing the 3D positions.

    """
    n_frames = motion_data.shape[0]
    n_joints = len(joints)
    joint_names = [j["name"] for j in joints]
    positions = np.zeros((n_frames, 3, n_joints))

    for frame in range(n_frames):
        transforms: list[tuple[np.ndarray, np.ndarray] | None] = [
            None
        ] * n_joints
        for j_idx, joint in enumerate(joints):
            trans, rot_ang, rot_ord = _extract_bvh_channels(
                joint["channels"],
                motion_data[frame],
                joint["channel_offset"],
            )
            local_rot = (
                _euler_to_rotation_matrix(rot_ang, rot_ord)
                if rot_ord
                else np.eye(3)
            )
            parent = joint["parent_index"]
            if parent == -1:
                g_pos = trans + joint["offset"]
                g_rot = local_rot
            else:
                p_rot, p_pos = transforms[parent]
                g_pos = p_pos + p_rot @ joint["offset"]
                g_rot = p_rot @ local_rot
            transforms[j_idx] = (g_rot, g_pos)
            positions[frame, :, j_idx] = g_pos

    return joint_names, positions
