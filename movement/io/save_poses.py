"""Save pose tracking data from ``movement`` to various file formats."""

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import pynwb
import xarray as xr

from movement.io.nwb import (
    NWBFileSaveConfig,
    _ds_to_pose_and_skeletons,
    _write_processing_module,
)
from movement.utils.logging import logger
from movement.validators.datasets import ValidPosesDataset, _validate_dataset
from movement.validators.files import _validate_file_path


def _ds_to_dlc_style_df(
    ds: xr.Dataset, columns: pd.MultiIndex
) -> pd.DataFrame:
    """Convert a ``movement`` dataset to a DLC-style DataFrame."""
    # Check shapes of position and confidence data
    position_shape = ds.position.data.shape
    confidence_shape = ds.confidence.data.shape
    print("Position shape:", position_shape)
    print("Confidence shape:", confidence_shape)

    # Concatenate the pose tracks and confi scores into one array
    tracks_with_scores = np.concatenate(
        (
            ds.position.data,
            ds.confidence.data[:, np.newaxis, ...],
        ),
        axis=1,
    )

    # Check the shape after concatenation
    print("Tracks with scores shape:", tracks_with_scores.shape)

    # Reverse the order of the dimensions except for the time dimension
    transpose_order = [0] + list(range(tracks_with_scores.ndim - 1, 0, -1))
    tracks_with_scores = tracks_with_scores.transpose(transpose_order)

    # Check the shape of the data
    expected_columns = columns.shape[0]
    actual_shape = tracks_with_scores.reshape(ds.sizes["time"], -1).shape[1]

    if actual_shape != expected_columns:
        raise ValueError(f"""Shape of passed values is {actual_shape},
                        but indices imply {expected_columns}.""")

    # Create DataFrame with multi-index columns
    df = pd.DataFrame(
        data=tracks_with_scores.reshape(ds.sizes["time"], -1),
        index=np.arange(ds.sizes["time"], dtype=int),
        columns=columns,
        dtype=float,
    )

    return df


def _auto_split_individuals(ds: xr.Dataset) -> bool:
    """Return True if there is only one individual in the dataset."""
    n_individuals = ds.sizes["individuals"]
    return n_individuals == 1


def _save_dlc_df(filepath: Path, df: pd.DataFrame) -> None:
    """Save the dataframe as either a .h5 or .csv depending on the file path.

    Parameters
    ----------
    filepath : pathlib.Path
        Path of the file to save the dataframe to. The file extension
        must be either .h5 (recommended) or .csv.
    df : pandas.DataFrame
        Pandas Dataframe to save

    """
    if filepath.suffix == ".csv":
        df.to_csv(filepath, sep=",")
    else:  # at this point it can only be .h5 (because of validation)
        df.to_hdf(filepath, key="df_with_missing")


def to_dlc_style_df(
    ds: xr.Dataset,
    split_individuals: bool = False,
    dlc_df_format: Literal["single-animal", "multi-animal"] = "multi-animal",
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Convert a ``movement`` dataset to DeepLabCut-style DataFrame(s).

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    split_individuals : bool, optional
        If True, return a dictionary of DataFrames per individual, with
        individual names as keys. If False (default), return a single
        DataFrame for all individuals.
    dlc_df_format : {"single-animal", "multi-animal"}, optional
        Specifies the DLC dataframe format. "single-animal" produces the
        older format (<DLC 2.0) without the "individuals" column level,
        while "multi-animal" includes it (DLC >=2.0).
        Defaults to "multi-animal".

    Returns
    -------
    pandas.DataFrame or dict
        DeepLabCut-style pandas DataFrame or dictionary of DataFrames.

    """
    _validate_dataset(ds, ValidPosesDataset)
    scorer = ["movement"]
    bodyparts = ds.coords["keypoints"].data.tolist()
    coords = ds.coords["space"].data.tolist() + ["likelihood"]
    individuals = ds.coords["individuals"].data.tolist()

    if split_individuals:
        df_dict = {}

        for individual in individuals:
            individual_data = ds.sel(individuals=individual)

            index_levels = ["scorer", "bodyparts", "coords"]
            if dlc_df_format == "multi-animal":
                index_levels.insert(1, "individuals")

            columns = pd.MultiIndex.from_product(
                [scorer]
                + ([individuals] if dlc_df_format == "multi-animal" else [])
                + [bodyparts, coords],
                names=index_levels,
            )

            df = _ds_to_dlc_style_df(individual_data, columns)
            df_dict[individual] = df

        logger.info(
            f"""Converted poses dataset to DeepLabCut-style DataFrames
            per individual using '{dlc_df_format}' format."""
        )
        return df_dict
    else:
        index_levels = (
            ["scorer", "individuals", "bodyparts", "coords"]
            if dlc_df_format == "multi-animal"
            else ["scorer", "bodyparts", "coords"]
        )

        columns = pd.MultiIndex.from_product(
            [scorer]
            + ([individuals] if dlc_df_format == "multi-animal" else [])
            + [bodyparts, coords],
            names=index_levels,
        )

        df_all = _ds_to_dlc_style_df(ds, columns)

        logger.info(
            f"""Converted poses dataset to DeepLabCut-style
            DataFrame using '{dlc_df_format}' format."""
        )
        return df_all


def to_dlc_file(
    ds: xr.Dataset,
    file_path: str | Path,
    split_individuals: bool | Literal["auto"] = "auto",
    dlc_df_format: Literal["single-animal", "multi-animal"] = "multi-animal",
) -> None:
    """Save a ``movement`` dataset to DeepLabCut file(s).

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    file_path : pathlib.Path or str
        Path to the file to save the poses to. The file extension
        must be either .h5 (recommended) or .csv.
    split_individuals : bool or "auto", optional
        Whether to save individuals to separate files or to the same file.
        Defaults to "auto" (determined based on dataset individuals).
    dlc_df_format : {"single-animal", "multi-animal"}, optional
        Specifies the DLC dataframe format. "single-animal" produces the
        older format (<DLC 2.0) without the "individuals" column level,
        while "multi-animal" includes it (DLC >=2.0).
        Defaults to "multi-animal".

    Notes
    -----
    If ``split_individuals`` is True, each individual will be saved to a
    separate file, but the DLC format is determined by ``dlc_df_format``.
    If False, all individuals will be saved to the same file, also formatted
    according to ``dlc_df_format``.

    """  # noqa: D301
    file = _validate_file_path(file_path, expected_suffix=[".csv", ".h5"])

    # Determine splitting behavior
    if split_individuals == "auto":
        split_individuals = _auto_split_individuals(ds)
    elif not isinstance(split_individuals, bool):
        raise logger.error(
            ValueError(
                "Expected 'split_individuals' to be a boolean or 'auto', "
                f"but got {type(split_individuals)}."
            )
        )

    # Validate DLC format
    if dlc_df_format not in ["single-animal", "multi-animal"]:
        raise log_error(
            ValueError,
            f"""Invalid value for 'dlc_df_format': {dlc_df_format}.
            Expected 'single-animal' or 'multi-animal'.""",
        )

    if split_individuals:
        # Split dataset into multiple files while maintaining DLC format
        df_dict = to_dlc_style_df(
            ds, split_individuals=True, dlc_df_format=dlc_df_format
        )

        for key, df in df_dict.items():
            filepath = f"{file.path.with_suffix('')}_{key}{file.path.suffix}"
            if isinstance(df, pd.DataFrame):
                _save_dlc_df(Path(filepath), df)
            logger.info(f"Saved poses for individual {key} to {filepath}.")
    else:
        # Convert dataset to a single dataframe using the chosen DLC format
        df_all = to_dlc_style_df(
            ds, split_individuals=False, dlc_df_format=dlc_df_format
        )
        if isinstance(df_all, pd.DataFrame):
            _save_dlc_df(file.path, df_all)
        logger.info(f"Saved poses dataset to {file.path}.")


def to_lp_file(
    ds: xr.Dataset,
    file_path: str | Path,
) -> None:
    """Save a ``movement`` dataset to a LightningPose file.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    file_path : pathlib.Path or str
        Path to the file to save the poses to. File extension must be .csv.

    Notes
    -----
    LightningPose saves pose estimation outputs as .csv files, using the same
    format as single-animal DeepLabCut projects. Therefore, under the hood,
    this function calls :func:`movement.io.save_poses.to_dlc_file`
    with ``split_individuals=True``. This setting means that each individual
    is saved to a separate file, with the individual's name appended to the
    file path, just before the file extension,
    i.e. "/path/to/filename_individual1.csv".

    See Also
    --------
    to_dlc_file : Save dataset to a DeepLabCut-style .h5 or .csv file.

    """
    file = _validate_file_path(file_path=file_path, expected_suffix=[".csv"])
    _validate_dataset(ds, ValidPosesDataset)
    to_dlc_file(ds, file.path, split_individuals=True)


def to_sleap_analysis_file(ds: xr.Dataset, file_path: str | Path) -> None:
    """Save a ``movement`` dataset to a SLEAP analysis file.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    file_path : pathlib.Path or str
        Path to the file to save the poses to. File extension must be .h5.

    Notes
    -----
    The output file will contain the following keys (as in SLEAP .h5 analysis
    files):
    "track_names", "node_names", "tracks", "track_occupancy", "point_scores",
    "instance_scores", "tracking_scores", "labels_path", "edge_names",
    "edge_inds", "video_path", "video_ind", "provenance" [1]_.
    However, only "track_names", "node_names", "tracks", "track_occupancy"
    and "point_scores" will contain data extracted from the input dataset.
    "labels_path" will contain the path to the input file only if the source
    file of the dataset is a SLEAP .slp file. Otherwise, it will be an empty
    string.
    The other attributes and data variables that are not present in the input
    dataset will contain default (empty) values.

    References
    ----------
    .. [1] https://docs.sleap.ai/latest/api/info/write_tracking_h5/#sleap.info.write_tracking_h5

    Examples
    --------
    >>> from movement.io import save_poses, load_poses
    >>> ds = load_poses.from_dlc_file("path/to/file.h5")
    >>> save_poses.to_sleap_analysis_file(
    ...     ds, "/path/to/file_sleap.analysis.h5"
    ... )

    """
    file = _validate_file_path(file_path=file_path, expected_suffix=[".h5"])
    _validate_dataset(ds, ValidPosesDataset)

    ds = _remove_unoccupied_tracks(ds)

    # Target shapes:
    # "track_occupancy"     n_frames * n_individuals
    # "tracks"              n_individuals * n_space * n_keypoints * n_frames
    # "track_names"         n_individuals
    # "point_scores"        n_individuals * n_keypoints * n_frames
    # "instance_scores"     n_individuals * n_frames
    # "tracking_scores"     n_individuals * n_frames
    individual_names = ds.individuals.values.tolist()
    n_individuals = len(individual_names)
    keypoint_names = ds.keypoints.values.tolist()
    # Compute frame indices from fps, if set
    fps = getattr(ds, "fps", None)
    if fps is not None:
        frame_idxs = np.rint(ds.time.values * fps).astype(int).tolist()
    else:
        frame_idxs = ds.time.values.astype(int).tolist()
    n_frames = frame_idxs[-1] - frame_idxs[0] + 1
    pos_x = ds.position.sel(space="x").values
    # Mask denoting which individuals are present in each frame
    track_occupancy = (~np.all(np.isnan(pos_x), axis=1)).astype(int)
    tracks = ds.position.data.transpose(3, 1, 2, 0)
    point_scores = ds.confidence.data.T
    instance_scores = np.full((n_individuals, n_frames), np.nan, dtype=float)
    tracking_scores = np.full((n_individuals, n_frames), np.nan, dtype=float)

    source_file = getattr(ds, "source_file", None)
    labels_path = (
        source_file
        if source_file is not None and Path(source_file).suffix == ".slp"
        else ""
    )
    data_dict = dict(
        track_names=individual_names,
        node_names=keypoint_names,
        tracks=tracks,
        track_occupancy=track_occupancy,
        point_scores=point_scores,
        instance_scores=instance_scores,
        tracking_scores=tracking_scores,
        labels_path=labels_path,
        edge_names=[],
        edge_inds=[],
        video_path="",
        video_ind=0,
        provenance="{}",
    )
    with h5py.File(file.path, "w") as f:
        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                f.create_dataset(
                    key,
                    data=val,
                    compression="gzip",
                    compression_opts=9,
                )
            else:
                f.create_dataset(key, data=val)
    logger.info(f"Saved poses dataset to {file.path}.")


def to_nwb_file(
    ds: xr.Dataset, config: NWBFileSaveConfig | None = None
) -> pynwb.file.NWBFile | list[pynwb.file.NWBFile]:
    """Save a ``movement`` dataset to one or more NWBFile objects.

    The data will be written to :class:`pynwb.file.NWBFile` object(s)
    in the "behavior" processing module, formatted according to the
    ``ndx-pose`` NWB extension [1]_.
    Each individual in the dataset will be written to a separate NWBFile,
    as required by the NWB format.
    Note that the NWBFile(s) are not automatically saved to disk.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` poses dataset containing the data to be converted to
        NWBFile(s).
    config : NWBFileSaveConfig, optional
        Configuration object containing keyword arguments to customise the
        :class:`pynwb.file.NWBFile` (s) that will be created
        for each individual.
        If None (default), default values will be used.

    Returns
    -------
    pynwb.file.NWBFile or list[pynwb.file.NWBFile]
        If the dataset contains only one individual, a single NWBFile object
        will be returned. If the dataset contains multiple individuals,
        a list of NWBFile objects will be returned, one for each individual.

    References
    ----------
    .. [1] https://github.com/rly/ndx-pose

    Examples
    --------
    Create :class:`pynwb.file.NWBFile` objects for each individual in
    a ``movement`` poses dataset ``ds`` and save them to disk:

    >>> from movement.sample_data import fetch_dataset
    >>> from movement.io import save_poses
    >>> from pynwb import NWBHDF5IO
    >>> ds = fetch_dataset("DLC_two-mice.predictions.csv")
    >>> nwb_files = save_poses.to_nwb_file(ds)
    >>> for file in nwb_files:
    ...     with NWBHDF5IO(f"{file.identifier}.nwb", "w") as io:
    ...         io.write(file)

    Create NWBFiles with custom metadata shared across individuals.
    Specifically, we add metadata for :class:`pynwb.file.NWBFile`,
    :class:`pynwb.base.ProcessingModule`, and :class:`pynwb.file.Subject`
    via the :class:`NWBFileSaveConfig<movement.io.nwb.NWBFileSaveConfig>`
    object.

    >>> from movement.io.nwb import NWBFileSaveConfig
    >>> config = NWBFileSaveConfig(
    ...     nwbfile_kwargs={"session_description": "test session"},
    ...     processing_module_kwargs={"description": "processed behav data"},
    ...     subject_kwargs={"age": "P90D", "species": "Mus musculus"},
    ... )
    >>> nwb_files = save_poses.to_nwb_file(ds, config)

    Create NWBFiles with different :class:`pynwb.file.NWBFile`
    and :class:`pynwb.file.Subject` metadata for each individual
    (e.g. ``individual1``, ``individual2``) in the dataset:

    >>> config = NWBFileSaveConfig(
    ...     nwbfile_kwargs={
    ...         "individual1": {
    ...             "experimenter": "experimenter1",
    ...             "session_description": "subj1 session",
    ...         },
    ...         "individual2": {
    ...             "experimenter": "experimenter2",
    ...             "session_description": "subj2 session",
    ...         },
    ...     },
    ...     subject_kwargs={
    ...         "individual1": {"age": "P90D", "sex": "M"},
    ...         "individual2": {"age": "P91D", "sex": "F"},
    ...     },
    ... )
    >>> nwb_files = save_poses.to_nwb_file(ds, config)

    Create NWBFiles with different ``ndx_pose.PoseEstimationSeries``
    metadata for different keypoints (e.g. ``leftear``, ``rightear``):

    >>> config = NWBFileSaveConfig(
    ...     pose_estimation_series_kwargs={
    ...         "leftear": {
    ...             "description": "left ear",
    ...         },
    ...         "rightear": {
    ...             "description": "right ear",
    ...         },
    ...     },
    ... )
    >>> nwb_files = save_poses.to_nwb_file(ds, config)

    See Also
    --------
    movement.io.nwb.NWBFileSaveConfig :
        For further details on the configuration object and its parameters.

    """
    config = config or NWBFileSaveConfig()
    individuals = ds.individuals.values.tolist()
    if isinstance(individuals, str):
        individuals = [individuals]
    is_multi_ind = len(individuals) > 1
    subjects = {
        id: pynwb.file.Subject(
            **(config._resolve_subject_kwargs(id, is_multi_ind))
        )
        for id in individuals
    }
    nwb_files = [
        pynwb.NWBFile(
            subject=subjects[id],
            **(config._resolve_nwbfile_kwargs(id, is_multi_ind)),
        )
        for id in individuals
    ]
    pose_estimation_skeletons = {
        id: _ds_to_pose_and_skeletons(
            ds
            if ds.sizes.get("individuals") is None
            else ds.sel(individuals=id),
            config,
            subjects[id],
            is_multi_ind,
        )
        for id in individuals
    }
    for nwb_file, id in zip(nwb_files, individuals, strict=True):
        pose_estimation, skeletons = pose_estimation_skeletons[id]
        processing_module_kwargs = config._resolve_processing_module_kwargs(id)
        _write_processing_module(
            nwb_file, processing_module_kwargs, pose_estimation, skeletons
        )
    return nwb_files if is_multi_ind else nwb_files[0]


def _remove_unoccupied_tracks(ds: xr.Dataset):
    """Remove tracks that are completely unoccupied from the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.

    Returns
    -------
    xarray.Dataset
        The input dataset without the unoccupied tracks.

    """
    all_nan = ds.position.isnull().all(dim=["keypoints", "space", "time"])
    return ds.where(~all_nan, drop=True)
