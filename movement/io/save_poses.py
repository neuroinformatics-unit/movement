"""Save pose tracking data from ``movement`` to various file formats."""

import warnings
from pathlib import Path
from typing import Literal, overload

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
from movement.io.save import register_writer
from movement.utils.logging import logger
from movement.validators.datasets import ValidPosesInputs
from movement.validators.files import _validate_file_path


def _ds_to_dlc_style_df(
    ds: xr.Dataset, columns: pd.MultiIndex
) -> pd.DataFrame:
    """Convert a ``movement`` dataset to a DeepLabCut-style DataFrame.

    Parameters
    ----------
    ds
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    columns
        DeepLabCut-style multi-index columns

    Returns
    -------
    pandas.DataFrame

    """
    # DLC stores confidence scores per keypoint. If confidence is provided
    # per individual, expand it across all keypoints before export.
    confidence = ds.confidence.data
    if "keypoint" not in ds.confidence.dims:
        logger.warning(
            "Dataset contains individual-wise confidence scores. "
            "DLC-style formats only supports keypoint-wise confidence scores, "
            "so confidence values will be expanded to all keypoints."
        )
        confidence = np.repeat(
            np.expand_dims(confidence, axis=1), ds.sizes["keypoint"], axis=1
        )
    # Keep position data as is, if data is 3D (i.e. contains 'z' coordinate)
    # Otherwise, concatenate position and confidence scores into one array
    tracks = (
        ds.position.data
        if "z" in columns.get_level_values("coords")
        else np.concatenate(
            (
                ds.position.data,
                confidence[:, np.newaxis, ...],
            ),
            axis=1,
        )
    )
    # Reverse the order of the dimensions except for the time dimension
    transpose_order = [0] + list(range(tracks.ndim - 1, 0, -1))
    tracks = tracks.transpose(transpose_order)
    # Create DataFrame with multi-index columns
    df = pd.DataFrame(
        data=tracks.reshape(ds.sizes["time"], -1),
        index=np.arange(ds.sizes["time"], dtype=int),
        columns=columns,
        dtype=float,
    )
    return df


def _save_dlc_df(
    filepath: Path, df: pd.DataFrame, *, overwrite: bool = True
) -> None:
    """Save the dataframe as either a .h5 or .csv depending on the file path.

    Parameters
    ----------
    filepath
        Path of the file to save the dataframe to. The file extension
        must be either .h5 (recommended) or .csv.
    df
        Pandas DataFrame to save.
    overwrite
        If True (default), overwrite an existing writable file.
        If False, raise ``FileExistsError`` if the file exists.

    """
    _validate_file_path(
        filepath, overwrite=overwrite, suffixes={".csv", ".h5"}
    )
    if filepath.suffix == ".csv":
        df.to_csv(filepath, sep=",")
    else:  # at this point it can only be .h5 (because of validation)
        df.to_hdf(filepath, key="df_with_missing", mode="w")


@overload
def to_dlc_style_df(
    ds: xr.Dataset, split_individuals: Literal[False] = False
) -> pd.DataFrame: ...
@overload
def to_dlc_style_df(
    ds: xr.Dataset, split_individuals: Literal[True]
) -> dict[str, pd.DataFrame]: ...
@overload
def to_dlc_style_df(
    ds: xr.Dataset, split_individuals: bool
) -> pd.DataFrame | dict[str, pd.DataFrame]: ...
def to_dlc_style_df(
    ds: xr.Dataset, split_individuals: bool = False
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Convert a ``movement`` dataset to DeepLabCut-style DataFrame(s).

    Parameters
    ----------
    ds
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    split_individuals
        If True, return a dictionary of DataFrames per individual, with
        individual names as keys. If False (default), return a single
        DataFrame for all individuals (see Notes).

    Returns
    -------
    pandas.DataFrame or dict
        DeepLabCut-style pandas DataFrame or dictionary of DataFrames.

    See Also
    --------
    to_dlc_file : Save dataset directly to a DeepLabCut-style .h5 or .csv file.

    Notes
    -----
    The DataFrame(s) will have a multi-index column with the following levels:
    "scorer", "bodyparts", "coords" (if split_individuals is True),
    or "scorer", "individuals", "bodyparts", "coords"
    (if split_individuals is False).

    For 2D data, regardless of the provenance of the points-wise confidence
    scores, they will be referred to as "likelihood", and stored in
    the "coords" level as DeepLabCut expects.

    For 3D data, the "coords" level will only contain "x", "y", and "z",
    as DeepLabCut does not currently provide 3D likelihoods.

    """
    ValidPosesInputs.validate(ds)
    scorer = ["movement"]
    bodyparts = ds.coords["keypoint"].data.tolist()
    base_coords = ds.coords["space"].data.tolist()
    coords = (
        base_coords
        if "z" in ds.coords["space"]
        else base_coords + ["likelihood"]
    )
    individuals = ds.coords["individual"].data.tolist()
    if split_individuals:
        df_dict = {}
        for individual in individuals:
            individual_data = ds.sel(individual=individual)
            index_levels = ["scorer", "bodyparts", "coords"]
            columns = pd.MultiIndex.from_product(
                [scorer, bodyparts, coords], names=index_levels
            )
            df = _ds_to_dlc_style_df(individual_data, columns)
            df_dict[individual] = df
        logger.info(
            "Converted poses dataset to DeepLabCut-style DataFrames "
            "per individual."
        )
        return df_dict
    else:
        index_levels = ["scorer", "individuals", "bodyparts", "coords"]
        columns = pd.MultiIndex.from_product(
            [scorer, individuals, bodyparts, coords], names=index_levels
        )
        df_all = _ds_to_dlc_style_df(ds, columns)
        logger.info("Converted poses dataset to DeepLabCut-style DataFrame.")
        return df_all


@register_writer("DeepLabCut", ds_type="poses", suffixes={".h5", ".csv"})
def to_dlc_file(
    ds: xr.Dataset,
    file: str | Path,
    split_individuals: bool | Literal["auto"] = "auto",
    *,
    overwrite: bool = True,
) -> None:
    """Save a ``movement`` dataset to DeepLabCut style .h5 or .csv file(s).

    Parameters
    ----------
    ds
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    file
        Path to the output file. The file extension must be either
        .h5 (recommended) or .csv. When ``split_individuals=True``, this
        path is treated as a template: each individual is written to
        ``{stem}_{individual}{suffix}``.
    split_individuals
        Whether to save individuals to separate files or to the same file.
        Defaults to ``"auto"``: split only when the dataset has one
        individual, matching the previous behavior.
    overwrite
        If True (default), an existing target file will be overwritten
        provided it is writable. If False, a ``FileExistsError`` is raised
        if any target file already exists. When ``split_individuals=True``,
        all derived paths are validated *before* any file is written, so
        the operation fails atomically rather than partially overwriting.

    Raises
    ------
    FileExistsError
        If a target file exists and ``overwrite=False``.
    PermissionError
        If a target file exists, ``overwrite=True``, but is not writable.

    """
    file_path = Path(file)
    if split_individuals == "auto":
        split_individuals = ds.individual.size == 1
    elif not isinstance(split_individuals, bool):
        raise logger.error(
            ValueError(
                "Expected 'split_individuals' to be a boolean or 'auto', "
                f"but got {type(split_individuals)}."
            )
        )

    if split_individuals and ds.individual.size > 1:
        suffix = file_path.suffix
        individuals = list(ds.individual.values)

        derived_paths = []
        for indiv in individuals:
            derived = file_path.with_name(f"{file_path.stem}_{indiv}{suffix}")
            _validate_file_path(
                derived, overwrite=overwrite, suffixes={".csv", ".h5"}
            )
            derived_paths.append((indiv, derived))

        # Reuse the public converter instead of a non-existent _build_columns
        df_dict = to_dlc_style_df(ds, split_individuals=True)
        for indiv, derived in derived_paths:
            _save_dlc_df(derived, df_dict[indiv], overwrite=overwrite)
    elif split_individuals:
        _validate_file_path(
            file_path, overwrite=overwrite, suffixes={".csv", ".h5"}
        )
        individual = ds.individual.values[0]
        df_dict = to_dlc_style_df(ds, split_individuals=True)
        _save_dlc_df(file_path, df_dict[individual], overwrite=overwrite)
    else:
        _validate_file_path(
            file_path, overwrite=overwrite, suffixes={".csv", ".h5"}
        )
        df = to_dlc_style_df(ds, split_individuals=False)
        _save_dlc_df(file_path, df, overwrite=overwrite)


@register_writer("SLEAP", ds_type="poses", suffixes={".h5"})
def to_sleap_analysis_file(
    ds: xr.Dataset,
    file: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    """Save a ``movement`` dataset to a SLEAP analysis .h5 file.

    Parameters
    ----------
    ds
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    file
        Path to the output .h5 file.
    overwrite
        If True (default), an existing target file will be overwritten
        provided it is writable. If False, a ``FileExistsError`` is raised
        if the target file already exists.

    Raises
    ------
    FileExistsError
        If the target file exists and ``overwrite=False``.
    PermissionError
        If the target file exists, ``overwrite=True``, but is not writable.

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
    file_path = Path(file)

    # Validate before doing any work
    _validate_file_path(file_path, overwrite=overwrite, suffixes={".h5"})

    # (building the tracks arrays and writing them with h5py/pandas)

    ds = _remove_unoccupied_tracks(ds)
    # Target shapes:
    # "track_occupancy"     n_frames * n_individuals
    # "tracks"              n_individuals * n_space * n_keypoints * n_frames
    # "track_names"         n_individuals
    # "point_scores"        n_individuals * n_keypoints * n_frames
    # "instance_scores"     n_individuals * n_frames
    # "tracking_scores"     n_individuals * n_frames
    individual_names = ds.individual.values.tolist()
    n_individuals = len(individual_names)
    keypoint_names = ds.keypoint.values.tolist()
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
    with h5py.File(file_path, "w") as f:
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
    logger.info(f"Saved poses dataset to {file_path}.")


@register_writer("LightningPose", ds_type="poses", suffixes={".csv"})
def to_lp_file(
    ds: xr.Dataset,
    file: str | Path,
    *,
    overwrite: bool = True,
) -> None:
    """Save a ``movement`` dataset to LightningPose-style files."""
    to_dlc_file(ds, file, split_individuals=True, overwrite=overwrite)


def to_nwb_file_object(
    ds: xr.Dataset, config: NWBFileSaveConfig | None = None
) -> pynwb.file.NWBFile | list[pynwb.file.NWBFile]:
    """Convert a ``movement`` dataset to one or more NWBFile objects.

    The data will be written to :class:`pynwb.file.NWBFile` object(s)
    in the "behavior" processing module, formatted according to the
    ``ndx-pose`` NWB extension [1]_.
    Each individual in the dataset will be written to a separate NWBFile,
    as required by the NWB format.
    Note that the NWBFile(s) are not automatically saved to disk.

    Parameters
    ----------
    ds
        ``movement`` poses dataset containing the data to be converted to
        NWBFile(s).
    config
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

    See Also
    --------
    movement.io.nwb.NWBFileSaveConfig :
        For further details on the configuration object and its parameters.

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
    >>> nwb_files = save_poses.to_nwb_file_object(ds)
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
    >>> nwb_files = save_poses.to_nwb_file_object(ds, config)

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
    >>> nwb_files = save_poses.to_nwb_file_object(ds, config)

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
    >>> nwb_files = save_poses.to_nwb_file_object(ds, config)

    """
    config = config or NWBFileSaveConfig()
    individuals = ds.individual.values.tolist()
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
            if ds.sizes.get("individual") is None
            else ds.sel(individual=id),
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


def to_nwb_file(
    ds: xr.Dataset, config: NWBFileSaveConfig | None = None
) -> pynwb.file.NWBFile | list[pynwb.file.NWBFile]:
    """Convert a ``movement`` dataset to one or more NWBFile objects.

    The data will be written to :class:`pynwb.file.NWBFile` object(s)
    in the "behavior" processing module, formatted according to the
    ``ndx-pose`` NWB extension [1]_.
    Each individual in the dataset will be written to a separate NWBFile,
    as required by the NWB format.
    Note that the NWBFile(s) are not automatically saved to disk.

    .. deprecated:: 0.18.0
        ``to_nwb_file`` has been renamed to :func:`to_nwb_file_object`
        and is now deprecated. A future release will repurpose
        ``to_nwb_file`` as a function that writes NWB file(s) directly
        to disk. Use :func:`to_nwb_file_object` to create NWBFile object(s)
        without writing to disk.

    Parameters
    ----------
    ds
        ``movement`` poses dataset containing the data to be converted to
        NWBFile(s).
    config
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

    See Also
    --------
    movement.io.nwb.NWBFileSaveConfig :
        For further details on the configuration object and its parameters.

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
    >>> nwb_files = save_poses.to_nwb_file_object(ds)
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
    >>> nwb_files = save_poses.to_nwb_file_object(ds, config)

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
    >>> nwb_files = save_poses.to_nwb_file_object(ds, config)

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
    >>> nwb_files = save_poses.to_nwb_file_object(ds, config)

    """
    warnings.warn(
        "`to_nwb_file` has been renamed to `to_nwb_file_object` and is now "
        "deprecated. In a future release, `to_nwb_file` will write NWB files "
        "directly to disk. Use `to_nwb_file_object` to create NWBFile "
        "object(s) without writing to disk.",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_nwb_file_object(ds, config)


@register_writer("NWB", ds_type="poses", suffixes={".nwb"})
def _write_nwb_file(ds: xr.Dataset, file: str | Path, **kwargs) -> None:
    """Save a ``movement`` poses dataset to one or more NWB files.

    :func:`to_nwb_file_object` builds the NWBFile object(s) but does not
    write them to disk; this helper writes them. Multi-individual datasets
    yield one file per individual, with the individual name appended to
    the file path. This function will be promoted to public `to_nwb_file` in a
    future release.
    """
    valid_path = Path(file)
    nwb_files = to_nwb_file_object(ds, **kwargs)
    if isinstance(nwb_files, pynwb.file.NWBFile):
        _write_nwb_to_disk(nwb_files, valid_path)
    else:  # list of NWBFile objects for multi-individual datasets
        for nwb_file in nwb_files:
            individual_path = valid_path.with_name(
                f"{valid_path.stem}_{nwb_file.identifier}{valid_path.suffix}"
            )
            individual_path = _validate_file_path(
                individual_path, overwrite=True, suffixes={".nwb"}
            )
            _write_nwb_to_disk(nwb_file, individual_path)


def _write_nwb_to_disk(nwb_file: pynwb.file.NWBFile, file: Path) -> None:
    """Write a single NWBFile object to disk."""
    with pynwb.NWBHDF5IO(file, mode="w") as io:
        io.write(nwb_file)
    logger.info(f"Saved dataset to {file}.")


def _remove_unoccupied_tracks(ds: xr.Dataset):
    """Remove tracks that are completely unoccupied from the dataset.

    Parameters
    ----------
    ds
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.

    Returns
    -------
    xarray.Dataset
        The input dataset without the unoccupied tracks.

    """
    all_nan = ds.position.isnull().all(dim=["keypoint", "space", "time"])
    return ds.where(~all_nan, drop=True)
