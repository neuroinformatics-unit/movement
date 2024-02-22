import logging
from pathlib import Path
from typing import Literal, Union

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from movement.io.validators import ValidFile
from movement.logging import log_error

logger = logging.getLogger(__name__)


def _xarray_to_dlc_df(ds: xr.Dataset, columns: pd.MultiIndex) -> pd.DataFrame:
    """Takes an xarray dataset and DLC-style multi-index columns and outputs
    a pandas dataframe.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    columns : pandas.MultiIndex
        DLC-style multi-index columns

    Returns
    -------
    pandas.DataFrame
    """

    # Concatenate the pose tracks and confidence scores into one array
    tracks_with_scores = np.concatenate(
        (
            ds.pose_tracks.data,
            ds.confidence.data[..., np.newaxis],
        ),
        axis=-1,
    )

    # Create DataFrame with multi-index columns
    df = pd.DataFrame(
        data=tracks_with_scores.reshape(ds.dims["time"], -1),
        index=np.arange(ds.dims["time"], dtype=int),
        columns=columns,
        dtype=float,
    )

    return df


def _auto_split_individuals(ds: xr.Dataset) -> bool:
    """Returns True if there is only one individual in the dataset,
    else returns False."""

    n_individuals = ds.sizes["individuals"]
    return True if n_individuals == 1 else False


def _save_dlc_df(filepath: Path, df: pd.DataFrame) -> None:
    """Given a filepath, will save the dataframe as either a .h5 or .csv.

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


def to_dlc_df(
    ds: xr.Dataset, split_individuals: bool = False
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Convert an xarray dataset containing pose tracks into a single
    DeepLabCut-style pandas DataFrame or a dictionary of DataFrames
    per individual, depending on the 'split_individuals' argument.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    split_individuals : bool, optional
        If True, return a dictionary of pandas DataFrames per individual,
        with individual names as keys and DataFrames as values.
        If False, return a single pandas DataFrame for all individuals.
        Default is False.

    Returns
    -------
    pandas.DataFrame or dict
        DeepLabCut-style pandas DataFrame or dictionary of DataFrames.

    Notes
    -----
    The DataFrame(s) will have a multi-index column with the following levels:
    "scorer", "bodyparts", "coords" (if split_individuals is True),
    or "scorer", "individuals", "bodyparts", "coords"
    (if split_individuals is False).

    Regardless of the provenance of the points-wise confidence scores,
    they will be referred to as "likelihood", and stored in
    the "coords" level (as DeepLabCut expects).

    See Also
    --------
    to_dlc_file : Save the xarray dataset containing pose tracks directly
        to a DeepLabCut-style .h5 or .csv file.
    """

    _validate_dataset(ds)
    scorer = ["movement"]
    bodyparts = ds.coords["keypoints"].data.tolist()
    coords = ds.coords["space"].data.tolist() + ["likelihood"]
    individuals = ds.coords["individuals"].data.tolist()

    if split_individuals:
        df_dict = {}

        for individual in individuals:
            individual_data = ds.sel(individuals=individual)

            index_levels = ["scorer", "bodyparts", "coords"]
            columns = pd.MultiIndex.from_product(
                [scorer, bodyparts, coords], names=index_levels
            )

            df = _xarray_to_dlc_df(individual_data, columns)
            df_dict[individual] = df

        logger.info(
            "Converted PoseTracks dataset to DeepLabCut-style DataFrames "
            "per individual."
        )
        return df_dict
    else:
        index_levels = ["scorer", "individuals", "bodyparts", "coords"]
        columns = pd.MultiIndex.from_product(
            [scorer, individuals, bodyparts, coords], names=index_levels
        )

        df_all = _xarray_to_dlc_df(ds, columns)

        logger.info("Converted PoseTracks dataset to DLC-style DataFrame.")
        return df_all


def to_dlc_file(
    ds: xr.Dataset,
    file_path: Union[str, Path],
    split_individuals: Union[bool, Literal["auto"]] = "auto",
) -> None:
    """Save the xarray dataset containing pose tracks to a
    DeepLabCut-style .h5 or .csv file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    file_path : pathlib.Path or str
        Path to the file to save the DLC poses to. The file extension
        must be either .h5 (recommended) or .csv.
    split_individuals : bool or "auto", optional
        Whether to save individuals to separate files or to the same file.\n
        If True, each individual will be saved to a separate file,
        formatted as in a single-animal DeepLabCut project - i.e. without
        the "individuals" column level. The individual's name will be appended
        to the file path, just before the file extension, i.e.
        "/path/to/filename_individual1.h5".\n
        If False, all individuals will be saved to the same file,
        formatted as in a multi-animal DeepLabCut project - i.e. the columns
        will include the "individuals" level. The file path will not be
        modified.\n
        If "auto" the argument's value is determined based on the number of
        individuals in the dataset: True if there is only one, and
        False if there are more than one. This is the default.

    See Also
    --------
    to_dlc_df : Convert an xarray dataset containing pose tracks into a single
        DeepLabCut-style pandas DataFrame or a dictionary of DataFrames
        per individual.

    Examples
    --------
    >>> from movement.io import save_poses, load_poses
    >>> ds = load_poses.from_sleap_file("/path/to/file_sleap.analysis.h5")
    >>> save_poses.to_dlc_file(ds, "/path/to/file_dlc.h5")
    """

    file = _validate_file_path(file_path, expected_suffix=[".csv", ".h5"])

    # Sets default behaviour for the function
    if split_individuals == "auto":
        split_individuals = _auto_split_individuals(ds)

    elif not isinstance(split_individuals, bool):
        raise log_error(
            ValueError,
            "Expected 'split_individuals' to be a boolean or 'auto', but got "
            f"{type(split_individuals)}.",
        )

    if split_individuals:
        # split the dataset into a dictionary of dataframes per individual
        df_dict = to_dlc_df(ds, split_individuals=True)

        for key, df in df_dict.items():
            # the key is the individual's name
            filepath = f"{file.path.with_suffix('')}_{key}{file.path.suffix}"
            if isinstance(df, pd.DataFrame):
                _save_dlc_df(Path(filepath), df)
            logger.info(
                f"Saved PoseTracks data for individual {key} to {file.path}."
            )
    else:
        # convert the dataset to a single dataframe for all individuals
        df_all = to_dlc_df(ds, split_individuals=False)
        if isinstance(df_all, pd.DataFrame):
            _save_dlc_df(file.path, df_all)
        logger.info(f"Saved PoseTracks dataset to {file.path}.")


def to_lp_file(
    ds: xr.Dataset,
    file_path: Union[str, Path],
) -> None:
    """Save the xarray dataset containing pose tracks to a LightningPose-style
    .csv file. See Notes for more details.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    file_path : pathlib.Path or str
        Path to the .csv file to save the poses to.

    Notes
    -----
    LightningPose saves pose estimation outputs as .csv files, using the same
    format as single-animal DeepLabCut projects. Therefore, under the hood,
    this function calls ``to_dlc_file`` with ``split_individuals=True``. This
    setting means that each individual is saved to a separate file, with
    the individual's name appended to the file path, just before the file
    extension, i.e. "/path/to/filename_individual1.csv".

    See Also
    --------
    to_dlc_file : Save the xarray dataset containing pose tracks to a
        DeepLabCut-style .h5 or .csv file.
    """

    file = _validate_file_path(file_path=file_path, expected_suffix=[".csv"])
    _validate_dataset(ds)
    to_dlc_file(ds, file.path, split_individuals=True)


def to_sleap_analysis_file(
    ds: xr.Dataset, file_path: Union[str, Path]
) -> None:
    """Save the xarray dataset containing pose tracks to a SLEAP-style
    .h5 analysis file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    file_path : pathlib.Path or str
        Path to the file to save the poses to. The file extension must be .h5.

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
    .. [1] https://sleap.ai/api/sleap.info.write_tracking_h5.html

    Examples
    --------
    >>> from movement.io import save_poses, load_poses
    >>> ds = load_poses.from_dlc_file("path/to/file.h5")
    >>> save_poses.to_sleap_analysis_file(
    ...     ds, "/path/to/file_sleap.analysis.h5"
    ... )
    """

    file = _validate_file_path(file_path=file_path, expected_suffix=[".h5"])
    _validate_dataset(ds)

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
    if ds.fps is not None:
        frame_idxs = np.rint(ds.time.values * ds.fps).astype(int).tolist()
    else:
        frame_idxs = ds.time.values.astype(int).tolist()
    n_frames = frame_idxs[-1] - frame_idxs[0] + 1
    pos_x = ds.pose_tracks.sel(space="x").values
    # Mask denoting which individuals are present in each frame
    track_occupancy = (~np.all(np.isnan(pos_x), axis=2)).astype(int)
    tracks = np.transpose(ds.pose_tracks.data, (1, 3, 2, 0))
    point_scores = np.transpose(ds.confidence.data, (1, 2, 0))
    instance_scores = np.full((n_individuals, n_frames), np.nan, dtype=float)
    tracking_scores = np.full((n_individuals, n_frames), np.nan, dtype=float)
    labels_path = (
        ds.source_file if Path(ds.source_file).suffix == ".slp" else ""
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
    logger.info(f"Saved PoseTracks dataset to {file.path}.")


def _remove_unoccupied_tracks(ds: xr.Dataset):
    """Remove tracks that are completely unoccupied in the xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing pose tracks, confidence scores, and metadata.

    Returns
    -------
    xarray.Dataset
        The input dataset without the unoccupied tracks.
    """

    all_nan = ds.pose_tracks.isnull().all(dim=["keypoints", "space", "time"])
    return ds.where(~all_nan, drop=True)


def _validate_file_path(
    file_path: Union[str, Path], expected_suffix: list[str]
) -> ValidFile:
    """Validate the input file path by checking that the file has
    write permission and expected suffix(es). If the file is not valid,
    an appropriate error is raised.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file to validate.
    expected_suffix : list of str
        Expected suffix(es) for the file.

    Returns
    -------
    ValidFile
        The validated file.

    Raises
    ------
    OSError
        If the file cannot be written.
    ValueError
        If the file does not have the expected suffix.
    """

    try:
        file = ValidFile(
            file_path,
            expected_permission="w",
            expected_suffix=expected_suffix,
        )
    except (OSError, ValueError) as error:
        logger.error(error)
        raise error
    return file


def _validate_dataset(ds: xr.Dataset) -> None:
    """Validate the input dataset is an xarray Dataset with valid PoseTracks.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to validate.

    Raises
    ------
    ValueError
        If `ds` is not an xarray Dataset with valid PoseTracks.
    """

    if not isinstance(ds, xr.Dataset):
        raise log_error(
            ValueError, f"Expected an xarray Dataset, but got {type(ds)}."
        )
    ds.move.validate()  # validate the dataset
