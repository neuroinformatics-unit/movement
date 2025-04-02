"""Save pose tracking data from ``movement`` to various file formats."""

import logging
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from movement.utils.logging import log_error
from movement.validators.datasets import ValidPosesDataset
from movement.validators.files import ValidFile

logger = logging.getLogger(__name__)


def _ds_to_dlc_style_df(
    ds: xr.Dataset, columns: pd.MultiIndex
) -> pd.DataFrame:
    """Convert a ``movement`` dataset to a DeepLabCut-style DataFrame.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.
    columns : pandas.MultiIndex
        DeepLabCut-style multi-index columns

    Returns
    -------
    pandas.DataFrame

    """
    # Concatenate the pose tracks and confidence scores into one array
    tracks_with_scores = np.concatenate(
        (
            ds.position.data,
            ds.confidence.data[:, np.newaxis, ...],
        ),
        axis=1,
    )
    # Reverse the order of the dimensions except for the time dimension
    transpose_order = [0] + list(range(tracks_with_scores.ndim - 1, 0, -1))
    tracks_with_scores = tracks_with_scores.transpose(transpose_order)
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
    ds: xr.Dataset, split_individuals: bool = False
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
        DataFrame for all individuals (see Notes).

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
    to_dlc_file : Save dataset directly to a DeepLabCut-style .h5 or .csv file.

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


def to_dlc_file(
    ds: xr.Dataset,
    file_path: str | Path,
    split_individuals: bool | Literal["auto"] = "auto",
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
        Whether to save individuals to separate files or to the same file
        (see Notes). Defaults to "auto".

    Notes
    -----
    If ``split_individuals`` is True, each individual will be saved to a
    separate file, formatted as in a single-animal DeepLabCut project
    (without the "individuals" column level). The individual's name will be
    appended to the file path, just before the file extension, e.g.
    "/path/to/filename_individual1.h5". If False, all individuals will be
    saved to the same file, formatted as in a multi-animal DeepLabCut project
    (with the "individuals" column level). The file path will not be modified.
    If "auto", the argument's value is determined based on the number of
    individuals in the dataset: True if there is only one, False otherwise.

    See Also
    --------
    to_dlc_style_df : Convert dataset to DeepLabCut-style DataFrame(s).

    Examples
    --------
    >>> from movement.io import save_dataset, load_dataset
    >>> ds = load_dataset.from_sleap_file("/path/to/file_sleap.analysis.h5")
    >>> save_dataset.to_dlc_file(ds, "/path/to/file_dlc.h5")

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
        df_dict = to_dlc_style_df(ds, split_individuals=True)

        for key, df in df_dict.items():
            # the key is the individual's name
            filepath = f"{file.path.with_suffix('')}_{key}{file.path.suffix}"
            if isinstance(df, pd.DataFrame):
                _save_dlc_df(Path(filepath), df)
            logger.info(f"Saved poses for individual {key} to {file.path}.")
    else:
        # convert the dataset to a single dataframe for all individuals
        df_all = to_dlc_style_df(ds, split_individuals=False)
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
    this function calls :func:`movement.io.save_dataset.to_dlc_file`
    with ``split_individuals=True``. This setting means that each individual
    is saved to a separate file, with the individual's name appended to the
    file path, just before the file extension,
    i.e. "/path/to/filename_individual1.csv".

    See Also
    --------
    to_dlc_file : Save dataset to a DeepLabCut-style file.

    Examples
    --------
    >>> from movement.io import save_dataset, load_dataset
    >>> ds = load_dataset.from_dlc_file("path/to/file.h5")
    >>> save_dataset.to_lp_file(ds, "path/to/file.csv")

    """
    file = _validate_file_path(file_path, expected_suffix=[".csv"])
    to_dlc_file(ds, file.path, split_individuals=True)


def to_sleap_analysis_file(
    ds: xr.Dataset,
    file_path: str | Path,
) -> None:
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
    The SLEAP analysis file format is a .h5 file with the following structure:
    - "tracks": array of shape (n_frames, n_individuals, n_keypoints, 2)
      containing the pose tracks.
    - "point_scores": array of shape (n_frames, n_individuals, n_keypoints)
      containing the point-wise confidence scores.
    - "track_occupancy": array of shape (n_frames, n_individuals) containing
      the track occupancy (1 if the track is present, 0 otherwise).

    See Also
    --------
    to_dlc_file : Save dataset to a DeepLabCut-style file.

    Examples
    --------
    >>> from movement.io import save_dataset, load_dataset
    >>> ds = load_dataset.from_dlc_file("path/to/file.h5")
    >>> save_dataset.to_sleap_analysis_file(ds, "path/to/file.analysis.h5")

    """
    file = _validate_file_path(file_path, expected_suffix=[".h5"])
    _validate_dataset(ds)

    # reshape the data into (n_frames, n_individuals, n_keypoints, 2)
    tracks = ds.position.data.transpose(0, 3, 2, 1)
    scores = ds.confidence.data.transpose(0, 2, 1)

    # create track occupancy array
    track_occupancy = np.ones(
        (ds.sizes["time"], ds.sizes["individuals"]), dtype=bool
    )

    with h5py.File(file.path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("point_scores", data=scores)
        f.create_dataset("track_occupancy", data=track_occupancy)

    logger.info(f"Saved poses dataset to {file.path}.")


def _validate_dataset(ds: xr.Dataset) -> None:
    """Validate a ``movement`` dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.

    Raises
    ------
    ValueError
        If the dataset is not valid.

    """
    ValidPosesDataset(
        position_array=ds.position.data,
        confidence_array=ds.confidence.data,
        individual_names=ds.coords["individuals"].data.tolist(),
        keypoint_names=ds.coords["keypoints"].data.tolist(),
        fps=ds.attrs.get("fps"),
        source_software=ds.attrs.get("source_software"),
    )


def _validate_file_path(
    file_path: str | Path, expected_suffix: list[str]
) -> ValidFile:
    """Validate a file path.

    Parameters
    ----------
    file_path : pathlib.Path or str
        Path to the file.
    expected_suffix : list of str
        List of expected file extensions.

    Returns
    -------
    movement.validators.files.ValidFile
        Validated file path.

    Raises
    ------
    ValueError
        If the file path is not valid.

    """
    file = Path(file_path)
    if file.suffix not in expected_suffix:
        raise log_error(
            ValueError,
            f"Expected file extension to be one of {expected_suffix}, but got "
            f"{file.suffix}.",
        )
    return ValidFile(file)
