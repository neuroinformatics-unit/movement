import logging
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import xarray as xr

from movement.io.validators import ValidFile
from movement.logging import log_error

logger = logging.getLogger(__name__)


def _xarry_to_dlc_df(ds: xr.Dataset, columns: pd.MultiIndex) -> pd.DataFrame:
    """Takes an xarray dataset and DLC-style multi-index columns and outputs
    a pandas dataframe.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    columns : pd.MultiIndex
        DLC-style multi-index columns
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


def _auto_split_individuals(ds: xr.Dataset):
    """Returns True if there is only one individual in the dataset,
    else returns False.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    """

    individuals = ds.coords["individuals"].data.tolist()
    return True if len(individuals) == 1 else False


def _save_dlc_df(filepath: Path, dataframe: pd.DataFrame):
    """Given a filepath, will save the dataframe as either a .h5 or .csv.

    Parameters
    ----------
    suffix : os.PathLike
        Suffix of path provided.
    filepath : os.PathLike
        Path of where to save data.
    suffix : pd.DataFrame
        Pandas Dataframe to save to .csv or .h5.
    """

    if filepath.suffix == ".csv":
        dataframe.to_csv(filepath, sep=",")
    elif filepath.suffix == ".h5":
        dataframe.to_hdf(filepath, key="df_with_missing")
    # for invalid suffix
    else:
        log_error(ValueError, "Expected filepath to end in .csv or .h5.")


def to_dlc_df(
    ds: xr.Dataset, split_individuals: bool = True
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Convert an xarray dataset containing pose tracks into a DeepLabCut-style
    pandas DataFrame with multi-index columns for each individual or a
    dictionary of DataFrames for each individual based on the
    'split_individuals' argument.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    split_individuals : bool, optional
        If True, return a dictionary of pandas DataFrames, for each individual.
        If False, return a single pandas DataFrame with multi-index columns
        for all individuals.
        Default is True.

    Returns
    -------
    pandas DataFrame or dict
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
        to a DeepLabCut-style ".h5" or ".csv" file.
    """
    if not isinstance(ds, xr.Dataset):
        log_error(
            ValueError, f"Expected an xarray Dataset, but got {type(ds)}."
        )

    ds.poses.validate()  # validate the dataset

    scorer = ["movement"]
    bodyparts = ds.coords["keypoints"].data.tolist()
    coords = ds.coords["space"].data.tolist() + ["likelihood"]
    individuals = ds.coords["individuals"].data.tolist()

    if split_individuals:
        result = {}

        for individual in individuals:
            # Select data for the current individual
            individual_data = ds.sel(individuals=individual)

            # Create the DLC-style multi-index columns
            index_levels = ["scorer", "bodyparts", "coords"]
            columns = pd.MultiIndex.from_product(
                [scorer, bodyparts, coords], names=index_levels
            )

            # Uses the columns and data to make a df
            df = _xarry_to_dlc_df(individual_data, columns)

            """ Add the DataFrame to the result
            dictionary with individual's name as key """
            result[individual] = df

        logger.info(
            """Converted PoseTracks dataset to
            DLC-style DataFrames for each individual."""
        )
        return result
    else:
        # Create the DLC-style multi-index columns
        index_levels = ["scorer", "individuals", "bodyparts", "coords"]
        columns = pd.MultiIndex.from_product(
            [scorer, individuals, bodyparts, coords], names=index_levels
        )

        df = _xarry_to_dlc_df(ds, columns)

        logger.info("Converted PoseTracks dataset to DLC-style DataFrame.")
        return df


def to_dlc_file(
    ds: xr.Dataset,
    file_path: Union[str, Path],
    split_individuals: Union[bool, Literal["auto"]] = "auto",
) -> None:
    """Save the xarray dataset containing pose tracks to a
    DeepLabCut-style ".h5" or ".csv" file.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    file_path : pathlib Path or str
        Path to the file to save the DLC poses to. The file extension
        must be either ".h5" (recommended) or ".csv".
    split_individuals : bool, optional
        If True, the file will be formatted as in a single-animal
        DeepLabCut project: no "individuals" level, and each individual will be
        saved in a separate file. The individual's name will be appended to the
        file path, just before the file extension, i.e.
        "/path/to/filename_individual1.h5".
        If False, the file will be formatted as in a multi-animal
        DeepLabCut project: the columns will include the
        "individuals" level and all individuals will be saved to the same file.
        If "auto" the format will be determined based on the number of
        individuals in the dataset: True if there is only one, and
        False if there are more than one. This is the default.

    See Also
    --------
    to_dlc_df : Convert an xarray dataset containing pose tracks into a
        DeepLabCut-style pandas DataFrame with multi-index columns
        for each individual or a dictionary of DataFrames for each individual
        based on the 'split_individuals' argument.

    Examples
    --------
    >>> from movement.io import save_poses, load_poses
    >>> ds = load_poses.from_sleap("/path/to/file_sleap.analysis.h5")
    >>> save_poses.to_dlc_file(ds, "/path/to/file_dlc.h5")
    """

    try:
        file = ValidFile(
            file_path,
            expected_permission="w",
            expected_suffix=[".csv", ".h5"],
        )
    except (OSError, ValueError) as error:
        logger.error(error)
        raise error

    # Sets default behaviour for the function
    if split_individuals == "auto":
        split_individuals = _auto_split_individuals(ds)

    elif not isinstance(split_individuals, bool):
        raise log_error(
            ValueError,
            f"Expected 'split_individuals' to be a boolean or 'auto', but got "
            f"{type(split_individuals)}.",
        )

    if split_individuals:
        """If split_individuals is True then it will split the file into a
        dictionary of pandas dataframes for each individual."""

        df_dict = to_dlc_df(ds, split_individuals=True)

        for key, df in df_dict.items():
            """Iterates over dictionary, the key is the name of the
            individual and the value is the corresponding df."""
            filepath = f"{file.path.with_suffix('')}_{key}{file.path.suffix}"

            if isinstance(df, pd.DataFrame):
                _save_dlc_df(Path(filepath), df)

            logger.info(f"Saved PoseTracks dataset to {file.path}.")
    else:
        """If split_individuals is False then it will save the file as
        a dataframe with multi-index columns for each individual."""
        dataframe = to_dlc_df(ds, split_individuals=False)
        if isinstance(dataframe, pd.DataFrame):
            _save_dlc_df(file.path, dataframe)
        logger.info(f"Saved PoseTracks dataset to {file.path}.")
