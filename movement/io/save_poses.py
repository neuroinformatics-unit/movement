import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from movement.io.validators import ValidFile

logger = logging.getLogger(__name__)


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
    "scorer", "individuals", "bodyparts", "coords"
    (if multi_individual is True),
    or "scorer", "bodyparts", "coords" (if multi_individual is False).
    Regardless of the provenance of the points-wise confidence scores,
    they will be referred to as "likelihood", and stored in
    the "coords" level (as DeepLabCut expects).

    See Also
    --------
    to_dlc_file : Save the xarray dataset containing pose tracks directly
        to a DeepLabCut-style ".h5" or ".csv" file.
    """
    if not isinstance(ds, xr.Dataset):
        error_msg = f"Expected an xarray Dataset, but got {type(ds)}. "
        logger.error(error_msg)
        raise ValueError(error_msg)

    ds.poses.validate()  # validate the dataset

    scorer = ["movement"]
    bodyparts = ds.coords["keypoints"].data.tolist()
    coords = ds.coords["space"].data.tolist() + ["likelihood"]

    if split_individuals:
        individuals = ds.coords["individuals"].data.tolist()
        result = {}

        for individual in individuals:
            # Select data for the current individual
            individual_data = ds.sel(individuals=individual)

            # Concatenate the pose tracks and confidence scores into one array
            tracks_with_scores = np.concatenate(
                (
                    individual_data.pose_tracks.data,
                    individual_data.confidence.data[..., np.newaxis],
                ),
                axis=-1,
            )

            # Create the DLC-style multi-index columns
            index_levels = ["scorer", "bodyparts", "coords"]
            columns = pd.MultiIndex.from_product(
                [scorer, bodyparts, coords], names=index_levels
            )

            # Create DataFrame for the current individual
            df = pd.DataFrame(
                data=tracks_with_scores.reshape(
                    individual_data.dims["time"], -1
                ),
                index=np.arange(individual_data.dims["time"], dtype=int),
                columns=columns,
                dtype=float,
            )

            """ Add the DataFrame to the result
            dictionary with individual's name as key """
            result[individual] = df

        logger.info(
            """Converted PoseTracks dataset to
            DLC-style DataFrames for each individual."""
        )
        return result
    else:
        """Concatenate the pose tracks and
        confidence scores into one array for all individuals"""
        tracks_with_scores = np.concatenate(
            (
                ds.pose_tracks.data,
                ds.confidence.data[..., np.newaxis],
            ),
            axis=-1,
        )

        # Create the DLC-style multi-index columns
        index_levels = ["scorer", "individuals", "bodyparts", "coords"]
        individuals = ds.coords["individuals"].data.tolist()
        columns = pd.MultiIndex.from_product(
            [scorer, individuals, bodyparts, coords], names=index_levels
        )

        """ Create a single DataFrame with
        multi-index columns for each individual """
        df = pd.DataFrame(
            data=tracks_with_scores.reshape(ds.dims["time"], -1),
            index=np.arange(ds.dims["time"], dtype=int),
            columns=columns,
            dtype=float,
        )

        logger.info("Converted PoseTracks dataset to DLC-style DataFrame.")
        return df


def to_dlc_file(
    ds: xr.Dataset,
    file_path: Union[str, Path],
    split_individuals: Union[bool, None] = None,
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
        Format of the DeepLabcut output file.
        - If True, the file will be formatted as in a single-animal
        DeepLabCut project: no "individuals" level, and each individual will be
        saved in a separate file. The individual's name will be appended to the
        file path, just before the file extension, i.e.
        "/path/to/filename_individual1.h5".
        - If False, the file will be formatted as in a multi-animal
        DeepLabCut project: the columns will include the
        "individuals" level and all individuals will be saved to the same file.
        - If "auto" the format will be determined based on the number of
        individuals in the dataset: True if there are more than one, and
        False if there is only one. This is the default.

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
    if split_individuals is None:
        individuals = ds.coords["individuals"].data.tolist()
        if len(individuals) == 1:
            split_individuals = True
        else:
            split_individuals = False

    """If split_individuals is True then it will split the file into a
    dictionary of pandas dataframes for each individual."""
    if split_individuals:
        dfdict = to_dlc_df(ds, True)
        if file.path.suffix == ".csv":
            for (
                key,
                df,
            ) in dfdict.items():
                """Iterates over dictionary, the key is the name of the
                individual and the value is the corresponding df."""
                filepath = (
                    str(file.path.with_suffix("")) + "_" + str(key) + ".csv"
                )
                # Convert the string back to a PosixPath object
                filepath_posix = Path(filepath)
                df.to_csv(filepath_posix, sep=",")

        else:  # file.path.suffix == ".h5"
            for key, df in dfdict.items():
                filepath = (
                    str(file.path.with_suffix("")) + "_" + str(key) + ".h5"
                )
                # Convert the string back to a PosixPath object
                filepath_posix = Path(filepath)
                df.to_hdf(filepath, key="df_with_missing")

        logger.info(f"Saved PoseTracks dataset to {file.path}.")

    """If split_individuals is False then it will save the file as a dataframe
    with multi-index columns for each individual."""
    if not split_individuals:
        dataframe = to_dlc_df(ds, False)  # convert to pandas DataFrame
        if isinstance(dataframe, pd.DataFrame):  # checking it's a dataframe
            if file.path.suffix == ".csv":
                dataframe.to_csv(file.path, sep=",")
            else:  # file.path.suffix == ".h5"
                dataframe.to_hdf(file.path, key="df_with_missing")
        logger.info(f"Saved PoseTracks dataset to {file.path}.")
