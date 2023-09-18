import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from movement.io.validators import ValidFile

logger = logging.getLogger(__name__)


def to_dlc_df(ds: xr.Dataset) -> pd.DataFrame:
    """Convert an xarray dataset containing pose tracks into a
    DeepLabCut-style pandas DataFrame with multi-index columns.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing pose tracks, confidence scores, and metadata.

    Returns
    -------
    pandas DataFrame

    Notes
    -----
    The DataFrame will have a multi-index column with the following levels:
    "scorer", "individuals", "bodyparts", "coords" (even if there is only
    one individual present). Regardless of the provenance of the
    points-wise confidence scores, they will be referred to as
    "likelihood", and stored in the "coords" level (as DeepLabCut expects).

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

    # Concatenate the pose tracks and confidence scores into one array
    tracks_with_scores = np.concatenate(
        (
            ds.pose_tracks.data,
            ds.confidence.data[..., np.newaxis],
        ),
        axis=-1,
    )

    # Create the DLC-style multi-index columns
    # Use the DLC terminology: scorer, individuals, bodyparts, coords
    scorer = ["movement"]
    individuals = ds.coords["individuals"].data.tolist()
    bodyparts = ds.coords["keypoints"].data.tolist()
    # The confidence scores in DLC are referred to as "likelihood"
    coords = ds.coords["space"].data.tolist() + ["likelihood"]

    index_levels = ["scorer", "individuals", "bodyparts", "coords"]
    columns = pd.MultiIndex.from_product(
        [scorer, individuals, bodyparts, coords], names=index_levels
    )
    df = pd.DataFrame(
        data=tracks_with_scores.reshape(ds.dims["time"], -1),
        index=np.arange(ds.dims["time"], dtype=int),
        columns=columns,
        dtype=float,
    )
    logger.info("Converted PoseTracks dataset to DLC-style DataFrame.")
    return df


def to_dlc_file(ds: xr.Dataset, file_path: Union[str, Path]) -> None:
    """Save the xarray dataset containing pose tracks to a
    DeepLabCut-style ".h5" or ".csv" file.

    Parameters
    ----------
    ds : xarray Dataset
        Dataset containing pose tracks, confidence scores, and metadata.
    file_path : pathlib Path or str
        Path to the file to save the DLC poses to. The file extension
        must be either ".h5" (recommended) or ".csv".

    See Also
    --------
    to_dlc_df : Convert an xarray dataset containing pose tracks into a
        DeepLabCut-style pandas DataFrame with multi-index columns.
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

    df = to_dlc_df(ds)  # convert to pandas DataFrame
    if file.path.suffix == ".csv":
        df.to_csv(file.path, sep=",")
    else:  # file.path.suffix == ".h5"
        df.to_hdf(file.path, key="df_with_missing")
    logger.info(f"Saved PoseTracks dataset to {file.path}.")
