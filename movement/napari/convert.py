import logging

import numpy as np
import pandas as pd
import xarray as xr

# get logger
logger = logging.getLogger(__name__)


def _replace_nans_with_zeros(
    ds: xr.Dataset, data_vars: list[str]
) -> xr.Dataset:
    """Replace NaN values in specified data variables with zeros."""
    for data_var in data_vars:
        if ds[data_var].isnull().any():
            logger.warning(
                f"NaNs found in {data_var}, will be replaced with zeros."
            )
            ds[data_var] = ds[data_var].fillna(0)
    return ds


def _construct_properties_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Construct a pandas DataFrame with properties from the dataset."""
    properties = pd.DataFrame(
        {
            "individual": ds.coords["individuals"].values,
            "keypoint": ds.coords["keypoints"].values,
            "time": ds.coords["time"].values,
            "confidence": ds["confidence"].values.flatten(),
        }
    )
    return properties


def ds_to_napari_tracks(ds: xr.Dataset) -> tuple[np.ndarray, pd.DataFrame]:
    """Converts movement xarray dataset to napari tracks array and properties.
    For reference, see the napari tracks array documentation [1]_.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset with pose tracks and confidence data variables.

    Returns
    -------
    data : np.ndarray
        Napari Tracks array with shape (N, 4),
        where N is n_keypoints * n_individuals * n_frames
        and the 4 columns are (track_id, frame_idx, y, x).
    properties : pd.DataFrame
        DataFrame with properties (individual, keypoint, time, confidence).

    Notes
    -----
    A corresponding napari Points array can be derived from the Tracks array
    by taking its last 3 columns: (frame_idx, y, x). See the napari Points
    array documentation [2]_.

    References
    ----------
    .. [1] https://napari.org/stable/howtos/layers/tracks.html
    .. [2] https://napari.org/stable/howtos/layers/points.html

    """
    ds_ = ds.copy()  # make a copy to avoid modifying the original dataset

    n_frames = ds_.sizes["time"]
    n_individuals = ds_.sizes["individuals"]
    n_keypoints = ds_.sizes["keypoints"]
    n_tracks = n_individuals * n_keypoints

    ds_ = _replace_nans_with_zeros(ds_, ["confidence"])
    # assign unique integer ids to individuals and keypoints
    ds_.coords["individual_ids"] = ("individuals", range(n_individuals))
    ds_.coords["keypoint_ids"] = ("keypoints", range(n_keypoints))

    # Convert 4D to 2D array by stacking
    ds_ = ds_.stack(tracks=("individuals", "keypoints", "time"))
    # Track ids are unique ints (individual_id * n_keypoints + keypoint_id)
    individual_ids = ds_.coords["individual_ids"].values
    keypoint_ids = ds_.coords["keypoint_ids"].values
    track_ids = (individual_ids * n_keypoints + keypoint_ids).reshape(-1, 1)

    # Construct the napari Tracks array
    yx_columns = np.fliplr(ds_["pose_tracks"].values.T)
    time_column = np.tile(range(n_frames), n_tracks).reshape(-1, 1)
    data = np.hstack((track_ids, time_column, yx_columns))

    # Construct the properties DataFrame
    properties = _construct_properties_dataframe(ds_)

    return data, properties
