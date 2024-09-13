"""Conversion functions from ``movement`` datasets to napari layers."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

# get logger
logger = logging.getLogger(__name__)


def _construct_properties_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Construct a properties DataFrame from a ``movement`` dataset."""
    return pd.DataFrame(
        {
            "individual": ds.coords["individuals"].values,
            "keypoint": ds.coords["keypoints"].values,
            "time": ds.coords["time"].values,
            "confidence": ds["confidence"].values.flatten(),
        }
    )


def poses_to_napari_tracks(ds: xr.Dataset) -> tuple[np.ndarray, pd.DataFrame]:
    """Convert poses dataset to napari Tracks array and properties.

    Parameters
    ----------
    ds : xr.Dataset
        ``movement`` dataset containing pose tracks, confidence scores,
        and associated metadata.

    Returns
    -------
    data : np.ndarray
        napari Tracks array with shape (N, 4),
        where N is n_keypoints * n_individuals * n_frames
        and the 4 columns are (track_id, frame_idx, y, x).
    properties : pd.DataFrame
        DataFrame with properties (individual, keypoint, time, confidence).

    Notes
    -----
    A corresponding napari Points array can be derived from the Tracks array
    by taking its last 3 columns: (frame_idx, y, x). See the the documentation
    on the napari Tracks [1]_  and Points [2]_ layers.

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

    # assign unique integer ids to individuals and keypoints
    ds_.coords["individual_ids"] = ("individuals", range(n_individuals))
    ds_.coords["keypoint_ids"] = ("keypoints", range(n_keypoints))

    # Stack 3 dimensions into a new single dimension named "tracks"
    ds_ = ds_.stack(tracks=("individuals", "keypoints", "time"))
    # Track ids are unique ints (individual_id * n_keypoints + keypoint_id)
    individual_ids = ds_.coords["individual_ids"].values
    keypoint_ids = ds_.coords["keypoint_ids"].values
    track_ids = (individual_ids * n_keypoints + keypoint_ids).reshape(-1, 1)

    # Construct the napari Tracks array
    yx_columns = np.fliplr(ds_["position"].values.T)
    time_column = np.tile(range(n_frames), n_tracks).reshape(-1, 1)
    data = np.hstack((track_ids, time_column, yx_columns))

    # Construct the properties DataFrame
    properties = _construct_properties_dataframe(ds_)

    return data, properties