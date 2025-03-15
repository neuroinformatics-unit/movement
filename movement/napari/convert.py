"""Conversion functions from ``movement`` datasets to napari layers."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

# get logger
logger = logging.getLogger(__name__)


def _construct_properties_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Construct a properties DataFrame from a ``movement`` dataset."""
    data = {
        "individual": ds.coords["individuals"].values,
        "time": ds.coords["time"].values,
        "confidence": ds["confidence"].values.flatten(),
    }
    desired_order = list(data.keys())
    if "keypoints" in ds.coords:
        data["keypoint"] = ds.coords["keypoints"].values
        desired_order.insert(1, "keypoint")

    # sort
    return pd.DataFrame(data).reindex(columns=desired_order)


def ds_to_napari_tracks(
    ds: xr.Dataset,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Convert ``movement`` dataset to napari Tracks array and properties.

    Parameters
    ----------
    ds : xr.Dataset
        ``movement`` dataset containing pose or bounding box tracks,
        confidence scores, and associated metadata.

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
    by taking its last 3 columns: (frame_idx, y, x). See the documentation
    on the napari Tracks [1]_  and Points [2]_ layers.

    References
    ----------
    .. [1] https://napari.org/stable/howtos/layers/tracks.html
    .. [2] https://napari.org/stable/howtos/layers/points.html

    """
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individuals"]
    n_keypoints = ds.sizes.get("keypoints", 1)
    n_tracks = n_individuals * n_keypoints

    # Construct the napari Tracks array
    # Reorder axes to (individuals, keypoints, frames, xy)
    axes_reordering: tuple[int, ...] = (2, 0, 1)
    if "keypoints" in ds.coords:
        axes_reordering = (3,) + axes_reordering
    yx_cols = np.transpose(
        ds.position.values,  # from: frames, xy, keypoints, individuals
        axes_reordering,  # to: individuals, keypoints, frames, xy
    ).reshape(-1, 2)[:, [1, 0]]  # swap x and y columns

    # Each keypoint of each individual is a separate track
    track_id_col = np.repeat(np.arange(n_tracks), n_frames).reshape(-1, 1)
    time_col = np.tile(np.arange(n_frames), (n_tracks)).reshape(-1, 1)
    data = np.hstack((track_id_col, time_col, yx_cols))

    # Construct the properties DataFrame
    # Stack individuals, time and keypoints (if present) dimensions
    # into a new single dimension named "tracks"
    dimensions_to_stack: tuple[str, ...] = ("individuals", "time")
    if "keypoints" in ds.coords:
        dimensions_to_stack += ("keypoints",)  # add last
    ds_ = ds.stack(tracks=sorted(dimensions_to_stack))

    properties = _construct_properties_dataframe(ds_)

    return data, properties
