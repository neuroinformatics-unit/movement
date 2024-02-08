import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# get logger
logger = logging.getLogger(__name__)


def ds_to_napari_tracks(
    ds: xr.Dataset,
) -> tuple[np.ndarray[Any, Any], pd.DataFrame]:
    """Converts movement xarray dataset to a napari Tracks layers.

    Napari tracks arrays are 2D arrays with shape (N, 4), where N is
    n_keypoints * n_individuals * n_frames and the 4 columns are
    (track_id, frame_idx, y, x). The track_id is a unique integer
    that identifies the individual and keypoint. The frame_id is the frame
    number or timepoint in sec. The y and x coordinates are in pixels.

    From the above, a corresponding napari Points array can be derived
    by taking the last 3 columns of the napari Tracks array (frame_idx, y, x).

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset with pose tracks and confidence data variables.

    Returns
    -------
    napari_tracks, properties : tuple
        A tuple containing the napari tracks array and the properties
        DataFrame.

    """
    # Copy the dataset to avoid modifying the original
    ds_ = ds.copy()

    n_frames = ds_.sizes["time"]
    n_individuals = ds_.sizes["individuals"]
    n_keypoints = ds_.sizes["keypoints"]
    n_tracks = n_individuals * n_keypoints

    # if NaN values in data variable, log a warning and replace with zeros
    for data_var in ["confidence"]:
        if ds_[data_var].isnull().any():
            logger.warning(
                f"NaNs found in {data_var}, will be replaced with zeros."
            )
            ds_[data_var] = ds_[data_var].fillna(0)

    # Assign unique integer ids to individuals and keypoints
    ds_.coords["individual_ids"] = ("individuals", range(n_individuals))
    ds_.coords["keypoint_ids"] = ("keypoints", range(n_keypoints))

    # Convert 4D to 2D array by stacking
    ds_ = ds_.stack(tracks=("individuals", "keypoints", "time"))
    # Track ids are unique ints (individual_id * n_keypoints + keypoint_id)
    individual_ids = ds_.coords["individual_ids"].values
    keypoint_ids = ds_.coords["keypoint_ids"].values
    track_ids = individual_ids * n_keypoints + keypoint_ids

    # Construct the napari Tracks array
    yx_columns = np.fliplr(ds_["pose_tracks"].values.T)
    time_column = np.tile(np.arange(n_frames), n_tracks)
    napari_tracks = np.hstack(
        (track_ids.reshape(-1, 1), time_column.reshape(-1, 1), yx_columns)
    )

    n_rows = napari_tracks.shape[0]
    for col in "individuals", "keypoints", "time":
        assert n_rows == ds_.coords[col].size

    # Construct pandas DataFrame with properties
    properties = pd.DataFrame(
        {
            "individual": ds_.coords["individuals"].values,
            "keypoint": ds_.coords["keypoints"].values,
            "time": ds_.coords["time"].values,
            "confidence": ds_["confidence"].values.flatten(),
        }
    )

    return napari_tracks, properties
