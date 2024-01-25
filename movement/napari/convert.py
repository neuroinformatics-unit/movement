import logging
from typing import Any

import numpy as np
import xarray as xr

# get logger
logger = logging.getLogger(__name__)


def ds_to_napari_tracks(
    ds: xr.Dataset,
) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
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
        dictionary.

    """
    n_frames = ds.dims["time"]
    n_individuals = ds.dims["individuals"]
    n_keypoints = ds.dims["keypoints"]
    n_tracks = n_individuals * n_keypoints

    # Assign unique integer ids to individuals and keypoints
    ds.coords["individual_ids"] = ("individuals", range(n_individuals))
    ds.coords["keypoint_ids"] = ("keypoints", range(n_keypoints))

    # Convert 4D to 2D array by stacking
    ds = ds.stack(tracks=("individuals", "keypoints", "time"))
    # Track ids are unique ints (individual_id * n_keypoints + keypoint_id)
    individual_ids = ds.coords["individual_ids"].values
    keypoint_ids = ds.coords["keypoint_ids"].values
    track_ids = individual_ids * n_keypoints + keypoint_ids

    # Construct the napari Tracks array
    yx_columns = np.fliplr(ds["pose_tracks"].values.T)
    time_column = np.tile(np.arange(n_frames), n_tracks)
    napari_tracks = np.hstack(
        (track_ids.reshape(-1, 1), time_column.reshape(-1, 1), yx_columns)
    )

    properties = {
        "confidence": ds["confidence"].values.flatten(),
        "individual": individual_ids,
        "keypoint": keypoint_ids,
        "time": ds.coords["time"].values,
    }

    return napari_tracks, properties
