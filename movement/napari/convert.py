"""Conversion functions from ``movement`` datasets to napari layers."""

import numpy as np
import pandas as pd
import xarray as xr


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


def _construct_track_and_time_cols(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute napari track_id and time columns from a ``movement`` dataset."""
    n_frames = ds.sizes["time"]
    n_individuals = ds.sizes["individuals"]
    n_keypoints = ds.sizes.get("keypoints", 1)
    n_tracks = n_individuals * n_keypoints

    # Each keypoint of each individual is a separate track
    track_id_col = np.repeat(np.arange(n_tracks), n_frames).reshape(-1, 1)
    time_col = np.tile(np.arange(n_frames), (n_tracks)).reshape(-1, 1)

    return track_id_col, time_col


def ds_to_napari_layers(
    ds: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Convert ``movement`` dataset to napari Tracks array and properties.

    Parameters
    ----------
    ds : xr.Dataset
        ``movement`` dataset containing pose or bounding box tracks,
        confidence scores, and associated metadata.

    Returns
    -------
    points_as_napari : np.ndarray
        position data as a napari Tracks array with shape (N, 4),
        where N is n_keypoints * n_individuals * n_frames
        and the 4 columns are (track_id, frame_idx, y, x).
    bboxes_as_napari : np.ndarray | None
        bounding box data as a napari Shapes array with shape (N, 4, 4),
        where N is n_individuals * n_frames and each (4, 4) entry is
        a matrix of 4 rows (1 per corner vertex, starting from upper left
        and progressing in counterclockwise order) with the columns
        (track_id, frame, y, x). Returns None when the input dataset doesn't
        have a "shape" variable.
    properties : pd.DataFrame
        DataFrame with properties (individual, keypoint, time, confidence)
        for use with napari layers.

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
    # Construct the track_ID and time columns for the napari Tracks array
    track_id_col, time_col = _construct_track_and_time_cols(ds)

    # Reorder axes to (individuals, keypoints, frames, xy)
    axes_reordering: tuple[int, ...] = (2, 0, 1)
    if "keypoints" in ds.coords:
        axes_reordering = (3,) + axes_reordering
    yx_cols = np.transpose(
        ds.position.values,  # from: frames, xy, keypoints, individuals
        axes_reordering,  # to: individuals, keypoints, frames, xy
    ).reshape(-1, 2)[:, [1, 0]]  # swap x and y columns

    points_as_napari = np.hstack((track_id_col, time_col, yx_cols))
    bboxes_as_napari = None

    # Construct the napari Shapes array if the input dataset is a
    # bounding boxes one
    if ds.ds_type == "bboxes":
        # Compute bbox corners
        xmin_ymin = ds.position - (ds.shape / 2)
        xmax_ymax = ds.position + (ds.shape / 2)

        # initialise xmax, ymin corner as xmin, ymin
        xmax_ymin = xmin_ymin.copy()
        # overwrite its x coordinate to xmax
        xmax_ymin.loc[{"space": "x"}] = xmax_ymax.loc[{"space": "x"}]

        # initialise xmin, ymin corner as xmin, ymin
        xmin_ymax = xmin_ymin.copy()
        # overwrite its y coordinate to ymax
        xmin_ymax.loc[{"space": "y"}] = xmax_ymax.loc[{"space": "y"}]

        # Add track_id and time columns to each corner array
        corner_arrays_with_track_id_and_time = [
            np.c_[
                track_id_col,
                time_col,
                np.transpose(corner.values, axes_reordering).reshape(-1, 2),
            ]
            for corner in [xmin_ymin, xmin_ymax, xmax_ymax, xmax_ymin]
        ]

        # Concatenate corner arrays along columns
        corners_array = np.concatenate(
            corner_arrays_with_track_id_and_time, axis=1
        )

        # Reshape to napari expected format
        # goes through corners counterclockwise from xmin_ymin
        # in image coordinates
        corners_array = corners_array.reshape(
            -1, 4, 4
        )  # last dimension: track_id, time, x, y
        bboxes_as_napari = corners_array[
            :, :, [0, 1, 3, 2]
        ]  # swap x and y columns

    # Construct the properties DataFrame
    # Stack individuals, time and keypoints (if present) dimensions
    # into a new single dimension named "tracks"
    dimensions_to_stack: tuple[str, ...] = ("individuals", "time")
    if "keypoints" in ds.coords:
        dimensions_to_stack += ("keypoints",)  # add last
    ds_ = ds.stack(tracks=sorted(dimensions_to_stack))

    properties = _construct_properties_dataframe(ds_)

    return points_as_napari, bboxes_as_napari, properties
