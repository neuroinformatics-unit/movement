"""Conversion functions from ``movement`` datasets to napari layers."""

import numpy as np
import pandas as pd
import xarray as xr

# get logger


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


def ds_to_napari_shapes(
    ds: xr.Dataset,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Convert ``movement`` dataset to napari Shapes array and properties.

    Parameters
    ----------
    ds : xr.Dataset
        ``movement`` dataset containing bounding box tracks,
        confidence scores, and associated metadata.

    Returns
    -------
    data : np.ndarray
        napari Shapes array with shape (N, 4, 4),
        where N is n_individuals * n_frames
        and each (4, 4) entry is a matrix of 4 rows (1 per corner vertex)
        with the columns (track_id, frame_id, y, x).
    properties : pd.DataFrame
        DataFrame with properties (individual, time).

    Notes
    -----
    A corresponding napari Shapes array can be derived from the Tracks array
    by taking its last 5 columns: (frame_idx, c_y, c_x, height, width).
    See the documentation
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
    hw_cols = np.transpose(
        ds.shape.values,  # from: frames, xy, keypoints, individuals
        axes_reordering,  # to: individuals, keypoints, frames, xy
    ).reshape(-1, 2)[:, [1, 0]]  # swap x and y columns

    # Each keypoint of each individual is a separate track
    track_id_col = np.repeat(np.arange(n_tracks), n_frames).reshape(-1, 1)
    time_col = np.tile(np.arange(n_frames), (n_tracks)).reshape(-1, 1)
    data = np.hstack((track_id_col, time_col, yx_cols, hw_cols))

    # Repeat certain entries that correspond to multiple frames in
    # the data, so that one 4x3 array = 1 frame for 1 individual.
    # Assume the last time entry corresponds to one frame.

    # This block of code was originally intended to be used to
    # duplicate frames when each entry in the data array corresponds
    # to more than one frame.
    # If we should proceed with implementing this feature, it should
    # be reworked to play nicely with the fps widget.
    # If we shouldn't, it should be deleted before this is merged.
    """
    frames = ds.time.values - min(ds.time.values)   # frame index of each entry
    repeats = np.array(   #number of frames each entry should last for
        [frames[i+1]-frames[i] for i in range(0,len(frames)-1)]
        + [1]
    )
    repeat_idx = time_col - min(time_col) # ensure min idx is 0
    repeats_col = repeats[repeat_idx] # number of times to repeat a row

    # Expand the time column to account for any newly-generated frames.
    full_time_col = np.concatenate([
        np.arange(i,j)
        for i,j in zip(
            frames[repeat_idx],frames[repeat_idx]+repeats[repeat_idx]
        )
    ]).reshape(-1,1)
    """
    # Convert centroid, height/width representation
    # to corner vertex representation.
    # Start with getting extents.
    # need to make sure repeats_col is flat
    # repeats_flat = repeats_col.flatten().astype('int64')
    repeats_flat = np.ones(shape=data.shape[0]).astype("int64")  # temp
    min_y = np.repeat(
        (yx_cols[:, 0] - (hw_cols[:, 0] / 2)), repeats_flat
    ).reshape(-1, 1)
    max_y = np.repeat(
        (yx_cols[:, 0] + (hw_cols[:, 0] / 2)), repeats_flat
    ).reshape(-1, 1)
    min_x = np.repeat(
        (yx_cols[:, 1] - (hw_cols[:, 1] / 2)), repeats_flat
    ).reshape(-1, 1)
    max_x = np.repeat(
        (yx_cols[:, 1] + (hw_cols[:, 1] / 2)), repeats_flat
    ).reshape(-1, 1)

    # Convert extents to corner vertex representation.
    # full_track_id_col = np.repeat(track_id_col,repeats_flat).reshape(-1,1)
    full_track_id_col = track_id_col  # temp
    full_time_col = time_col  # temp
    ll = np.concatenate(
        (full_track_id_col, full_time_col, min_y, min_x), axis=1
    )
    ul = np.concatenate(
        (full_track_id_col, full_time_col, max_y, min_x), axis=1
    )
    ur = np.concatenate(
        (full_track_id_col, full_time_col, max_y, max_x), axis=1
    )
    lr = np.concatenate(
        (full_track_id_col, full_time_col, min_y, max_x), axis=1
    )

    data = np.array([ll, ul, ur, lr])
    data = np.moveaxis(data, (0, 1, 2), (1, 0, 2))
    # track id requires repeats

    # Construct the properties DataFrame
    # Stack individuals, time and keypoints (if present) dimensions
    # into a new single dimension named "tracks"
    dimensions_to_stack: tuple[str, ...] = ("individuals", "time")
    if "keypoints" in ds.coords:
        dimensions_to_stack += ("keypoints",)  # add last
    ds_ = ds.stack(tracks=sorted(dimensions_to_stack))

    properties = _construct_properties_dataframe(ds_)

    # Repeat rows of properties array to match newly-added frames
    # properties = properties.loc[properties.index.repeat(repeats_flat)]\
    #    .reset_index(drop=True)

    return data, properties
